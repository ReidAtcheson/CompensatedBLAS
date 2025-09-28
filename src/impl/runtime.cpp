#include "compensated_blas.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <new>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include <memory_resource>

#include "impl/compensated_blas_backend_ilp64.hpp"
#include "impl/naive_blas_backend.hpp"

namespace compensated_blas::runtime {
namespace {

struct matrix_entry_t {
    deferred_rounding_matrix_t descriptor{};
    void *compensation = nullptr;
    std::size_t compensation_bytes = 0;
};

struct vector_entry_t {
    deferred_rounding_vector_t descriptor{};
    void *compensation = nullptr;
    std::size_t compensation_bytes = 0;
};

std::size_t matrix_element_count(const deferred_rounding_matrix_t &descriptor) {
    if (descriptor.rows == 0 || descriptor.cols == 0) {
        return 0;
    }
    if (descriptor.row_major) {
        return descriptor.rows * descriptor.leading_dimension;
    }
    return descriptor.cols * descriptor.leading_dimension;
}

std::size_t vector_element_count(const deferred_rounding_vector_t &descriptor) {
    if (descriptor.length == 0) {
        return 0;
    }
    if (descriptor.stride == 0) {
        return descriptor.length;
    }
    return 1 + (descriptor.length - 1) * descriptor.stride;
}

class descriptor_pool_t {
public:
    explicit descriptor_pool_t(std::pmr::memory_resource *resource)
        : resource_(resource ? resource : std::pmr::get_default_resource()),
          matrices_(0,
                    std::hash<const void *>(),
                    std::equal_to<const void *>(),
                    std::pmr::polymorphic_allocator<std::pair<const void *const, matrix_entry_t>>(resource_)),
          matrix_keys_(std::pmr::polymorphic_allocator<const void *>(resource_)),
          vectors_(0,
                   std::hash<const void *>(),
                   std::equal_to<const void *>(),
                   std::pmr::polymorphic_allocator<std::pair<const void *const, vector_entry_t>>(resource_)),
          vector_keys_(std::pmr::polymorphic_allocator<const void *>(resource_)) {}

    descriptor_pool_t(const descriptor_pool_t &) = delete;
    descriptor_pool_t &operator=(const descriptor_pool_t &) = delete;

    ~descriptor_pool_t() { clear(); }

    void reset_resource(std::pmr::memory_resource *resource) {
        clear();
        resource_ = resource ? resource : std::pmr::get_default_resource();
        decltype(matrices_) empty_matrices(0,
                                           std::hash<const void *>(),
                                           std::equal_to<const void *>(),
                                           std::pmr::polymorphic_allocator<std::pair<const void *const, matrix_entry_t>>(resource_));
        matrices_.swap(empty_matrices);
        auto empty_matrix_keys = std::pmr::vector<const void *>{
            std::pmr::polymorphic_allocator<const void *>(resource_)};
        matrix_keys_.swap(empty_matrix_keys);

        decltype(vectors_) empty_vectors(0,
                                         std::hash<const void *>(),
                                         std::equal_to<const void *>(),
                                         std::pmr::polymorphic_allocator<std::pair<const void *const, vector_entry_t>>(resource_));
        vectors_.swap(empty_vectors);
        auto empty_vector_keys = std::pmr::vector<const void *>{
            std::pmr::polymorphic_allocator<const void *>(resource_)};
        vector_keys_.swap(empty_vector_keys);
    }

    void clear() noexcept {
        for (auto &entry : matrices_) {
            release_matrix(entry.second);
        }
        matrices_.clear();
        matrix_keys_.clear();

        for (auto &entry : vectors_) {
            release_vector(entry.second);
        }
        vectors_.clear();
        vector_keys_.clear();
    }

    void preallocate(std::size_t matrix_capacity, std::size_t vector_capacity) {
        matrices_.reserve(matrix_capacity);
        matrix_keys_.reserve(matrix_capacity);
        vectors_.reserve(vector_capacity);
        vector_keys_.reserve(vector_capacity);
    }

    void set_compensation_terms(std::size_t terms) {
        for (auto &pair : matrices_) {
            configure_matrix_compensation(pair.second, terms);
        }
        for (auto &pair : vectors_) {
            configure_vector_compensation(pair.second, terms);
        }
    }

    void register_matrix(const deferred_rounding_matrix_t &descriptor, std::size_t terms) {
        if (descriptor.data == nullptr) {
            throw std::invalid_argument("deferred rounding matrix requires non-null data");
        }
        auto [it, inserted] = matrices_.try_emplace(descriptor.data);
        matrix_entry_t &entry = it->second;
        if (inserted) {
            matrix_keys_.push_back(descriptor.data);
        } else {
            release_matrix(entry);
        }
        entry.descriptor = descriptor;
        entry.descriptor.compensation = nullptr;
        entry.descriptor.compensation_elements = 0;
        entry.descriptor.compensation_terms = 0;
        configure_matrix_compensation(entry, terms);
    }

    void register_vector(const deferred_rounding_vector_t &descriptor, std::size_t terms) {
        if (descriptor.data == nullptr) {
            throw std::invalid_argument("deferred rounding vector requires non-null data");
        }
        auto [it, inserted] = vectors_.try_emplace(descriptor.data);
        vector_entry_t &entry = it->second;
        if (inserted) {
            vector_keys_.push_back(descriptor.data);
        } else {
            release_vector(entry);
        }
        entry.descriptor = descriptor;
        entry.descriptor.compensation = nullptr;
        entry.descriptor.compensation_elements = 0;
        entry.descriptor.compensation_terms = 0;
        configure_vector_compensation(entry, terms);
    }

    [[nodiscard]] std::size_t matrix_count() const noexcept { return matrix_keys_.size(); }
    [[nodiscard]] std::size_t vector_count() const noexcept { return vector_keys_.size(); }

    [[nodiscard]] deferred_rounding_matrix_t matrix_at(std::size_t index) const {
        if (index >= matrix_keys_.size()) {
            throw std::out_of_range("matrix index out of range");
        }
        const void *key = matrix_keys_[index];
        auto it = matrices_.find(key);
        if (it == matrices_.end()) {
            throw std::runtime_error("matrix descriptor missing");
        }
        return it->second.descriptor;
    }

    [[nodiscard]] deferred_rounding_vector_t vector_at(std::size_t index) const {
        if (index >= vector_keys_.size()) {
            throw std::out_of_range("vector index out of range");
        }
        const void *key = vector_keys_[index];
        auto it = vectors_.find(key);
        if (it == vectors_.end()) {
            throw std::runtime_error("vector descriptor missing");
        }
        return it->second.descriptor;
    }

    [[nodiscard]] std::optional<deferred_rounding_matrix_t> find_matrix(const void *address) const {
        auto it = matrices_.find(address);
        if (it == matrices_.end()) {
            return std::nullopt;
        }
        return it->second.descriptor;
    }

    [[nodiscard]] std::optional<deferred_rounding_vector_t> find_vector(const void *address) const {
        auto it = vectors_.find(address);
        if (it == vectors_.end()) {
            return std::nullopt;
        }
        return it->second.descriptor;
    }

private:
    void release_matrix(matrix_entry_t &entry) noexcept {
        if (entry.compensation) {
            resource_->deallocate(entry.compensation, entry.compensation_bytes, entry.descriptor.alignment);
            entry.compensation = nullptr;
            entry.compensation_bytes = 0;
        }
    }

    void release_vector(vector_entry_t &entry) noexcept {
        if (entry.compensation) {
            resource_->deallocate(entry.compensation, entry.compensation_bytes, entry.descriptor.alignment);
            entry.compensation = nullptr;
            entry.compensation_bytes = 0;
        }
    }

    void configure_matrix_compensation(matrix_entry_t &entry, std::size_t terms) {
        release_matrix(entry);
        const std::size_t elements = matrix_element_count(entry.descriptor);
        if (terms == 0 || elements == 0 || entry.descriptor.element_size == 0) {
            entry.descriptor.compensation = nullptr;
            entry.descriptor.compensation_elements = elements;
            entry.descriptor.compensation_terms = 0;
            return;
        }
        const std::size_t bytes = elements * entry.descriptor.element_size * terms;
        entry.compensation = resource_->allocate(bytes, entry.descriptor.alignment);
        entry.compensation_bytes = bytes;
        std::memset(entry.compensation, 0, bytes);
        entry.descriptor.compensation = entry.compensation;
        entry.descriptor.compensation_elements = elements;
        entry.descriptor.compensation_terms = terms;
    }

    void configure_vector_compensation(vector_entry_t &entry, std::size_t terms) {
        release_vector(entry);
        const std::size_t elements = vector_element_count(entry.descriptor);
        if (terms == 0 || elements == 0 || entry.descriptor.element_size == 0) {
            entry.descriptor.compensation = nullptr;
            entry.descriptor.compensation_elements = elements;
            entry.descriptor.compensation_terms = 0;
            return;
        }
        const std::size_t bytes = elements * entry.descriptor.element_size * terms;
        entry.compensation = resource_->allocate(bytes, entry.descriptor.alignment);
        entry.compensation_bytes = bytes;
        std::memset(entry.compensation, 0, bytes);
        entry.descriptor.compensation = entry.compensation;
        entry.descriptor.compensation_elements = elements;
        entry.descriptor.compensation_terms = terms;
    }

    std::pmr::memory_resource *resource_;
    std::pmr::unordered_map<const void *, matrix_entry_t> matrices_;
    std::pmr::vector<const void *> matrix_keys_;
    std::pmr::unordered_map<const void *, vector_entry_t> vectors_;
    std::pmr::vector<const void *> vector_keys_;
};

struct runtime_state_t {
    runtime_state_t()
        : active_resource(std::pmr::get_default_resource()),
          registry(active_resource) {}

    ~runtime_state_t() {
        destroy_arena();
    }

    void reset_to_default() {
        destroy_arena();
        active_resource = std::pmr::get_default_resource();
        registry.reset_resource(active_resource);
    }

    void install_arena(const arena_config_t &config) {
        destroy_arena();
        if (!config.buffer || config.size == 0) {
            throw std::invalid_argument("arena requires non-null buffer and non-zero size");
        }
        const std::size_t alignment = config.alignment == 0 ? alignof(std::max_align_t) : config.alignment;
        if (reinterpret_cast<std::uintptr_t>(config.buffer) % alignment != 0) {
            throw std::invalid_argument("arena buffer alignment mismatch");
        }
        arena = config;
        arena_resource = new (&arena_storage) std::pmr::monotonic_buffer_resource(
            config.buffer, config.size, std::pmr::null_memory_resource());
        active_resource = arena_resource;
        registry.reset_resource(active_resource);
        registry.set_compensation_terms(compensation_terms);
    }

    void destroy_arena() noexcept {
        if (!arena_resource) {
            return;
        }
        auto *default_resource = std::pmr::get_default_resource();
        registry.reset_resource(default_resource);
        registry.set_compensation_terms(compensation_terms);
        active_resource = default_resource;
        arena_resource->~monotonic_buffer_resource();
        arena_resource = nullptr;
    }

    void set_compensation_terms_value(std::size_t terms) {
        compensation_terms = terms;
        registry.set_compensation_terms(compensation_terms);
    }

    std::pmr::memory_resource *active_resource = nullptr;
    descriptor_pool_t registry;
    arena_config_t arena{};
    std::pmr::monotonic_buffer_resource *arena_resource = nullptr;
    alignas(std::pmr::monotonic_buffer_resource) unsigned char arena_storage[sizeof(std::pmr::monotonic_buffer_resource)];
    std::size_t compensation_terms = 1;
};

runtime_state_t &state() {
    static runtime_state_t instance{};
    return instance;
}
}  // namespace

void set_default_allocatr() {
    state().reset_to_default();
}

void set_arena(const arena_config_t *config_pointer) {
    if (!config_pointer) {
        set_default_allocatr();
        return;
    }
    state().install_arena(*config_pointer);
}

void set_compensation_terms(std::size_t terms) {
    state().set_compensation_terms_value(terms);
}

std::size_t compensation_terms() {
    return state().compensation_terms;
}

void *allocate(std::size_t size, std::size_t alignment) {
    alignment = alignment == 0 ? alignof(std::max_align_t) : alignment;
    return state().active_resource->allocate(size, alignment);
}

void deallocate(void *pointer, std::size_t size, std::size_t alignment) {
    if (!pointer) {
        return;
    }
    alignment = alignment == 0 ? alignof(std::max_align_t) : alignment;
    state().active_resource->deallocate(pointer, size, alignment);
}

void preallocate(const preallocation_request_t &request) {
    state().registry.preallocate(request.deferred_rounding_matrices, request.deferred_rounding_vectors);
}

void register_deferred_rounding_matrix(const deferred_rounding_matrix_t &descriptor) {
    state().registry.register_matrix(descriptor, state().compensation_terms);
}

void register_deferred_rounding_vector(const deferred_rounding_vector_t &descriptor) {
    state().registry.register_vector(descriptor, state().compensation_terms);
}

std::size_t deferred_rounding_matrix_count() {
    return state().registry.matrix_count();
}

deferred_rounding_matrix_t deferred_rounding_matrix_at(std::size_t index) {
    return state().registry.matrix_at(index);
}

std::optional<deferred_rounding_matrix_t> find_deferred_rounding_matrix(const void *address) {
    return state().registry.find_matrix(address);
}

std::size_t deferred_rounding_vector_count() {
    return state().registry.vector_count();
}

deferred_rounding_vector_t deferred_rounding_vector_at(std::size_t index) {
    return state().registry.vector_at(index);
}

std::optional<deferred_rounding_vector_t> find_deferred_rounding_vector(const void *address) {
    return state().registry.find_vector(address);
}

void clear_deferred_rounding_registrations() {
    state().registry.clear();
}

void set_backend(backend_kind_t kind) {
    switch (kind) {
        case backend_kind_t::empty:
            impl::set_active_backend(nullptr);
            break;
        case backend_kind_t::naive:
            impl::set_active_backend(impl::detail::acquire_naive_backend());
            break;
    }
}

}  // namespace compensated_blas::runtime
