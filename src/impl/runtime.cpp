#include "compensated_blas.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <stdexcept>

#include <memory_resource>

namespace compensated_blas::runtime {
namespace {

// Lightweight container that grows descriptor storage via the active PMR resource.
class descriptor_pool {
public:
    explicit descriptor_pool(std::pmr::memory_resource *resource)
        : resource_(resource ? resource : std::pmr::get_default_resource()) {}

    descriptor_pool(const descriptor_pool &) = delete;
    descriptor_pool &operator=(const descriptor_pool &) = delete;

    ~descriptor_pool() { release(); }

    void reset_resource(std::pmr::memory_resource *resource) {
        resource = resource ? resource : std::pmr::get_default_resource();
        if (resource_ == resource) {
            return;
        }

        reallocate(resource);
    }

    void clear() noexcept {
        matrix_size_ = 0;
        vector_size_ = 0;
    }

    void preallocate(std::size_t matrix_capacity, std::size_t vector_capacity) {
        ensure_matrix_capacity(matrix_capacity);
        ensure_vector_capacity(vector_capacity);
    }

    void add_matrix(const deferred_rounding_matrix &descriptor) {
        ensure_matrix_capacity(matrix_size_ + 1);
        matrices_[matrix_size_++] = descriptor;
    }

    void add_vector(const deferred_rounding_vector &descriptor) {
        ensure_vector_capacity(vector_size_ + 1);
        vectors_[vector_size_++] = descriptor;
    }

    [[nodiscard]] std::size_t matrix_count() const noexcept { return matrix_size_; }
    [[nodiscard]] std::size_t vector_count() const noexcept { return vector_size_; }

    [[nodiscard]] deferred_rounding_matrix matrix_at(std::size_t index) const {
        if (index >= matrix_size_) {
            throw std::out_of_range("matrix index out of range");
        }
        return matrices_[index];
    }

    [[nodiscard]] deferred_rounding_vector vector_at(std::size_t index) const {
        if (index >= vector_size_) {
            throw std::out_of_range("vector index out of range");
        }
        return vectors_[index];
    }

private:
    void ensure_matrix_capacity(std::size_t desired) {
        if (desired <= matrix_capacity_) {
            return;
        }

        const std::size_t new_capacity = desired;
        auto *new_block = allocate_block<deferred_rounding_matrix>(new_capacity);
        if (matrices_) {
            std::memcpy(new_block, matrices_, matrix_size_ * sizeof(deferred_rounding_matrix));
            deallocate_block(matrices_, matrix_capacity_);
        }
        matrices_ = new_block;
        matrix_capacity_ = new_capacity;
    }

    void ensure_vector_capacity(std::size_t desired) {
        if (desired <= vector_capacity_) {
            return;
        }

        const std::size_t new_capacity = desired;
        auto *new_block = allocate_block<deferred_rounding_vector>(new_capacity);
        if (vectors_) {
            std::memcpy(new_block, vectors_, vector_size_ * sizeof(deferred_rounding_vector));
            deallocate_block(vectors_, vector_capacity_);
        }
        vectors_ = new_block;
        vector_capacity_ = new_capacity;
    }

    template <typename T>
    T *allocate_block(std::size_t count) {
        if (count == 0) {
            return nullptr;
        }
        const std::size_t bytes = count * sizeof(T);
        return static_cast<T *>(resource_->allocate(bytes, alignof(T)));
    }

    template <typename T>
    void deallocate_block(T *ptr, std::size_t count) noexcept {
        if (!ptr || count == 0) {
            return;
        }
        const std::size_t bytes = count * sizeof(T);
        resource_->deallocate(ptr, bytes, alignof(T));
    }

    void reallocate(std::pmr::memory_resource *new_resource) {
        auto *old_resource = resource_;
        resource_ = new_resource ? new_resource : std::pmr::get_default_resource();

        deferred_rounding_matrix *new_matrices = nullptr;
        if (matrix_capacity_ != 0) {
            new_matrices = static_cast<deferred_rounding_matrix *>(
                resource_->allocate(matrix_capacity_ * sizeof(deferred_rounding_matrix),
                                     alignof(deferred_rounding_matrix)));
            if (matrices_) {
                std::memcpy(new_matrices, matrices_, matrix_size_ * sizeof(deferred_rounding_matrix));
                old_resource->deallocate(matrices_,
                                         matrix_capacity_ * sizeof(deferred_rounding_matrix),
                                         alignof(deferred_rounding_matrix));
            }
        }

        deferred_rounding_vector *new_vectors = nullptr;
        if (vector_capacity_ != 0) {
            new_vectors = static_cast<deferred_rounding_vector *>(
                resource_->allocate(vector_capacity_ * sizeof(deferred_rounding_vector),
                                     alignof(deferred_rounding_vector)));
            if (vectors_) {
                std::memcpy(new_vectors, vectors_, vector_size_ * sizeof(deferred_rounding_vector));
                old_resource->deallocate(vectors_,
                                         vector_capacity_ * sizeof(deferred_rounding_vector),
                                         alignof(deferred_rounding_vector));
            }
        }

        matrices_ = new_matrices;
        vectors_ = new_vectors;
    }

    void release() noexcept {
        if (matrices_) {
            resource_->deallocate(matrices_,
                                   matrix_capacity_ * sizeof(deferred_rounding_matrix),
                                   alignof(deferred_rounding_matrix));
            matrices_ = nullptr;
        }
        if (vectors_) {
            resource_->deallocate(vectors_,
                                   vector_capacity_ * sizeof(deferred_rounding_vector),
                                   alignof(deferred_rounding_vector));
            vectors_ = nullptr;
        }
        matrix_capacity_ = 0;
        vector_capacity_ = 0;
        matrix_size_ = 0;
        vector_size_ = 0;
    }

    std::pmr::memory_resource *resource_ = std::pmr::get_default_resource();
    deferred_rounding_matrix *matrices_ = nullptr;
    deferred_rounding_vector *vectors_ = nullptr;
    std::size_t matrix_size_ = 0;
    std::size_t vector_size_ = 0;
    std::size_t matrix_capacity_ = 0;
    std::size_t vector_capacity_ = 0;
};

// Singleton runtime context that routes allocations and tracks registered descriptors.
struct runtime_state {
    runtime_state()
        : active_resource(std::pmr::get_default_resource()),
          registry(active_resource) {}

    ~runtime_state() {
        destroy_arena();
    }

    void reset_to_default() {
        destroy_arena();
        active_resource = std::pmr::get_default_resource();
        registry.reset_resource(active_resource);
    }

    void install_arena(const arena_config &config) {
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
    }

    void destroy_arena() noexcept {
        if (!arena_resource) {
            return;
        }
        auto *default_resource = std::pmr::get_default_resource();
        if (active_resource == arena_resource) {
            registry.reset_resource(default_resource);
            active_resource = default_resource;
        }
        arena_resource->~monotonic_buffer_resource();
        arena_resource = nullptr;
    }

    std::pmr::memory_resource *active_resource = nullptr;
    descriptor_pool registry;
    arena_config arena{};
    std::pmr::monotonic_buffer_resource *arena_resource = nullptr;
    alignas(std::pmr::monotonic_buffer_resource) unsigned char arena_storage[sizeof(std::pmr::monotonic_buffer_resource)];
};

runtime_state &state() {
    static runtime_state instance{};
    return instance;
}

}  // namespace

void set_default_allocatr() {
    state().reset_to_default();
}

void set_arena(void *arena_config_pointer) {
    if (!arena_config_pointer) {
        set_default_allocatr();
        return;
    }
    const auto config = *static_cast<const arena_config *>(arena_config_pointer);
    state().install_arena(config);
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

void preallocate(const preallocation_request &request) {
    state().registry.preallocate(request.deferred_rounding_matrices, request.deferred_rounding_vectors);
}

void register_deferred_rounding_matrix(const deferred_rounding_matrix &descriptor) {
    state().registry.add_matrix(descriptor);
}

void register_deferred_rounding_vector(const deferred_rounding_vector &descriptor) {
    state().registry.add_vector(descriptor);
}

std::size_t deferred_rounding_matrix_count() {
    return state().registry.matrix_count();
}

deferred_rounding_matrix deferred_rounding_matrix_at(std::size_t index) {
    return state().registry.matrix_at(index);
}

std::size_t deferred_rounding_vector_count() {
    return state().registry.vector_count();
}

deferred_rounding_vector deferred_rounding_vector_at(std::size_t index) {
    return state().registry.vector_at(index);
}

void clear_deferred_rounding_registrations() {
    state().registry.clear();
}

}  // namespace compensated_blas::runtime
