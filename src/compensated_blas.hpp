#pragma once

#include <cstddef>
#include <cstdint>
#include <memory_resource>
#include <optional>

namespace compensated_blas {

#if COMPENSATEDBLAS_USE_ILP64
using index_t = std::int64_t;
#else
using index_t = int;
#endif

index_t identity(index_t value);

namespace runtime {

enum class scalar_type_t {
    real32,
    real64,
    complex64,
    complex128,
    custom
};

struct arena_config_t {
    void *buffer = nullptr;
    std::size_t size = 0;
    std::size_t alignment = alignof(std::max_align_t);
};

struct preallocation_request_t {
    std::size_t deferred_rounding_matrices = 0;
    std::size_t deferred_rounding_vectors = 0;
};

struct deferred_rounding_matrix_t {
    void *data = nullptr;
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::size_t leading_dimension = 0;
    std::size_t element_size = 0;
    std::size_t alignment = alignof(std::max_align_t);
    scalar_type_t type = scalar_type_t::custom;
    bool row_major = true;
    void *compensation = nullptr;
    std::size_t compensation_elements = 0;
    std::size_t compensation_terms = 0;
};

struct deferred_rounding_vector_t {
    void *data = nullptr;
    std::size_t length = 0;
    std::size_t stride = 1;
    std::size_t element_size = 0;
    std::size_t alignment = alignof(std::max_align_t);
    scalar_type_t type = scalar_type_t::custom;
    void *compensation = nullptr;
    std::size_t compensation_elements = 0;
    std::size_t compensation_terms = 0;
};

void set_default_allocatr();
inline void set_default_allocator() { set_default_allocatr(); }
void set_arena(const arena_config_t *config);

void set_compensation_terms(std::size_t terms);
[[nodiscard]] std::size_t compensation_terms();

void *allocate(std::size_t size, std::size_t alignment);
void deallocate(void *pointer, std::size_t size, std::size_t alignment);

void preallocate(const preallocation_request_t &request);

void register_deferred_rounding_matrix(const deferred_rounding_matrix_t &descriptor);
void register_deferred_rounding_vector(const deferred_rounding_vector_t &descriptor);
inline void register_deferred_roudning_vector(const deferred_rounding_vector_t &descriptor) {
    register_deferred_rounding_vector(descriptor);
}

[[nodiscard]] std::size_t deferred_rounding_matrix_count();
[[nodiscard]] deferred_rounding_matrix_t deferred_rounding_matrix_at(std::size_t index);
[[nodiscard]] std::optional<deferred_rounding_matrix_t> find_deferred_rounding_matrix(const void *address);

[[nodiscard]] std::size_t deferred_rounding_vector_count();
[[nodiscard]] deferred_rounding_vector_t deferred_rounding_vector_at(std::size_t index);
[[nodiscard]] std::optional<deferred_rounding_vector_t> find_deferred_rounding_vector(const void *address);

enum class backend_kind_t {
    empty,  // no-op backend that ignores calls
    naive   // reference compensated BLAS backend with deferred-rounding support
};

void set_backend(backend_kind_t kind);

void clear_deferred_rounding_registrations();

}  // namespace runtime

}  // namespace compensated_blas
