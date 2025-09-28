#pragma once

#include <cstddef>
#include <cstdint>
#include <memory_resource>

namespace compensated_blas {

#if COMPENSATEDBLAS_USE_ILP64
using index_t = std::int64_t;
#else
using index_t = int;
#endif

index_t identity(index_t value);

namespace runtime {

enum class scalar_type {
    real32,
    real64,
    complex64,
    complex128,
    custom
};

struct arena_config {
    void *buffer = nullptr;
    std::size_t size = 0;
    std::size_t alignment = alignof(std::max_align_t);
};

struct preallocation_request {
    std::size_t deferred_rounding_matrices = 0;
    std::size_t deferred_rounding_vectors = 0;
};

struct deferred_rounding_matrix {
    void *data = nullptr;
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::size_t leading_dimension = 0;
    std::size_t element_size = 0;
    std::size_t alignment = alignof(std::max_align_t);
    scalar_type type = scalar_type::custom;
    bool row_major = true;
};

struct deferred_rounding_vector {
    void *data = nullptr;
    std::size_t length = 0;
    std::size_t stride = 1;
    std::size_t element_size = 0;
    std::size_t alignment = alignof(std::max_align_t);
    scalar_type type = scalar_type::custom;
};

void set_default_allocatr();
inline void set_default_allocator() { set_default_allocatr(); }
void set_arena(void *arena_config_pointer);

void *allocate(std::size_t size, std::size_t alignment);
void deallocate(void *pointer, std::size_t size, std::size_t alignment);

void preallocate(const preallocation_request &request);

void register_deferred_rounding_matrix(const deferred_rounding_matrix &descriptor);
void register_deferred_rounding_vector(const deferred_rounding_vector &descriptor);
inline void register_deferred_roudning_vector(const deferred_rounding_vector &descriptor) {
    register_deferred_rounding_vector(descriptor);
}

[[nodiscard]] std::size_t deferred_rounding_matrix_count();
[[nodiscard]] deferred_rounding_matrix deferred_rounding_matrix_at(std::size_t index);

[[nodiscard]] std::size_t deferred_rounding_vector_count();
[[nodiscard]] deferred_rounding_vector deferred_rounding_vector_at(std::size_t index);

void clear_deferred_rounding_registrations();

}  // namespace runtime

}  // namespace compensated_blas
