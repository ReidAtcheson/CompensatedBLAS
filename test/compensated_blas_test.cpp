#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <complex>
#include <memory_resource>

#include <gtest/gtest.h>

#include "impl/compensated_arithmetic.hpp"
#include "compensated_blas.hpp"
#include "impl/compensated_accumulator.hpp"

TEST(compensated_blas_sanity, identity_returns_input) {
    constexpr compensated_blas::index_t value = 123;
    EXPECT_EQ(compensated_blas::identity(value), value);
}

namespace {

template <compensated_blas::accumulator_layout Layout>
void check_accumulates_without_compensation() {
    std::array<double, 3> working{};
    compensated_blas::compensated_accumulator_t<double, Layout> acc(working.size(), 0);
    acc.accumulate(0, working[0], 1.0);
    acc.accumulate(0, working[0], 2.0);
    EXPECT_DOUBLE_EQ(acc.round(0, working[0]), 3.0);
}

template <compensated_blas::accumulator_layout Layout>
void check_accumulates_with_single_compensation() {
    std::array<double, 1> working{};
    compensated_blas::compensated_accumulator_t<double, Layout> acc(working.size(), 1);
    acc.accumulate(0, working[0], 1e16);
    acc.accumulate(0, working[0], 1.0);
    acc.round(0, working[0]);
    EXPECT_DOUBLE_EQ(working[0], 1e16 + 1.0);
}

template <compensated_blas::accumulator_layout Layout>
void check_uses_custom_memory_resource() {
    std::array<std::byte, 1024> buffer{};
    std::pmr::monotonic_buffer_resource resource(buffer.data(), buffer.size());
    std::array<double, 4> working{};
    compensated_blas::compensated_accumulator_t<double, Layout> acc(working.size(), 2, &resource);
    acc.accumulate(1, working[1], 3.5);
    EXPECT_DOUBLE_EQ(acc.round(1, working[1]), 3.5);
}

template <compensated_blas::accumulator_layout Layout>
void check_multiple_compensation_bins() {
    std::array<double, 1> working{};
    compensated_blas::compensated_accumulator_t<double, Layout> acc(working.size(), 3);
    for (int i = 0; i < 1000; ++i) {
        acc.accumulate(0, working[0], 1e-10);
    }
    EXPECT_NEAR(acc.round(0, working[0]), 1e-7, 1e-12);
}

}  // namespace

TEST(compensated_accumulator_test, accumulates_without_compensation_soa) {
    check_accumulates_without_compensation<compensated_blas::accumulator_layout::soa>();
}

TEST(compensated_accumulator_test, accumulates_without_compensation_aos) {
    check_accumulates_without_compensation<compensated_blas::accumulator_layout::aos>();
}

TEST(compensated_accumulator_test, accumulates_with_single_compensation_soa) {
    check_accumulates_with_single_compensation<compensated_blas::accumulator_layout::soa>();
}

TEST(compensated_accumulator_test, accumulates_with_single_compensation_aos) {
    check_accumulates_with_single_compensation<compensated_blas::accumulator_layout::aos>();
}

TEST(compensated_accumulator_test, uses_custom_memory_resource_soa) {
    check_uses_custom_memory_resource<compensated_blas::accumulator_layout::soa>();
}

TEST(compensated_accumulator_test, uses_custom_memory_resource_aos) {
    check_uses_custom_memory_resource<compensated_blas::accumulator_layout::aos>();
}

TEST(compensated_accumulator_test, multiple_compensation_bins_soa) {
    check_multiple_compensation_bins<compensated_blas::accumulator_layout::soa>();
}

TEST(compensated_accumulator_test, multiple_compensation_bins_aos) {
    check_multiple_compensation_bins<compensated_blas::accumulator_layout::aos>();
}

TEST(compensated_arithmetic_test, two_prod_recovers_squared_value) {
    const double value = std::nextafter(1.0, 2.0);  // fill mantissa with ones relative to 1.0
    const auto [hi, lo] = compensated_blas::two_prod(value, value);
    const long double expected = static_cast<long double>(value) * static_cast<long double>(value);
    const long double reconstructed = static_cast<long double>(hi) + static_cast<long double>(lo);
    EXPECT_EQ(reconstructed, expected);
}


TEST(compensated_runtime_allocator, default_allocation_and_reset) {
    compensated_blas::runtime::set_default_allocatr();
    void *block = compensated_blas::runtime::allocate(128, alignof(double));
    ASSERT_NE(block, nullptr);
    compensated_blas::runtime::deallocate(block, 128, alignof(double));
}

TEST(compensated_runtime_allocator, arena_allocation_with_alignment) {
    alignas(64) std::array<std::byte, 512> backing{};
    compensated_blas::runtime::arena_config cfg;
    cfg.buffer = backing.data();
    cfg.size = backing.size();
    cfg.alignment = 64;

    compensated_blas::runtime::set_arena(&cfg);
    void *block = compensated_blas::runtime::allocate(128, 64);
    ASSERT_NE(block, nullptr);
    auto *bytes = static_cast<std::byte *>(block);
    EXPECT_GE(bytes, backing.data());
    EXPECT_LE(bytes + 128, backing.data() + backing.size());

    compensated_blas::runtime::deallocate(block, 128, 64);
    compensated_blas::runtime::clear_deferred_rounding_registrations();
    compensated_blas::runtime::set_default_allocatr();
}

TEST(compensated_runtime_deferred_rounding, register_and_retrieve_descriptors) {
    compensated_blas::runtime::set_default_allocatr();
    compensated_blas::runtime::clear_deferred_rounding_registrations();

    compensated_blas::runtime::preallocation_request request{};
    request.deferred_rounding_matrices = 2;
    request.deferred_rounding_vectors = 2;
    compensated_blas::runtime::preallocate(request);

    compensated_blas::runtime::deferred_rounding_matrix matrix{};
    matrix.data = reinterpret_cast<void *>(0x1234);
    matrix.rows = 8;
    matrix.cols = 8;
    matrix.leading_dimension = 8;
    matrix.element_size = sizeof(double);
    matrix.alignment = alignof(double);
    matrix.type = compensated_blas::runtime::scalar_type::real64;
    matrix.row_major = true;

    compensated_blas::runtime::register_deferred_rounding_matrix(matrix);
    ASSERT_EQ(compensated_blas::runtime::deferred_rounding_matrix_count(), 1u);

    auto retrieved_matrix = compensated_blas::runtime::deferred_rounding_matrix_at(0);
    EXPECT_EQ(retrieved_matrix.data, matrix.data);
    EXPECT_EQ(retrieved_matrix.rows, matrix.rows);
    EXPECT_EQ(retrieved_matrix.cols, matrix.cols);
    EXPECT_EQ(retrieved_matrix.leading_dimension, matrix.leading_dimension);
    EXPECT_EQ(retrieved_matrix.element_size, matrix.element_size);
    EXPECT_EQ(retrieved_matrix.alignment, matrix.alignment);
    EXPECT_EQ(retrieved_matrix.type, matrix.type);
    EXPECT_TRUE(retrieved_matrix.row_major);

    compensated_blas::runtime::deferred_rounding_vector vector{};
    vector.data = reinterpret_cast<void *>(0x5678);
    vector.length = 16;
    vector.stride = 1;
    vector.element_size = sizeof(double);
    vector.alignment = alignof(double);
    vector.type = compensated_blas::runtime::scalar_type::real64;

    compensated_blas::runtime::register_deferred_rounding_vector(vector);
    ASSERT_EQ(compensated_blas::runtime::deferred_rounding_vector_count(), 1u);

    auto retrieved_vector = compensated_blas::runtime::deferred_rounding_vector_at(0);
    EXPECT_EQ(retrieved_vector.data, vector.data);
    EXPECT_EQ(retrieved_vector.length, vector.length);
    EXPECT_EQ(retrieved_vector.stride, vector.stride);
    EXPECT_EQ(retrieved_vector.element_size, vector.element_size);
    EXPECT_EQ(retrieved_vector.alignment, vector.alignment);
    EXPECT_EQ(retrieved_vector.type, vector.type);

    compensated_blas::runtime::clear_deferred_rounding_registrations();
}

TEST(compensated_arithmetic_test, two_prod_complex_recovers_squared_value) {
    const double base = std::nextafter(1.0, 2.0);
    const std::complex<double> value(base, -base);
    const auto [hi, lo] = compensated_blas::two_prod(value, value);

    const std::complex<long double> value_ld(static_cast<long double>(value.real()),
                                             static_cast<long double>(value.imag()));
    const std::complex<long double> expected = value_ld * value_ld;
    const std::complex<long double> reconstructed(
        static_cast<long double>(hi.real()) + static_cast<long double>(lo.real()),
        static_cast<long double>(hi.imag()) + static_cast<long double>(lo.imag()));

    EXPECT_EQ(reconstructed.real(), expected.real());
    EXPECT_EQ(reconstructed.imag(), expected.imag());
}
