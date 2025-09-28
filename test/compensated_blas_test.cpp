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
