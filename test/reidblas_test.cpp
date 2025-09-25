#include <array>
#include <cstddef>
#include <memory_resource>

#include <gtest/gtest.h>

#include "reidblas.hpp"
#include "soa_accumulator.hpp"

TEST(reidblas_sanity, identity_returns_input) {
    constexpr reidblas::index_t value = 123;
    EXPECT_EQ(reidblas::identity(value), value);
}

TEST(soa_accumulator_test, accumulates_without_compensation) {
    std::array<double, 3> working{};
    reidblas::soa_accumulator_t<double> acc(working.size(), 0);
    acc.accumulate(0, working[0], 1.0);
    acc.accumulate(0, working[0], 2.0);
    EXPECT_DOUBLE_EQ(acc.round(0, working[0]), 3.0);
}

TEST(soa_accumulator_test, accumulates_with_single_compensation) {
    std::array<double, 1> working{};
    reidblas::soa_accumulator_t<double> acc(working.size(), 1);
    acc.accumulate(0, working[0], 1e16);
    acc.accumulate(0, working[0], 1.0);
    acc.round(0, working[0]);
    EXPECT_DOUBLE_EQ(working[0], 1e16 + 1.0);
}

TEST(soa_accumulator_test, uses_custom_memory_resource) {
    std::array<std::byte, 1024> buffer{};
    std::pmr::monotonic_buffer_resource resource(buffer.data(), buffer.size());
    std::array<double, 4> working{};
    reidblas::soa_accumulator_t<double> acc(working.size(), 2, &resource);
    acc.accumulate(1, working[1], 3.5);
    EXPECT_DOUBLE_EQ(acc.round(1, working[1]), 3.5);
}

TEST(soa_accumulator_test, multiple_compensation_bins) {
    std::array<double, 1> working{};
    reidblas::soa_accumulator_t<double> acc(working.size(), 3);
    for (int i = 0; i < 1000; ++i) {
        acc.accumulate(0, working[0], 1e-10);
    }
    EXPECT_NEAR(acc.round(0, working[0]), 1e-7, 1e-12);
}
