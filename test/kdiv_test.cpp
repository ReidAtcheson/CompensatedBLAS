#include "impl/compensated_arithmetic.hpp"
#include "impl/runtime.cpp"

#include <gtest/gtest.h>

namespace compensated_blas::impl::detail {

// Forward declaration of helper we are about to add to the backend.
template <typename T>
T kdiv(T numerator, T denominator, T &primary, T *bins, std::size_t terms);

namespace {

template <typename T>
T summation_from_bins(T primary, const T *bins, std::size_t terms) {
    long double accumulator = static_cast<long double>(primary);
    for (std::size_t i = 0; i < terms; ++i) {
        accumulator += static_cast<long double>(bins[i]);
    }
    return static_cast<T>(accumulator);
}

template <typename T>
void run_division_case(T numerator, T denominator, std::size_t terms) {
    std::vector<T> storage(terms, T{});
    T primary{};

    const T result = kdiv<T>(numerator, denominator, primary, storage.data(), terms);

    T reconstructed = summation_from_bins(primary, storage.data(), terms);
    EXPECT_EQ(result, reconstructed);

    // Compare against high-precision baseline using long double.
    long double baseline = static_cast<long double>(numerator) / static_cast<long double>(denominator);
    EXPECT_NEAR(static_cast<long double>(result), baseline, std::numeric_limits<T>::epsilon() * 4);
}

template <typename T>
void run_multiple_cases(std::size_t terms) {
    run_division_case<T>(T(1.5), T(0.125), terms);
    run_division_case<T>(T(1e-4), T(3.0), terms);
    run_division_case<T>(T(-12.75), T(7.5), terms);
    run_division_case<T>(T(3.1415926), T(2.7182818), terms);
}

}  // namespace

TEST(KDivTest, FloatSingleBin) { run_multiple_cases<float>(1); }
TEST(KDivTest, FloatTwoBins) { run_multiple_cases<float>(2); }
TEST(KDivTest, DoubleSingleBin) { run_multiple_cases<double>(1); }
TEST(KDivTest, DoubleTwoBins) { run_multiple_cases<double>(2); }

}  // namespace compensated_blas::impl::detail
