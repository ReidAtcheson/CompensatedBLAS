#pragma once

#include <cmath>
#include <complex>
#include <type_traits>
#include <utility>

namespace compensated_blas {

// In-place compensated summation of two floating-point values.
template <typename T>
inline void two_sum(T &sum, T &value) {
    static_assert(std::is_floating_point_v<T>, "two_sum requires a floating-point type");
    const T temp = sum + value;
    const T bp = temp - sum;
    const T error = (sum - (temp - bp)) + (value - bp);
    sum = temp;
    value = error;
}

template <typename T>
struct is_std_complex : std::false_type {};

template <typename U>
struct is_std_complex<std::complex<U>> : std::true_type {};

template <typename T>
inline std::pair<T, T> two_sum_components(T a, T b) {
    two_sum(a, b);
    return {a, b};
}

template <typename T>
inline std::pair<T, T> add_double_double(const T a_hi,
                                         const T a_lo,
                                         const T b_hi,
                                         const T b_lo) {
    auto [s1, s2] = two_sum_components(a_hi, b_hi);
    auto [t1, t2] = two_sum_components(a_lo, b_lo);
    s2 += t1;
    two_sum(s1, s2);
    s2 += t2;
    two_sum(s1, s2);
    return {s1, s2};
}

template <typename T>
inline std::pair<T, T> sub_double_double(const T a_hi,
                                         const T a_lo,
                                         const T b_hi,
                                         const T b_lo) {
    return add_double_double(a_hi, a_lo, -b_hi, -b_lo);
}

// Dekker-style compensated product using FMA when available.
template <typename T>
inline auto two_prod(const T a, const T b) {
    if constexpr (std::is_floating_point_v<T>) {
        const T product = a * b;
        const T error = std::fma(a, b, -product);
        return std::pair<T, T>{product, error};
    } else if constexpr (is_std_complex<T>::value) {
        using value_type = typename T::value_type;

        const value_type ar = a.real();
        const value_type ai = a.imag();
        const value_type br = b.real();
        const value_type bi = b.imag();

        const auto [rr_hi, rr_lo] = two_prod(ar, br);
        const auto [ii_hi, ii_lo] = two_prod(ai, bi);
        const auto [ri_hi, ri_lo] = two_prod(ar, bi);
        const auto [ir_hi, ir_lo] = two_prod(ai, br);

        const auto [real_hi, real_lo] = sub_double_double(rr_hi, rr_lo, ii_hi, ii_lo);
        const auto [imag_hi, imag_lo] = add_double_double(ri_hi, ri_lo, ir_hi, ir_lo);

        return std::pair<T, T>{T(real_hi, imag_hi), T(real_lo, imag_lo)};
    } else {
        static_assert(std::is_floating_point_v<T>, "two_prod requires a floating-point or std::complex type");
    }
}

}  // namespace compensated_blas
