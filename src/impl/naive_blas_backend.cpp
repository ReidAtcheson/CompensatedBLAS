#include "impl/naive_blas_backend.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>
#include <cstdlib>
#include <limits>

#include "compensated_blas.hpp"
#include "impl/backend_stub.hpp"
#include "impl/compensated_arithmetic.hpp"

namespace compensated_blas::impl::detail {

namespace {

template <typename T>
struct rotmg_constants;

template <>
struct rotmg_constants<float> {
    static constexpr float gam = 4096.0f;
    static constexpr float gamsq = gam * gam;
    static constexpr float rgamsq = 5.96046e-8f;
};

template <>
struct rotmg_constants<double> {
    static constexpr double gam = 4096.0;
    static constexpr double gamsq = gam * gam;
    static constexpr double rgamsq = 5.9604645e-8;
};

}  // namespace

template <typename T>
struct scalar_traits_t {
    static constexpr bool is_complex = false;
    using real_type = T;
};

template <typename T>
struct scalar_traits_t<std::complex<T>> {
    static constexpr bool is_complex = true;
    using real_type = T;
};

inline std::complex<float> to_complex(const compensated_blas_complex_float &value) {
    return {value.real, value.imag};
}

inline compensated_blas_complex_float from_complex(const std::complex<float> &value) {
    return {static_cast<float>(value.real()), static_cast<float>(value.imag())};
}

inline std::complex<double> to_complex(const compensated_blas_complex_double &value) {
    return {value.real, value.imag};
}

inline compensated_blas_complex_double from_complex(const std::complex<double> &value) {
    return {value.real(), value.imag()};
}

static_assert(sizeof(std::complex<float>) == sizeof(compensated_blas_complex_float), "complex float layout mismatch");
static_assert(sizeof(std::complex<double>) == sizeof(compensated_blas_complex_double), "complex double layout mismatch");

inline bool is_zero(float value) { return value == 0.0f; }
inline bool is_zero(double value) { return value == 0.0; }
inline bool is_zero(const std::complex<float> &value) { return value.real() == 0.0f && value.imag() == 0.0f; }
inline bool is_zero(const std::complex<double> &value) { return value.real() == 0.0 && value.imag() == 0.0; }

template <typename T>
void zero_bins(T *bins, std::size_t terms) {
    if (!bins || terms == 0) {
        return;
    }
    std::fill_n(bins, terms, T{});
}

template <typename T>
T reconstruct_from_bins(T primary, const T *bins, std::size_t terms) {
    long double accumulator = static_cast<long double>(primary);
    for (std::size_t i = 0; i < terms; ++i) {
        accumulator += static_cast<long double>(bins[i]);
    }
    return static_cast<T>(accumulator);
}

template <typename T>
void accumulate_value(T &primary, T *bins, std::size_t terms, const T &value);

template <typename T>
const T &matrix_at(const T *matrix,
                   std::size_t leading_dimension,
                   bool row_major,
                   std::size_t row,
                   std::size_t column);

template <typename T>
T kdiv(T numerator, T denominator, T &primary, T *bins, std::size_t terms) {
    primary = numerator / denominator;
    if (bins && terms > 0) {
        zero_bins(bins, terms);
    }

    if (!bins || terms == 0) {
        return primary;
    }

    for (std::size_t iteration = 0; iteration < terms; ++iteration) {
        // Evaluate denominator * current quotient using compensated multiplication.
        auto [prod_hi, prod_lo] = compensated_blas::two_prod(denominator, primary);
        long double product_acc = static_cast<long double>(prod_hi) + static_cast<long double>(prod_lo);

        for (std::size_t i = 0; i < terms; ++i) {
            if (bins[i] == T{}) {
                continue;
            }
            auto [term_hi, term_lo] = compensated_blas::two_prod(denominator, bins[i]);
            product_acc += static_cast<long double>(term_hi);
            product_acc += static_cast<long double>(term_lo);
        }

        long double residual = static_cast<long double>(numerator) - product_acc;
        if (residual == 0.0L) {
            break;
        }

        T delta = static_cast<T>(residual / static_cast<long double>(denominator));
        if (is_zero(delta)) {
            break;
        }

        accumulate_value(primary, bins, terms, delta);
    }

    return reconstruct_from_bins(primary, bins, terms);
}

template float kdiv<float>(float, float, float &, float *, std::size_t);
template double kdiv<double>(double, double, double &, double *, std::size_t);

template <typename T>
std::vector<T> &thread_local_bins(std::size_t count) {
    thread_local std::vector<T> bins;
    if (count == 0) {
        bins.clear();
        return bins;
    }
    if (bins.size() < count) {
        bins.assign(count, T{});
    } else {
        std::fill_n(bins.data(), count, T{});
        bins.resize(count);
    }
    return bins;
}

template <typename T>
void accumulate_value(T &primary, T *bins, std::size_t terms, const T &value) {
    if (is_zero(value)) {
        return;
    }
    if (!bins || terms == 0) {
        primary += value;
        return;
    }

    if constexpr (scalar_traits_t<T>::is_complex) {
        using real_t = typename scalar_traits_t<T>::real_type;
        real_t primary_real = primary.real();
        real_t primary_imag = primary.imag();
        real_t carry_real = value.real();
        real_t carry_imag = value.imag();
        compensated_blas::two_sum(primary_real, carry_real);
        compensated_blas::two_sum(primary_imag, carry_imag);
        primary = T(primary_real, primary_imag);
        for (std::size_t i = 0; i < terms; ++i) {
            if (carry_real == real_t(0) && carry_imag == real_t(0)) {
                return;
            }
            T &bin = bins[i];
            real_t bin_real = bin.real();
            real_t bin_imag = bin.imag();
            compensated_blas::two_sum(bin_real, carry_real);
            compensated_blas::two_sum(bin_imag, carry_imag);
            bin = T(bin_real, bin_imag);
        }
        if (carry_real != real_t(0) || carry_imag != real_t(0)) {
            primary += T(carry_real, carry_imag);
        }
    } else {
        T carry = value;
        compensated_blas::two_sum(primary, carry);
        for (std::size_t i = 0; i < terms; ++i) {
            if (is_zero(carry)) {
                return;
            }
            compensated_blas::two_sum(bins[i], carry);
        }
        if (!is_zero(carry)) {
            primary += carry;
        }
    }
}

template <typename T>
T finalize_value(T &primary, T *bins, std::size_t terms) {
    if (!bins || terms == 0) {
        return primary;
    }
    if constexpr (scalar_traits_t<T>::is_complex) {
        using real_t = typename scalar_traits_t<T>::real_type;
        real_t sum_real = primary.real();
        real_t sum_imag = primary.imag();
        real_t carry_real = 0;
        real_t carry_imag = 0;
        for (std::size_t i = 0; i < terms; ++i) {
            real_t value_real = bins[i].real() + carry_real;
            real_t value_imag = bins[i].imag() + carry_imag;
            bins[i] = T{};
            carry_real = value_real;
            carry_imag = value_imag;
            if (carry_real == real_t(0) && carry_imag == real_t(0)) {
                continue;
            }
            compensated_blas::two_sum(sum_real, carry_real);
            compensated_blas::two_sum(sum_imag, carry_imag);
        }
        if (carry_real != real_t(0) || carry_imag != real_t(0)) {
            sum_real += carry_real;
            sum_imag += carry_imag;
        }
        primary = T(sum_real, sum_imag);
        return primary;
    } else {
        T sum = primary;
        T carry = T{};
        for (std::size_t i = 0; i < terms; ++i) {
            T value = bins[i] + carry;
            bins[i] = T{};
            carry = value;
            if (is_zero(carry)) {
                continue;
            }
            compensated_blas::two_sum(sum, carry);
        }
        if (!is_zero(carry)) {
            sum += carry;
        }
        primary = sum;
        return sum;
    }
}

template <typename T>
void scale_compensated(T &primary, T *bins, std::size_t terms, const T &scale) {
    if (is_zero(scale)) {
        primary = T{};
        zero_bins(bins, terms);
        return;
    }
    primary *= scale;
    if (!bins || terms == 0) {
        return;
    }
    for (std::size_t i = 0; i < terms; ++i) {
        bins[i] *= scale;
    }
}

std::ptrdiff_t to_stride(const std::int64_t *value) {
    return value ? static_cast<std::ptrdiff_t>(*value) : std::ptrdiff_t{1};
}

template <typename T>
T *adjust_pointer(T *ptr, std::int64_t count, std::ptrdiff_t inc) {
    if (ptr == nullptr || count <= 0) {
        return ptr;
    }
    if (inc < 0) {
        const std::int64_t steps = count - 1;
        const std::int64_t stride = static_cast<std::int64_t>(-inc);
        ptr += static_cast<std::ptrdiff_t>(steps * stride);
    }
    return ptr;
}

struct deferred_vector_metadata_t {
    void *compensation = nullptr;
    std::size_t terms = 0;
    std::size_t stride = 1;
    std::size_t element_span = 0;
};

struct deferred_matrix_metadata_t {
    void *compensation = nullptr;
    std::size_t terms = 0;
    std::size_t leading_dimension = 0;
    bool row_major = false;
    std::size_t element_span = 0;
};

template <typename T>
deferred_vector_metadata_t fetch_deferred_vector(T *data, std::ptrdiff_t stride) {
    if (stride <= 0) {
        return {};
    }
    auto descriptor = compensated_blas::runtime::find_deferred_rounding_vector(static_cast<const void *>(data));
    if (!descriptor.has_value()) {
        return {};
    }
    if (descriptor->compensation == nullptr || descriptor->compensation_terms == 0) {
        return {};
    }
    if (descriptor->stride != static_cast<std::size_t>(stride)) {
        return {};
    }
    return {descriptor->compensation,
            descriptor->compensation_terms,
            descriptor->stride,
            descriptor->compensation_elements};
}

template <typename T>
deferred_matrix_metadata_t fetch_deferred_matrix(T *data, std::size_t leading_dimension, bool row_major) {
    auto descriptor = compensated_blas::runtime::find_deferred_rounding_matrix(static_cast<const void *>(data));
    if (!descriptor.has_value()) {
        return {};
    }
    if (descriptor->compensation == nullptr || descriptor->compensation_terms == 0) {
        return {};
    }
    if (descriptor->leading_dimension != leading_dimension || descriptor->row_major != row_major) {
        return {};
    }
    return {descriptor->compensation,
            descriptor->compensation_terms,
            descriptor->leading_dimension,
            descriptor->row_major,
            descriptor->compensation_elements};
}

template <typename T>
T *deferred_bins_at(const deferred_vector_metadata_t &metadata, std::size_t index) {
    if (metadata.compensation == nullptr || metadata.terms == 0) {
        return nullptr;
    }
    const std::size_t offset = index * metadata.stride;
    if (offset >= metadata.element_span) {
        return nullptr;
    }
    auto *base = static_cast<T *>(metadata.compensation);
    return base + offset * metadata.terms;
}

inline float absolute_value(float value) { return std::abs(value); }
inline double absolute_value(double value) { return std::abs(value); }

template <typename T>
T complex_abs1(const std::complex<T> &value) {
    return std::abs(value.real()) + std::abs(value.imag());
}

template <typename T>
T reconstruct_compensated_value(const T &primary, const T *bins, std::size_t terms) {
    if (!bins || terms == 0) {
        return primary;
    }
    if constexpr (scalar_traits_t<T>::is_complex) {
        using real_t = typename scalar_traits_t<T>::real_type;
        long double sum_real = static_cast<long double>(primary.real());
        long double sum_imag = static_cast<long double>(primary.imag());
        for (std::size_t i = 0; i < terms; ++i) {
            sum_real += static_cast<long double>(bins[i].real());
            sum_imag += static_cast<long double>(bins[i].imag());
        }
        return T(static_cast<real_t>(sum_real), static_cast<real_t>(sum_imag));
    } else {
        long double sum = static_cast<long double>(primary);
        for (std::size_t i = 0; i < terms; ++i) {
            sum += static_cast<long double>(bins[i]);
        }
        return static_cast<T>(sum);
    }
}

template <typename T>
void rotg_impl(T *a, T *b, T *c, T *s) {
    if (!a || !b || !c || !s) {
        return;
    }
    const T zero = T(0);
    const T one = T(1);
    T roe = (std::abs(*a) > std::abs(*b)) ? *a : *b;
    T scale = std::abs(*a) + std::abs(*b);
    if (scale == zero) {
        *c = one;
        *s = zero;
        *a = zero;
        *b = zero;
        return;
    }
    T a_scaled = *a / scale;
    T b_scaled = *b / scale;
    T r = scale * std::sqrt(a_scaled * a_scaled + b_scaled * b_scaled);
    if (roe < zero) {
        r = -r;
    }
    *c = *a / r;
    *s = *b / r;
    T z;
    if (std::abs(*a) > std::abs(*b)) {
        z = *s;
    } else if (*c != zero) {
        z = one / *c;
    } else {
        z = one;
    }
    *a = r;
    *b = z;
}

template <typename T>
void complex_rotg_impl(std::complex<T> *a,
                       const std::complex<T> *b,
                       T *c,
                       std::complex<T> *s) {
    if (!a || !b || !c || !s) {
        return;
    }

    const std::complex<T> zero{};
    const T zero_real = T(0);
    const T one = T(1);

    const T abs_a = std::abs(*a);
    const T abs_b = std::abs(*b);

    if (abs_a == zero_real && abs_b == zero_real) {
        *c = one;
        *s = zero;
        *a = zero;
        return;
    }

    if (abs_a == zero_real) {
        *c = zero_real;
        *s = (abs_b == zero_real) ? zero : (*b / abs_b);
        *a = *b;
        return;
    }

    const T scale = abs_a + abs_b;
    if (scale == zero_real) {
        *c = one;
        *s = zero;
        *a = zero;
        return;
    }

    const T abs_a_over_scale = abs_a / scale;
    const T abs_b_over_scale = abs_b / scale;
    const T norm = scale * std::sqrt(abs_a_over_scale * abs_a_over_scale + abs_b_over_scale * abs_b_over_scale);
    if (norm == zero_real) {
        *c = one;
        *s = zero;
        *a = zero;
        return;
    }

    const std::complex<T> alpha = *a / abs_a;
    *c = abs_a / norm;
    *s = alpha * std::conj(*b) / norm;
    *a = alpha * norm;
}

template <typename T>
void rot_impl(std::int64_t count,
              T *x,
              std::ptrdiff_t incx,
              T *y,
              std::ptrdiff_t incy,
              const T c,
              const T s) {
    if (count <= 0 || incx == 0 || incy == 0) {
        return;
    }
    if (c == T(1) && s == T(0)) {
        return;
    }

    auto x_deferred = fetch_deferred_vector(x, incx > 0 ? incx : -incx);
    auto y_deferred = fetch_deferred_vector(y, incy > 0 ? incy : -incy);
    const bool x_has_deferred = x_deferred.compensation != nullptr && x_deferred.terms > 0;
    const bool y_has_deferred = y_deferred.compensation != nullptr && y_deferred.terms > 0;

    const std::size_t global_terms = compensated_blas::runtime::compensation_terms();
    std::vector<T> &local_bins = thread_local_bins<T>(global_terms);
    T *local_bins_ptr = local_bins.empty() ? nullptr : local_bins.data();

    T *x_ptr = adjust_pointer(x, count, incx);
    T *y_ptr = adjust_pointer(y, count, incy);

    for (std::int64_t i = 0; i < count; ++i) {
        T *x_bins_ptr = x_has_deferred ? deferred_bins_at<T>(x_deferred, static_cast<std::size_t>(i)) : nullptr;
        T *y_bins_ptr = y_has_deferred ? deferred_bins_at<T>(y_deferred, static_cast<std::size_t>(i)) : nullptr;

        T x_value = (x_bins_ptr != nullptr) ? reconstruct_compensated_value(*x_ptr, x_bins_ptr, x_deferred.terms) : *x_ptr;
        T y_value = (y_bins_ptr != nullptr) ? reconstruct_compensated_value(*y_ptr, y_bins_ptr, y_deferred.terms) : *y_ptr;

        const T new_x_contrib1 = c * x_value;
        const T new_x_contrib2 = s * y_value;
        const T new_y_contrib1 = c * y_value;
        const T new_y_contrib2 = -s * x_value;

        if (x_bins_ptr) {
            zero_bins(x_bins_ptr, x_deferred.terms);
            T &primary = *x_ptr;
            primary = T{};
            accumulate_value(primary, x_bins_ptr, x_deferred.terms, new_x_contrib1);
            accumulate_value(primary, x_bins_ptr, x_deferred.terms, new_x_contrib2);
        } else {
            if (local_bins_ptr && global_terms > 0) {
                zero_bins(local_bins_ptr, global_terms);
                T primary = T{};
                accumulate_value(primary, local_bins_ptr, global_terms, new_x_contrib1);
                accumulate_value(primary, local_bins_ptr, global_terms, new_x_contrib2);
                *x_ptr = finalize_value(primary, local_bins_ptr, global_terms);
            } else {
                *x_ptr = new_x_contrib1 + new_x_contrib2;
            }
        }

        if (y_bins_ptr) {
            zero_bins(y_bins_ptr, y_deferred.terms);
            T &primary = *y_ptr;
            primary = T{};
            accumulate_value(primary, y_bins_ptr, y_deferred.terms, new_y_contrib1);
            accumulate_value(primary, y_bins_ptr, y_deferred.terms, new_y_contrib2);
        } else {
            if (local_bins_ptr && global_terms > 0) {
                zero_bins(local_bins_ptr, global_terms);
                T primary = T{};
                accumulate_value(primary, local_bins_ptr, global_terms, new_y_contrib1);
                accumulate_value(primary, local_bins_ptr, global_terms, new_y_contrib2);
                *y_ptr = finalize_value(primary, local_bins_ptr, global_terms);
            } else {
                *y_ptr = new_y_contrib1 + new_y_contrib2;
            }
        }

        x_ptr += incx;
        y_ptr += incy;
    }
}

template <typename T>
void rotm_impl(std::int64_t count,
               T *x,
               std::ptrdiff_t incx,
               T *y,
               std::ptrdiff_t incy,
               const T *param) {
    if (count <= 0 || incx == 0 || incy == 0 || param == nullptr) {
        return;
    }

    const T flag = param[0];
    if (flag == static_cast<T>(-2)) {
        return;
    }

    T h11 = T(0);
    T h12 = T(0);
    T h21 = T(0);
    T h22 = T(0);

    if (flag == static_cast<T>(-1)) {
        h11 = param[1];
        h21 = param[2];
        h12 = param[3];
        h22 = param[4];
    } else if (flag == static_cast<T>(0)) {
        h11 = T(1);
        h22 = T(1);
        h21 = param[2];
        h12 = param[3];
    } else if (flag == static_cast<T>(1)) {
        h11 = param[1];
        h12 = T(1);
        h21 = static_cast<T>(-1);
        h22 = param[4];
    } else {
        return;
    }

    auto x_deferred = fetch_deferred_vector(x, incx > 0 ? incx : -incx);
    auto y_deferred = fetch_deferred_vector(y, incy > 0 ? incy : -incy);
    const bool x_has_deferred = x_deferred.compensation != nullptr && x_deferred.terms > 0;
    const bool y_has_deferred = y_deferred.compensation != nullptr && y_deferred.terms > 0;

    const std::size_t global_terms = compensated_blas::runtime::compensation_terms();
    std::vector<T> &local_bins = thread_local_bins<T>(global_terms);
    T *local_bins_ptr = local_bins.empty() ? nullptr : local_bins.data();

    T *x_ptr = adjust_pointer(x, count, incx);
    T *y_ptr = adjust_pointer(y, count, incy);

    for (std::int64_t i = 0; i < count; ++i) {
        T *x_bins_ptr = x_has_deferred ? deferred_bins_at<T>(x_deferred, static_cast<std::size_t>(i)) : nullptr;
        T *y_bins_ptr = y_has_deferred ? deferred_bins_at<T>(y_deferred, static_cast<std::size_t>(i)) : nullptr;

        T x_value = (x_bins_ptr != nullptr) ? reconstruct_compensated_value(*x_ptr, x_bins_ptr, x_deferred.terms) : *x_ptr;
        T y_value = (y_bins_ptr != nullptr) ? reconstruct_compensated_value(*y_ptr, y_bins_ptr, y_deferred.terms) : *y_ptr;

        T new_x;
        T new_y;
        if (flag == static_cast<T>(-1)) {
            new_x = h11 * x_value + h12 * y_value;
            new_y = h21 * x_value + h22 * y_value;
        } else if (flag == static_cast<T>(0)) {
            new_x = x_value + h12 * y_value;
            new_y = h21 * x_value + y_value;
        } else {  // flag == 1
            new_x = h11 * x_value + y_value;
            new_y = -x_value + h22 * y_value;
        }

        if (x_bins_ptr) {
            zero_bins(x_bins_ptr, x_deferred.terms);
            T &primary = *x_ptr;
            primary = T{};
            accumulate_value(primary, x_bins_ptr, x_deferred.terms, new_x);
        } else {
            if (local_bins_ptr && global_terms > 0) {
                zero_bins(local_bins_ptr, global_terms);
                T primary = T{};
                accumulate_value(primary, local_bins_ptr, global_terms, new_x);
                *x_ptr = finalize_value(primary, local_bins_ptr, global_terms);
            } else {
                *x_ptr = new_x;
            }
        }

        if (y_bins_ptr) {
            zero_bins(y_bins_ptr, y_deferred.terms);
            T &primary = *y_ptr;
            primary = T{};
            accumulate_value(primary, y_bins_ptr, y_deferred.terms, new_y);
        } else {
            if (local_bins_ptr && global_terms > 0) {
                zero_bins(local_bins_ptr, global_terms);
                T primary = T{};
                accumulate_value(primary, local_bins_ptr, global_terms, new_y);
                *y_ptr = finalize_value(primary, local_bins_ptr, global_terms);
            } else {
                *y_ptr = new_y;
            }
        }

        x_ptr += incx;
        y_ptr += incy;
    }
}

template <typename T>
void rotmg_impl(T *d1, T *d2, T *x1, const T *y1, T *param) {
    if (!d1 || !d2 || !x1 || !y1 || !param) {
        return;
    }

    const T zero = T(0);
    const T one = T(1);
    const T neg_one = T(-1);
    const T neg_two = T(-2);

    const T gam = rotmg_constants<T>::gam;
    const T gamsq = rotmg_constants<T>::gamsq;
    const T rgamsq = rotmg_constants<T>::rgamsq;

    T flag = neg_two;
    T h11 = zero;
    T h12 = zero;
    T h21 = zero;
    T h22 = zero;

    if (*d1 < zero) {
        flag = neg_one;
        *d1 = zero;
        *d2 = zero;
        *x1 = zero;
    } else {
        const T p2 = (*d2) * (*y1);
        if (p2 == zero) {
            param[0] = neg_two;
            return;
        }

        const T p1 = (*d1) * (*x1);
        const T q2 = p2 * (*y1);
        const T q1 = p1 * (*x1);

        if (std::abs(q1) > std::abs(q2)) {
            h21 = -(*y1) / (*x1);
            h12 = p2 / p1;
            const T u = one - h12 * h21;

            if (u > zero) {
                flag = zero;
                *d1 /= u;
                *d2 /= u;
                *x1 *= u;
            } else {
                flag = neg_one;
                *d1 = zero;
                *d2 = zero;
                *x1 = zero;
                h11 = h12 = h21 = h22 = zero;
            }
        } else {
            if (q2 < zero) {
                flag = neg_one;
                *d1 = zero;
                *d2 = zero;
                *x1 = zero;
                h11 = h12 = h21 = h22 = zero;
            } else {
                flag = one;
                h11 = p1 / p2;
                h22 = (*x1) / (*y1);
                const T u = one + h11 * h22;
                const T temp = (*d2) / u;
                *d2 = (*d1) / u;
                *d1 = temp;
                *x1 = (*y1) * u;
            }
        }

        if (flag != neg_one) {
            if (*d1 != zero) {
                while ((*d1 <= rgamsq) || (*d1 >= gamsq)) {
                    if (flag == zero) {
                        h11 = one;
                        h22 = one;
                        flag = neg_one;
                    } else {
                        h21 = neg_one;
                        h12 = one;
                        flag = neg_one;
                    }

                    if (*d1 <= rgamsq) {
                        *d1 *= gamsq;
                        *x1 /= gam;
                        h11 /= gam;
                        h12 /= gam;
                    } else {
                        *d1 /= gamsq;
                        *x1 *= gam;
                        h11 *= gam;
                        h12 *= gam;
                    }
                }
            }

            if (*d2 != zero) {
                while ((std::abs(*d2) <= rgamsq) || (std::abs(*d2) >= gamsq)) {
                    if (flag == zero) {
                        h11 = one;
                        h22 = one;
                        flag = neg_one;
                    } else {
                        h21 = neg_one;
                        h12 = one;
                        flag = neg_one;
                    }

                    if (std::abs(*d2) <= rgamsq) {
                        *d2 *= gamsq;
                        h21 /= gam;
                        h22 /= gam;
                    } else {
                        *d2 /= gamsq;
                        h21 *= gam;
                        h22 *= gam;
                    }
                }
            }
        }
    }

    param[0] = flag;

    if (flag < zero) {
        param[1] = h11;
        param[2] = h21;
        param[3] = h12;
        param[4] = h22;
    } else if (flag == zero) {
        param[2] = h21;
        param[3] = h12;
    } else if (flag == one) {
        param[1] = h11;
        param[4] = h22;
    }
}

class naive_blas_backend_t final : public stub_backend_t {
public:
    void srotg(float *a, float *b, float *c, float *s) override;
    void drotg(double *a, double *b, double *c, double *s) override;
    void srot(const std::int64_t *n, float *x, const std::int64_t *incx, float *y, const std::int64_t *incy, const float *c, const float *s) override;
    void drot(const std::int64_t *n, double *x, const std::int64_t *incx, double *y, const std::int64_t *incy, const double *c, const double *s) override;
    float sdot(const std::int64_t *n, const float *x, const std::int64_t *incx, const float *y, const std::int64_t *incy) override;
    float sdsdot(const std::int64_t *n, const float *sb, const float *x, const std::int64_t *incx, const float *y, const std::int64_t *incy) override;
    double ddot(const std::int64_t *n, const double *x, const std::int64_t *incx, const double *y, const std::int64_t *incy) override;
    double dsdot(const std::int64_t *n, const float *x, const std::int64_t *incx, const float *y, const std::int64_t *incy) override;
    compensated_blas_complex_float cdotu(const std::int64_t *n, const compensated_blas_complex_float *x, const std::int64_t *incx, const compensated_blas_complex_float *y, const std::int64_t *incy) override;
    compensated_blas_complex_float cdotc(const std::int64_t *n, const compensated_blas_complex_float *x, const std::int64_t *incx, const compensated_blas_complex_float *y, const std::int64_t *incy) override;
    compensated_blas_complex_double zdotu(const std::int64_t *n, const compensated_blas_complex_double *x, const std::int64_t *incx, const compensated_blas_complex_double *y, const std::int64_t *incy) override;
    compensated_blas_complex_double zdotc(const std::int64_t *n, const compensated_blas_complex_double *x, const std::int64_t *incx, const compensated_blas_complex_double *y, const std::int64_t *incy) override;
    void crotg(compensated_blas_complex_float *a, const compensated_blas_complex_float *b, float *c, compensated_blas_complex_float *s) override;
    void csrot(const std::int64_t *n, compensated_blas_complex_float *x, const std::int64_t *incx, compensated_blas_complex_float *y, const std::int64_t *incy, const float *c, const float *s) override;
    void csscal(const std::int64_t *n, const float *alpha, compensated_blas_complex_float *x, const std::int64_t *incx) override;
    void cscal(const std::int64_t *n, const compensated_blas_complex_float *alpha, compensated_blas_complex_float *x, const std::int64_t *incx) override;
    void cswap(const std::int64_t *n, compensated_blas_complex_float *x, const std::int64_t *incx, compensated_blas_complex_float *y, const std::int64_t *incy) override;
    void ccopy(const std::int64_t *n, const compensated_blas_complex_float *x, const std::int64_t *incx, compensated_blas_complex_float *y, const std::int64_t *incy) override;

    void sswap(const std::int64_t *n, float *x, const std::int64_t *incx, float *y, const std::int64_t *incy) override;
    void dswap(const std::int64_t *n, double *x, const std::int64_t *incx, double *y, const std::int64_t *incy) override;
    void scopy(const std::int64_t *n, const float *x, const std::int64_t *incx, float *y, const std::int64_t *incy) override;
    void dcopy(const std::int64_t *n, const double *x, const std::int64_t *incx, double *y, const std::int64_t *incy) override;
    void sscal(const std::int64_t *n, const float *alpha, float *x, const std::int64_t *incx) override;
    void dscal(const std::int64_t *n, const double *alpha, double *x, const std::int64_t *incx) override;
    float snrm2(const std::int64_t *n, const float *x, const std::int64_t *incx) override;
    double dnrm2(const std::int64_t *n, const double *x, const std::int64_t *incx) override;
    float sasum(const std::int64_t *n, const float *x, const std::int64_t *incx) override;
    double dasum(const std::int64_t *n, const double *x, const std::int64_t *incx) override;
    std::int64_t isamax(const std::int64_t *n, const float *x, const std::int64_t *incx) override;
    std::int64_t idamax(const std::int64_t *n, const double *x, const std::int64_t *incx) override;
    float scnrm2(const std::int64_t *n, const compensated_blas_complex_float *x, const std::int64_t *incx) override;
    float scasum(const std::int64_t *n, const compensated_blas_complex_float *x, const std::int64_t *incx) override;
    std::int64_t icamax(const std::int64_t *n, const compensated_blas_complex_float *x, const std::int64_t *incx) override;
    void zrotg(compensated_blas_complex_double *a, const compensated_blas_complex_double *b, double *c, compensated_blas_complex_double *s) override;
    void zdrot(const std::int64_t *n, compensated_blas_complex_double *x, const std::int64_t *incx, compensated_blas_complex_double *y, const std::int64_t *incy, const double *c, const double *s) override;
    void zdscal(const std::int64_t *n, const double *alpha, compensated_blas_complex_double *x, const std::int64_t *incx) override;
    void zscal(const std::int64_t *n, const compensated_blas_complex_double *alpha, compensated_blas_complex_double *x, const std::int64_t *incx) override;
    void zswap(const std::int64_t *n, compensated_blas_complex_double *x, const std::int64_t *incx, compensated_blas_complex_double *y, const std::int64_t *incy) override;
    void zcopy(const std::int64_t *n, const compensated_blas_complex_double *x, const std::int64_t *incx, compensated_blas_complex_double *y, const std::int64_t *incy) override;
    double dznrm2(const std::int64_t *n, const compensated_blas_complex_double *x, const std::int64_t *incx) override;
    double dzasum(const std::int64_t *n, const compensated_blas_complex_double *x, const std::int64_t *incx) override;
    std::int64_t izamax(const std::int64_t *n, const compensated_blas_complex_double *x, const std::int64_t *incx) override;
    void sgemv(const char *trans,
               const std::int64_t *m,
               const std::int64_t *n,
               const float *alpha,
               const float *a,
               const std::int64_t *lda,
               const float *x,
               const std::int64_t *incx,
               const float *beta,
               float *y,
               const std::int64_t *incy) override;
    void dgemv(const char *trans,
               const std::int64_t *m,
               const std::int64_t *n,
               const double *alpha,
               const double *a,
               const std::int64_t *lda,
               const double *x,
               const std::int64_t *incx,
               const double *beta,
               double *y,
               const std::int64_t *incy) override;
    void sgbmv(const char *trans,
               const std::int64_t *m,
               const std::int64_t *n,
               const std::int64_t *kl,
               const std::int64_t *ku,
               const float *alpha,
               const float *a,
               const std::int64_t *lda,
               const float *x,
               const std::int64_t *incx,
               const float *beta,
               float *y,
               const std::int64_t *incy) override;
    void dgbmv(const char *trans,
               const std::int64_t *m,
               const std::int64_t *n,
               const std::int64_t *kl,
               const std::int64_t *ku,
               const double *alpha,
               const double *a,
               const std::int64_t *lda,
               const double *x,
               const std::int64_t *incx,
               const double *beta,
               double *y,
               const std::int64_t *incy) override;
    void ssymv(const char *uplo,
               const std::int64_t *n,
               const float *alpha,
               const float *a,
               const std::int64_t *lda,
               const float *x,
               const std::int64_t *incx,
               const float *beta,
               float *y,
               const std::int64_t *incy) override;
    void dsymv(const char *uplo,
               const std::int64_t *n,
               const double *alpha,
               const double *a,
               const std::int64_t *lda,
               const double *x,
               const std::int64_t *incx,
               const double *beta,
               double *y,
               const std::int64_t *incy) override;
    void ssbmv(const char *uplo,
               const std::int64_t *n,
               const std::int64_t *k,
               const float *alpha,
               const float *a,
               const std::int64_t *lda,
               const float *x,
               const std::int64_t *incx,
               const float *beta,
               float *y,
               const std::int64_t *incy) override;
    void dsbmv(const char *uplo,
               const std::int64_t *n,
               const std::int64_t *k,
               const double *alpha,
               const double *a,
               const std::int64_t *lda,
               const double *x,
               const std::int64_t *incx,
               const double *beta,
               double *y,
               const std::int64_t *incy) override;
    void sspmv(const char *uplo,
               const std::int64_t *n,
               const float *alpha,
               const float *ap,
               const float *x,
               const std::int64_t *incx,
               const float *beta,
               float *y,
               const std::int64_t *incy) override;
    void dspmv(const char *uplo,
               const std::int64_t *n,
               const double *alpha,
               const double *ap,
               const double *x,
               const std::int64_t *incx,
               const double *beta,
               double *y,
               const std::int64_t *incy) override;
    void strsv(const char *uplo,
               const char *trans,
               const char *diag,
               const std::int64_t *n,
               const float *a,
               const std::int64_t *lda,
               float *x,
               const std::int64_t *incx) override;
    void dtrsv(const char *uplo,
               const char *trans,
               const char *diag,
               const std::int64_t *n,
               const double *a,
               const std::int64_t *lda,
               double *x,
               const std::int64_t *incx) override;
    void strmv(const char *uplo,
               const char *trans,
               const char *diag,
               const std::int64_t *n,
               const float *a,
               const std::int64_t *lda,
               float *x,
               const std::int64_t *incx) override;
    void dtrmv(const char *uplo,
               const char *trans,
               const char *diag,
               const std::int64_t *n,
               const double *a,
               const std::int64_t *lda,
               double *x,
               const std::int64_t *incx) override;
    void srotm(const std::int64_t *n, float *x, const std::int64_t *incx, float *y, const std::int64_t *incy, const float *param) override;
    void drotm(const std::int64_t *n, double *x, const std::int64_t *incx, double *y, const std::int64_t *incy, const double *param) override;
    void srotmg(float *d1, float *d2, float *x1, const float *y1, float *param) override;
    void drotmg(double *d1, double *d2, double *x1, const double *y1, double *param) override;

    void saxpy(const std::int64_t *n, const float *alpha, const float *x, const std::int64_t *incx, float *y, const std::int64_t *incy) override;
    void daxpy(const std::int64_t *n, const double *alpha, const double *x, const std::int64_t *incx, double *y, const std::int64_t *incy) override;
    void caxpy(const std::int64_t *n, const compensated_blas_complex_float *alpha, const compensated_blas_complex_float *x, const std::int64_t *incx, compensated_blas_complex_float *y, const std::int64_t *incy) override;
    void zaxpy(const std::int64_t *n, const compensated_blas_complex_double *alpha, const compensated_blas_complex_double *x, const std::int64_t *incx, compensated_blas_complex_double *y, const std::int64_t *incy) override;

    void ssyrk(const char *uplo, const char *trans, const std::int64_t *n, const std::int64_t *k, const float *alpha, const float *a, const std::int64_t *lda, const float *beta, float *c, const std::int64_t *ldc) override;
    void dsyrk(const char *uplo, const char *trans, const std::int64_t *n, const std::int64_t *k, const double *alpha, const double *a, const std::int64_t *lda, const double *beta, double *c, const std::int64_t *ldc) override;
    void csyrk(const char *uplo, const char *trans, const std::int64_t *n, const std::int64_t *k, const compensated_blas_complex_float *alpha, const compensated_blas_complex_float *a, const std::int64_t *lda, const compensated_blas_complex_float *beta, compensated_blas_complex_float *c, const std::int64_t *ldc) override;
    void zsyrk(const char *uplo, const char *trans, const std::int64_t *n, const std::int64_t *k, const compensated_blas_complex_double *alpha, const compensated_blas_complex_double *a, const std::int64_t *lda, const compensated_blas_complex_double *beta, compensated_blas_complex_double *c, const std::int64_t *ldc) override;
};

template <typename T>
void swap_impl(std::int64_t count,
               T *x,
               std::ptrdiff_t incx,
               T *y,
               std::ptrdiff_t incy) {
    if (count <= 0) {
        return;
    }
    auto x_deferred = fetch_deferred_vector(x, incx > 0 ? incx : -incx);
    auto y_deferred = fetch_deferred_vector(y, incy > 0 ? incy : -incy);
    const bool x_has_bins = x_deferred.compensation != nullptr && x_deferred.terms > 0;
    const bool y_has_bins = y_deferred.compensation != nullptr && y_deferred.terms > 0;

    T *x_ptr = adjust_pointer(x, count, incx);
    T *y_ptr = adjust_pointer(y, count, incy);

    for (std::int64_t i = 0; i < count; ++i) {
        T *x_bins = x_has_bins ? deferred_bins_at<T>(x_deferred, static_cast<std::size_t>(i)) : nullptr;
        T *y_bins = y_has_bins ? deferred_bins_at<T>(y_deferred, static_cast<std::size_t>(i)) : nullptr;

        if (x_bins && y_bins && x_deferred.terms == y_deferred.terms) {
            std::swap(*x_ptr, *y_ptr);
            for (std::size_t k = 0; k < x_deferred.terms; ++k) {
                std::swap(x_bins[k], y_bins[k]);
            }
        } else {
            const T x_value = x_bins ? reconstruct_compensated_value(*x_ptr, x_bins, x_deferred.terms) : *x_ptr;
            const T y_value = y_bins ? reconstruct_compensated_value(*y_ptr, y_bins, y_deferred.terms) : *y_ptr;

            if (x_bins) {
                zero_bins(x_bins, x_deferred.terms);
            }
            if (y_bins) {
                zero_bins(y_bins, y_deferred.terms);
            }

            *x_ptr = y_value;
            *y_ptr = x_value;
        }

        x_ptr += incx;
        y_ptr += incy;
    }
}

template <typename T>
void copy_impl(std::int64_t count,
               const T *x,
               std::ptrdiff_t incx,
               T *y,
               std::ptrdiff_t incy) {
    if (count <= 0) {
        return;
    }

    auto src_deferred = fetch_deferred_vector(const_cast<T *>(x), incx > 0 ? incx : -incx);
    auto dst_deferred = fetch_deferred_vector(y, incy > 0 ? incy : -incy);
    const bool src_has_bins = src_deferred.compensation != nullptr && src_deferred.terms > 0;
    const bool dst_has_bins = dst_deferred.compensation != nullptr && dst_deferred.terms > 0;

    const T *x_ptr = adjust_pointer(x, count, incx);
    T *y_ptr = adjust_pointer(y, count, incy);

    for (std::int64_t i = 0; i < count; ++i) {
        const T *src_bins = src_has_bins ? deferred_bins_at<T>(src_deferred, static_cast<std::size_t>(i)) : nullptr;
        const T value = src_bins ? reconstruct_compensated_value(*x_ptr, src_bins, src_deferred.terms) : *x_ptr;

        T *dst_bins = dst_has_bins ? deferred_bins_at<T>(dst_deferred, static_cast<std::size_t>(i)) : nullptr;
        if (dst_bins) {
            zero_bins(dst_bins, dst_deferred.terms);
        }
        *y_ptr = value;

        x_ptr += incx;
        y_ptr += incy;
    }
}

template <typename T>
void scal_impl(std::int64_t count,
               const T &alpha,
               T *x,
               std::ptrdiff_t incx) {
    if (count <= 0) {
        return;
    }
    if (incx == 0) {
        return;
    }

    if (alpha == T(1)) {
        return;
    }

    auto x_deferred = fetch_deferred_vector(x, incx > 0 ? incx : -incx);
    const bool use_deferred = x_deferred.compensation != nullptr && x_deferred.terms > 0;
    const std::size_t global_terms = compensated_blas::runtime::compensation_terms();

    std::vector<T> &local_bins = thread_local_bins<T>(global_terms);
    T *local_bins_ptr = local_bins.empty() ? nullptr : local_bins.data();
    if (local_bins_ptr) {
        zero_bins(local_bins_ptr, global_terms);
    }

    T *x_ptr = adjust_pointer(x, count, incx);
    for (std::int64_t i = 0; i < count; ++i) {
        if (use_deferred) {
            T *bins = deferred_bins_at<T>(x_deferred, static_cast<std::size_t>(i));
            if (bins && x_deferred.terms > 0) {
                scale_compensated(*x_ptr, bins, x_deferred.terms, alpha);
            } else {
                *x_ptr *= alpha;
            }
        } else {
            if (global_terms > 0 && local_bins_ptr) {
                scale_compensated(*x_ptr, local_bins_ptr, global_terms, alpha);
                *x_ptr = finalize_value(*x_ptr, local_bins_ptr, global_terms);
            } else {
                *x_ptr *= alpha;
            }
        }
        x_ptr += incx;
        if (local_bins_ptr) {
            zero_bins(local_bins_ptr, global_terms);
        }
    }
}

namespace {

extern "C" void xerbla_(const char *srname, const std::int64_t *info);

}  // namespace

template <typename T>
void gemv_impl(const char *routine,
               const char *trans,
               std::int64_t m,
               std::int64_t n,
               const T &alpha,
               const T *a,
               std::int64_t lda,
               const T *x,
               std::ptrdiff_t incx,
               const T &beta,
               T *y,
               std::ptrdiff_t incy) {
    const char trans_value = trans ? *trans : '\0';
    const bool nota = (trans_value == 'N' || trans_value == 'n');
    const bool transposed = (trans_value == 'T' || trans_value == 't');
    const bool conjugated = (trans_value == 'C' || trans_value == 'c');
    const bool transpose = transposed || conjugated;
    const bool valid_trans = nota || transpose;

    std::int64_t nrow_a = nota ? m : n;

    std::int64_t info = 0;
    if (!valid_trans) {
        info = 1;
    } else if (m < 0) {
        info = 2;
    } else if (n < 0) {
        info = 3;
    } else if (lda < std::max<std::int64_t>(std::int64_t{1}, nrow_a)) {
        info = 6;
    } else if (incx == 0) {
        info = 8;
    } else if (incy == 0) {
        info = 11;
    }

    if (info != 0) {
        xerbla_(routine, &info);
        return;
    }

    const std::int64_t len_y = transpose ? n : m;
    const std::int64_t len_x = transpose ? m : n;

    if (len_y <= 0) {
        return;
    }

    const std::size_t global_terms = compensated_blas::runtime::compensation_terms();
    const std::size_t lda_stride = static_cast<std::size_t>(lda);
    std::vector<T> &local_bins = thread_local_bins<T>(global_terms);
    T *local_bins_ptr = local_bins.empty() ? nullptr : local_bins.data();

    auto y_metadata = fetch_deferred_vector(y, incy > 0 ? incy : -incy);
    const bool y_has_bins = y_metadata.compensation != nullptr && y_metadata.terms > 0;

    const T one = T(1);
    const bool beta_is_zero = is_zero(beta);
    const bool beta_is_one = is_zero(beta - one);

    T *y_iter = adjust_pointer(y, len_y, incy);

    if (!transpose) {
        const T *x_base = adjust_pointer(x, len_x, incx);
        for (std::int64_t row = 0; row < len_y; ++row) {
            T &destination = *y_iter;

            T *bins = nullptr;
            std::size_t terms = 0;
            bool use_local_bins = true;

            if (y_has_bins) {
                bins = deferred_bins_at<T>(y_metadata, static_cast<std::size_t>(row));
                if (bins != nullptr) {
                    terms = y_metadata.terms;
                    use_local_bins = false;
                }
            }
            if (bins == nullptr) {
                bins = local_bins_ptr;
                terms = global_terms;
                use_local_bins = true;
            }
            if (use_local_bins && terms > 0) {
                zero_bins(bins, terms);
            }

            if (beta_is_zero) {
                destination = T{};
                if (!use_local_bins && bins != nullptr) {
                    zero_bins(bins, terms);
                }
            } else if (!beta_is_one) {
                scale_compensated(destination, bins, terms, beta);
            }

            if (len_x > 0 && !is_zero(alpha)) {
                const T *a_ptr = a + row;
                const T *x_iter = x_base;
                for (std::int64_t col = 0; col < len_x; ++col) {
                    const T x_value = *x_iter;
                    if (!is_zero(x_value)) {
                        const T a_value = *a_ptr;
                        if (!is_zero(a_value)) {
                            const T scaled = alpha * x_value;
                            accumulate_value(destination, bins, terms, scaled * a_value);
                        }
                    }
                    x_iter += incx;
                    a_ptr += static_cast<std::ptrdiff_t>(lda_stride);
                }
            }

            if (use_local_bins) {
                destination = finalize_value(destination, bins, terms);
            }

            y_iter += incy;
        }
        return;
    }

    const T *x_base = adjust_pointer(x, len_x, incx);
    for (std::int64_t col = 0; col < len_y; ++col) {
        T &destination = *y_iter;

        T *bins = nullptr;
        std::size_t terms = 0;
        bool use_local_bins = true;

        if (y_has_bins) {
            bins = deferred_bins_at<T>(y_metadata, static_cast<std::size_t>(col));
            if (bins != nullptr) {
                terms = y_metadata.terms;
                use_local_bins = false;
            }
        }
        if (bins == nullptr) {
            bins = local_bins_ptr;
            terms = global_terms;
            use_local_bins = true;
        }
        if (use_local_bins && terms > 0) {
            zero_bins(bins, terms);
        }

        if (beta_is_zero) {
            destination = T{};
            if (!use_local_bins && bins != nullptr) {
                zero_bins(bins, terms);
            }
        } else if (!beta_is_one) {
            scale_compensated(destination, bins, terms, beta);
        }

        if (len_x > 0 && !is_zero(alpha)) {
            const T *column_ptr = a + static_cast<std::size_t>(col) * lda_stride;
            const T *x_iter = x_base;
            for (std::int64_t row = 0; row < len_x; ++row) {
                T a_value = column_ptr[row];
                if constexpr (scalar_traits_t<T>::is_complex) {
                    if (conjugated) {
                        a_value = std::conj(a_value);
                    }
                }
                if (!is_zero(a_value)) {
                    const T x_value = *x_iter;
                    if (!is_zero(x_value)) {
                        const T contribution = alpha * a_value * x_value;
                        accumulate_value(destination, bins, terms, contribution);
                    }
                }
                x_iter += incx;
            }
        }

        if (use_local_bins) {
            destination = finalize_value(destination, bins, terms);
        }

        y_iter += incy;
    }
}

template <typename T>
void gbmv_impl(const char *routine,
               const char *trans,
               std::int64_t m,
               std::int64_t n,
               std::int64_t kl,
               std::int64_t ku,
               const T &alpha,
               const T *a,
               std::int64_t lda,
               const T *x,
               std::ptrdiff_t incx,
               const T &beta,
               T *y,
               std::ptrdiff_t incy) {
    const char trans_value = trans ? *trans : '\0';
    const bool nota = (trans_value == 'N' || trans_value == 'n');
    const bool transposed = (trans_value == 'T' || trans_value == 't');
    const bool conjugated = (trans_value == 'C' || trans_value == 'c');
    const bool transpose = transposed || conjugated;
    const bool valid_trans = nota || transpose;

    std::int64_t info = 0;
    if (!valid_trans) {
        info = 1;
    } else if (m < 0) {
        info = 2;
    } else if (n < 0) {
        info = 3;
    } else if (kl < 0) {
        info = 4;
    } else if (ku < 0) {
        info = 5;
    } else {
        const std::int64_t band_height = kl + ku + 1;
        if (band_height <= 0) {
            info = 8;
        } else if (lda < band_height) {
            info = 8;
        } else if (incx == 0) {
            info = 10;
        } else if (incy == 0) {
            info = 13;
        }
    }

    if (info != 0) {
        xerbla_(routine, &info);
        return;
    }

    const std::int64_t len_x = transpose ? m : n;
    const std::int64_t len_y = transpose ? n : m;

    if (len_y <= 0) {
        return;
    }

    const T one = T(1);
    const bool zero_alpha = is_zero(alpha);
    const bool beta_is_zero = is_zero(beta);
    const bool beta_is_one = is_zero(beta - one);

    const std::size_t global_terms = compensated_blas::runtime::compensation_terms();
    std::vector<T> &local_bins = thread_local_bins<T>(global_terms);
    T *local_bins_ptr = local_bins.empty() ? nullptr : local_bins.data();

    auto y_metadata = fetch_deferred_vector(y, incy > 0 ? incy : -incy);
    const bool y_has_bins = y_metadata.compensation != nullptr && y_metadata.terms > 0;

    auto x_metadata = fetch_deferred_vector(const_cast<T *>(x), incx > 0 ? incx : -incx);
    const bool x_has_bins = x_metadata.compensation != nullptr && x_metadata.terms > 0;

    const std::size_t lda_stride = static_cast<std::size_t>(lda);

    const T *x_base = adjust_pointer(x, len_x, incx);
    T *y_iter = adjust_pointer(y, len_y, incy);

    if (!transpose) {
        if (m <= 0) {
            return;
        }
        for (std::int64_t i = 0; i < m; ++i) {
            T &destination = *y_iter;

            T *bins = nullptr;
            std::size_t terms = 0;
            bool use_local_bins = true;

            if (y_has_bins) {
                bins = deferred_bins_at<T>(y_metadata, static_cast<std::size_t>(i));
                if (bins != nullptr) {
                    terms = y_metadata.terms;
                    use_local_bins = false;
                }
            }
            if (bins == nullptr) {
                bins = local_bins_ptr;
                terms = global_terms;
                use_local_bins = true;
            }
            if (use_local_bins && bins && terms > 0) {
                zero_bins(bins, terms);
            }

            if (beta_is_zero) {
                destination = T{};
                if (!use_local_bins && bins && terms > 0) {
                    zero_bins(bins, terms);
                }
            } else if (!beta_is_one) {
                scale_compensated(destination, bins, terms, beta);
            }

            if (!zero_alpha && len_x > 0) {
                const std::int64_t j_start = std::max<std::int64_t>(0, i - kl);
                const std::int64_t j_end = std::min<std::int64_t>(n - 1, i + ku);

                if (j_start <= j_end) {
                    const T *x_iter = x_base + j_start * incx;
                    const T *column_ptr = a + static_cast<std::size_t>(j_start) * lda_stride;
                    for (std::int64_t j = j_start; j <= j_end; ++j) {
                        const std::size_t band_row = static_cast<std::size_t>(ku + i - j);
                        const T a_value = column_ptr[band_row];
                        if (!is_zero(a_value)) {
                            T x_value = *x_iter;
                            if (x_has_bins) {
                                T *x_bins = deferred_bins_at<T>(x_metadata, static_cast<std::size_t>(j));
                                if (x_bins) {
                                    x_value = reconstruct_compensated_value(*x_iter, x_bins, x_metadata.terms);
                                }
                            }
                            if (!is_zero(x_value)) {
                                const T contribution = alpha * a_value * x_value;
                                accumulate_value(destination, bins, terms, contribution);
                            }
                        }
                        x_iter += incx;
                        column_ptr += lda_stride;
                    }
                }
            }

            if (use_local_bins) {
                destination = finalize_value(destination, bins, terms);
            }

            y_iter += incy;
        }
        return;
    }

    if (n <= 0) {
        return;
    }

    for (std::int64_t j = 0; j < n; ++j) {
        T &destination = *y_iter;

        T *bins = nullptr;
        std::size_t terms = 0;
        bool use_local_bins = true;

        if (y_has_bins) {
            bins = deferred_bins_at<T>(y_metadata, static_cast<std::size_t>(j));
            if (bins != nullptr) {
                terms = y_metadata.terms;
                use_local_bins = false;
            }
        }
        if (bins == nullptr) {
            bins = local_bins_ptr;
            terms = global_terms;
            use_local_bins = true;
        }
        if (use_local_bins && bins && terms > 0) {
            zero_bins(bins, terms);
        }

        if (beta_is_zero) {
            destination = T{};
            if (!use_local_bins && bins && terms > 0) {
                zero_bins(bins, terms);
            }
        } else if (!beta_is_one) {
            scale_compensated(destination, bins, terms, beta);
        }

        if (!zero_alpha && len_x > 0) {
            const std::int64_t i_start = std::max<std::int64_t>(0, j - ku);
            const std::int64_t i_end = std::min<std::int64_t>(m - 1, j + kl);

            if (i_start <= i_end) {
                const T *x_iter = x_base + i_start * incx;
                const T *column_ptr = a + static_cast<std::size_t>(j) * lda_stride;
                for (std::int64_t i = i_start; i <= i_end; ++i) {
                    std::size_t band_row = static_cast<std::size_t>(ku + i - j);
                    T a_value = column_ptr[band_row];
                    if constexpr (scalar_traits_t<T>::is_complex) {
                        if (conjugated) {
                            a_value = std::conj(a_value);
                        }
                    }
                    if (!is_zero(a_value)) {
                        T x_value = *x_iter;
                        if (x_has_bins) {
                            T *x_bins = deferred_bins_at<T>(x_metadata, static_cast<std::size_t>(i));
                            if (x_bins) {
                                x_value = reconstruct_compensated_value(*x_iter, x_bins, x_metadata.terms);
                            }
                        }
                        if (!is_zero(x_value)) {
                            const T contribution = alpha * a_value * x_value;
                            accumulate_value(destination, bins, terms, contribution);
                        }
                    }
                    x_iter += incx;
                }
            }
        }

        if (use_local_bins) {
            destination = finalize_value(destination, bins, terms);
        }

        y_iter += incy;
    }
}

template <typename T>
void symv_impl(const char *routine,
               const char *uplo,
               std::int64_t n,
               const T &alpha,
               const T *a,
               std::int64_t lda,
               const T *x,
               std::ptrdiff_t incx,
               const T &beta,
               T *y,
               std::ptrdiff_t incy) {
    const char uplo_value = uplo ? *uplo : '\0';
    const bool upper = (uplo_value == 'U' || uplo_value == 'u');
    const bool lower = (uplo_value == 'L' || uplo_value == 'l');

    std::int64_t info = 0;
    if (!(upper || lower)) {
        info = 1;
    } else if (n < 0) {
        info = 2;
    } else if (lda < std::max<std::int64_t>(std::int64_t{1}, n)) {
        info = 5;
    } else if (incx == 0) {
        info = 7;
    } else if (incy == 0) {
        info = 10;
    }

    if (info != 0) {
        xerbla_(routine, &info);
        return;
    }

    if (n == 0) {
        return;
    }

    const bool zero_alpha = is_zero(alpha);
    const T one = T(1);
    const bool beta_is_zero = is_zero(beta);
    const bool beta_is_one = is_zero(beta - one);

    const std::size_t lda_stride = static_cast<std::size_t>(lda);
    const std::size_t global_terms = compensated_blas::runtime::compensation_terms();
    std::vector<T> &local_bins = thread_local_bins<T>(global_terms);
    T *local_bins_ptr = local_bins.empty() ? nullptr : local_bins.data();

    auto y_metadata = fetch_deferred_vector(y, incy > 0 ? incy : -incy);
    const bool y_has_bins = y_metadata.compensation != nullptr && y_metadata.terms > 0;

    auto x_metadata = fetch_deferred_vector(const_cast<T *>(x), incx > 0 ? incx : -incx);
    const bool x_has_bins = x_metadata.compensation != nullptr && x_metadata.terms > 0;

    T *y_base = adjust_pointer(y, n, incy);
    const T *x_base = adjust_pointer(x, n, incx);

    auto y_bins_for = [&](std::int64_t index) -> T * {
        return y_has_bins ? deferred_bins_at<T>(y_metadata, static_cast<std::size_t>(index)) : nullptr;
    };

    auto x_value_at = [&](std::int64_t index) -> T {
        const T *ptr = x_base + index * incx;
        T value = *ptr;
        if (x_has_bins) {
            if (T *bins = deferred_bins_at<T>(x_metadata, static_cast<std::size_t>(index))) {
                value = reconstruct_compensated_value(*ptr, bins, x_metadata.terms);
            }
        }
        return value;
    };

    auto update_y = [&](std::int64_t index, const T &value) {
        if (is_zero(value)) {
            return;
        }
        T &destination = *(y_base + index * incy);
        if (T *bins = y_bins_for(index)) {
            accumulate_value(destination, bins, y_metadata.terms, value);
            return;
        }
        if (global_terms > 0 && local_bins_ptr) {
            zero_bins(local_bins_ptr, global_terms);
            accumulate_value(destination, local_bins_ptr, global_terms, value);
            destination = finalize_value(destination, local_bins_ptr, global_terms);
            return;
        }
        destination += value;
    };

    for (std::int64_t i = 0; i < n; ++i) {
        T &destination = *(y_base + i * incy);
        if (T *bins = y_bins_for(i)) {
            if (beta_is_zero) {
                destination = T{};
                zero_bins(bins, y_metadata.terms);
            } else if (!beta_is_one) {
                scale_compensated(destination, bins, y_metadata.terms, beta);
            }
        } else {
            if (beta_is_zero) {
                destination = T{};
            } else if (!beta_is_one) {
                if (global_terms > 0 && local_bins_ptr) {
                    zero_bins(local_bins_ptr, global_terms);
                    scale_compensated(destination, local_bins_ptr, global_terms, beta);
                    destination = finalize_value(destination, local_bins_ptr, global_terms);
                } else {
                    destination *= beta;
                }
            }
        }
    }

    if (zero_alpha) {
        return;
    }

    if (upper) {
        for (std::int64_t j = 0; j < n; ++j) {
            const T xj = x_value_at(j);
            const T temp1 = alpha * xj;

            long double temp2_acc = 0.0L;
            bool temp2_used = false;

            const T *column = a + static_cast<std::size_t>(j) * lda_stride;
            const T ajj = column[j];
            if (!is_zero(temp1) && !is_zero(ajj)) {
                update_y(j, temp1 * ajj);
            }

            for (std::int64_t i = 0; i < j; ++i) {
                const T aij = column[i];
                if (is_zero(aij)) {
                    continue;
                }
                const T xi = x_value_at(i);
                if (!is_zero(temp1)) {
                    update_y(i, temp1 * aij);
                }
                if (!is_zero(xi)) {
                    temp2_acc += static_cast<long double>(aij) * static_cast<long double>(xi);
                    temp2_used = true;
                }
            }

            if (temp2_used) {
                const T temp2 = static_cast<T>(temp2_acc);
                if (!is_zero(temp2)) {
                    update_y(j, alpha * temp2);
                }
            }
        }
    } else {
        for (std::int64_t j = 0; j < n; ++j) {
            const T xj = x_value_at(j);
            const T temp1 = alpha * xj;

            long double temp2_acc = 0.0L;
            bool temp2_used = false;

            const T *column = a + static_cast<std::size_t>(j) * lda_stride;
            const T ajj = column[j];
            if (!is_zero(temp1) && !is_zero(ajj)) {
                update_y(j, temp1 * ajj);
            }

            for (std::int64_t i = j + 1; i < n; ++i) {
                const T aij = column[i];
                if (is_zero(aij)) {
                    continue;
                }
                const T xi = x_value_at(i);
                if (!is_zero(temp1)) {
                    update_y(i, temp1 * aij);
                }
                if (!is_zero(xi)) {
                    temp2_acc += static_cast<long double>(aij) * static_cast<long double>(xi);
                    temp2_used = true;
                }
            }

            if (temp2_used) {
                const T temp2 = static_cast<T>(temp2_acc);
                if (!is_zero(temp2)) {
                    update_y(j, alpha * temp2);
                }
            }
        }
    }
}

template <typename T>
void sbmv_impl(const char *routine,
               const char *uplo,
               std::int64_t n,
               std::int64_t k,
               const T &alpha,
               const T *a,
               std::int64_t lda,
               const T *x,
               std::ptrdiff_t incx,
               const T &beta,
               T *y,
               std::ptrdiff_t incy) {
    const char uplo_value = uplo ? *uplo : '\0';
    const bool upper = (uplo_value == 'U' || uplo_value == 'u');
    const bool lower = (uplo_value == 'L' || uplo_value == 'l');

    std::int64_t info = 0;
    if (!(upper || lower)) {
        info = 1;
    } else if (n < 0) {
        info = 2;
    } else if (k < 0) {
        info = 3;
    } else if (lda < (k + 1)) {
        info = 5;
    } else if (incx == 0) {
        info = 7;
    } else if (incy == 0) {
        info = 10;
    }

    if (info != 0) {
        xerbla_(routine, &info);
        return;
    }

    if (n == 0) {
        return;
    }

    const bool zero_alpha = is_zero(alpha);
    const T one = T(1);
    const bool beta_is_zero = is_zero(beta);
    const bool beta_is_one = is_zero(beta - one);

    const std::size_t lda_stride = static_cast<std::size_t>(lda);
    const std::size_t global_terms = compensated_blas::runtime::compensation_terms();
    std::vector<T> &local_bins = thread_local_bins<T>(global_terms);
    T *local_bins_ptr = local_bins.empty() ? nullptr : local_bins.data();

    auto y_metadata = fetch_deferred_vector(y, incy > 0 ? incy : -incy);
    const bool y_has_bins = y_metadata.compensation != nullptr && y_metadata.terms > 0;

    auto x_metadata = fetch_deferred_vector(const_cast<T *>(x), incx > 0 ? incx : -incx);
    const bool x_has_bins = x_metadata.compensation != nullptr && x_metadata.terms > 0;

    T *y_base = adjust_pointer(y, n, incy);
    const T *x_base = adjust_pointer(x, n, incx);

    auto y_bins_for = [&](std::int64_t index) -> T * {
        return y_has_bins ? deferred_bins_at<T>(y_metadata, static_cast<std::size_t>(index)) : nullptr;
    };

    auto x_value_at = [&](std::int64_t index) -> T {
        const T *ptr = x_base + index * incx;
        T value = *ptr;
        if (x_has_bins) {
            if (T *bins = deferred_bins_at<T>(x_metadata, static_cast<std::size_t>(index))) {
                value = reconstruct_compensated_value(*ptr, bins, x_metadata.terms);
            }
        }
        return value;
    };

    auto update_y = [&](std::int64_t index, const T &value) {
        if (is_zero(value)) {
            return;
        }
        T &destination = *(y_base + index * incy);
        if (T *bins = y_bins_for(index)) {
            accumulate_value(destination, bins, y_metadata.terms, value);
            return;
        }
        if (global_terms > 0 && local_bins_ptr) {
            accumulate_value(destination, local_bins_ptr, global_terms, value);
            destination = finalize_value(destination, local_bins_ptr, global_terms);
            zero_bins(local_bins_ptr, global_terms);
            return;
        }
        destination += value;
    };

    for (std::int64_t i = 0; i < n; ++i) {
        T &destination = *(y_base + i * incy);
        if (T *bins = y_bins_for(i)) {
            if (beta_is_zero) {
                destination = T{};
                zero_bins(bins, y_metadata.terms);
            } else if (!beta_is_one) {
                scale_compensated(destination, bins, y_metadata.terms, beta);
            }
        } else {
            if (beta_is_zero) {
                destination = T{};
            } else if (!beta_is_one) {
                if (global_terms > 0 && local_bins_ptr) {
                    zero_bins(local_bins_ptr, global_terms);
                    scale_compensated(destination, local_bins_ptr, global_terms, beta);
                    destination = finalize_value(destination, local_bins_ptr, global_terms);
                } else {
                    destination *= beta;
                }
            }
        }
    }

    if (zero_alpha) {
        return;
    }

    if (upper) {
        for (std::int64_t j = 0; j < n; ++j) {
            const T xj = x_value_at(j);
            const T temp1 = alpha * xj;

            long double temp2_acc = 0.0L;
            bool temp2_used = false;

            const T *column = a + static_cast<std::size_t>(j) * lda_stride;
            const std::int64_t i_start = std::max<std::int64_t>(0, j - k);

            const std::size_t diag_index = static_cast<std::size_t>(k);
            const T ajj = column[diag_index];
            if (!is_zero(temp1) && !is_zero(ajj)) {
                update_y(j, temp1 * ajj);
            }

            for (std::int64_t i = i_start; i < j; ++i) {
                const std::size_t row_index = static_cast<std::size_t>(k + i - j);
                const T aij = column[row_index];
                if (is_zero(aij)) {
                    continue;
                }
                const T xi = x_value_at(i);
                if (!is_zero(temp1)) {
                    update_y(i, temp1 * aij);
                }
                if (!is_zero(xi)) {
                    temp2_acc += static_cast<long double>(aij) * static_cast<long double>(xi);
                    temp2_used = true;
                }
            }

            if (temp2_used) {
                const T temp2 = static_cast<T>(temp2_acc);
                if (!is_zero(temp2)) {
                    update_y(j, alpha * temp2);
                }
            }
        }
    } else {
        for (std::int64_t j = 0; j < n; ++j) {
            const T xj = x_value_at(j);
            const T temp1 = alpha * xj;

            long double temp2_acc = 0.0L;
            bool temp2_used = false;

            const T *column = a + static_cast<std::size_t>(j) * lda_stride;
            const std::int64_t i_end = std::min<std::int64_t>(n - 1, j + k);

            const std::size_t diag_index = 0;
            const T ajj = column[diag_index];
            if (!is_zero(temp1) && !is_zero(ajj)) {
                update_y(j, temp1 * ajj);
            }

            for (std::int64_t i = j + 1; i <= i_end; ++i) {
                const std::size_t row_index = static_cast<std::size_t>(i - j);
                const T aij = column[row_index];
                if (is_zero(aij)) {
                    continue;
                }
                const T xi = x_value_at(i);
                if (!is_zero(temp1)) {
                    update_y(i, temp1 * aij);
                }
                if (!is_zero(xi)) {
                    temp2_acc += static_cast<long double>(aij) * static_cast<long double>(xi);
                    temp2_used = true;
                }
            }

            if (temp2_used) {
                const T temp2 = static_cast<T>(temp2_acc);
                if (!is_zero(temp2)) {
                    update_y(j, alpha * temp2);
                }
            }
        }
    }
}

template <typename T>
std::size_t packed_triangle_offset(std::size_t n, std::size_t index, bool upper) {
    if (upper) {
        return index * (index + 1) / 2;
    }
    const std::size_t remaining = n - index;
    return index * (2 * n - index + 1) / 2;
}

template <typename T>
void spmv_impl(const char *routine,
               const char *uplo,
               std::int64_t n,
               const T &alpha,
               const T *ap,
               const T *x,
               std::ptrdiff_t incx,
               const T &beta,
               T *y,
               std::ptrdiff_t incy) {
    const char uplo_value = uplo ? *uplo : '\0';
    const bool upper = (uplo_value == 'U' || uplo_value == 'u');
    const bool lower = (uplo_value == 'L' || uplo_value == 'l');

    std::int64_t info = 0;
    if (!(upper || lower)) {
        info = 1;
    } else if (n < 0) {
        info = 2;
    } else if (incx == 0) {
        info = 5;
    } else if (incy == 0) {
        info = 8;
    }

    if (info != 0) {
        xerbla_(routine, &info);
        return;
    }

    if (n == 0) {
        return;
    }

    const bool zero_alpha = is_zero(alpha);
    const T one = T(1);
    const bool beta_is_zero = is_zero(beta);
    const bool beta_is_one = is_zero(beta - one);

    const std::size_t global_terms = compensated_blas::runtime::compensation_terms();
    std::vector<T> &local_bins = thread_local_bins<T>(global_terms);
    T *local_bins_ptr = local_bins.empty() ? nullptr : local_bins.data();

    auto y_metadata = fetch_deferred_vector(y, incy > 0 ? incy : -incy);
    const bool y_has_bins = y_metadata.compensation != nullptr && y_metadata.terms > 0;

    auto x_metadata = fetch_deferred_vector(const_cast<T *>(x), incx > 0 ? incx : -incx);
    const bool x_has_bins = x_metadata.compensation != nullptr && x_metadata.terms > 0;

    T *y_base = adjust_pointer(y, n, incy);
    const T *x_base = adjust_pointer(x, n, incx);

    auto y_bins_for = [&](std::int64_t index) -> T * {
        return y_has_bins ? deferred_bins_at<T>(y_metadata, static_cast<std::size_t>(index)) : nullptr;
    };

    auto x_value_at = [&](std::int64_t index) -> T {
        const T *ptr = x_base + index * incx;
        T value = *ptr;
        if (x_has_bins) {
            if (T *bins = deferred_bins_at<T>(x_metadata, static_cast<std::size_t>(index))) {
                value = reconstruct_compensated_value(*ptr, bins, x_metadata.terms);
            }
        }
        return value;
    };

    auto update_y = [&](std::int64_t index, const T &value) {
        if (is_zero(value)) {
            return;
        }
        T &destination = *(y_base + index * incy);
        if (T *bins = y_bins_for(index)) {
            accumulate_value(destination, bins, y_metadata.terms, value);
            return;
        }
        if (global_terms > 0 && local_bins_ptr) {
            accumulate_value(destination, local_bins_ptr, global_terms, value);
            destination = finalize_value(destination, local_bins_ptr, global_terms);
            zero_bins(local_bins_ptr, global_terms);
            return;
        }
        destination += value;
    };

    for (std::int64_t i = 0; i < n; ++i) {
        T &destination = *(y_base + i * incy);
        if (T *bins = y_bins_for(i)) {
            if (beta_is_zero) {
                destination = T{};
                zero_bins(bins, y_metadata.terms);
            } else if (!beta_is_one) {
                scale_compensated(destination, bins, y_metadata.terms, beta);
            }
        } else {
            if (beta_is_zero) {
                destination = T{};
            } else if (!beta_is_one) {
                if (global_terms > 0 && local_bins_ptr) {
                    zero_bins(local_bins_ptr, global_terms);
                    scale_compensated(destination, local_bins_ptr, global_terms, beta);
                    destination = finalize_value(destination, local_bins_ptr, global_terms);
                } else {
                    destination *= beta;
                }
            }
        }
    }

    if (zero_alpha) {
        return;
    }

    if (upper) {
        for (std::int64_t j = 0; j < n; ++j) {
            const T xj = x_value_at(j);
            const T temp1 = alpha * xj;

            long double temp2_acc = 0.0L;
            bool temp2_used = false;

            const std::size_t column_offset = packed_triangle_offset<T>(static_cast<std::size_t>(n), static_cast<std::size_t>(j), true);
            const T ajj = ap[column_offset + j];
            if (!is_zero(temp1) && !is_zero(ajj)) {
                update_y(j, temp1 * ajj);
            }

            for (std::int64_t i = 0; i < j; ++i) {
                const T aij = ap[column_offset + i];
                if (is_zero(aij)) {
                    continue;
                }
                const T xi = x_value_at(i);
                if (!is_zero(temp1)) {
                    update_y(i, temp1 * aij);
                }
                if (!is_zero(xi)) {
                    temp2_acc += static_cast<long double>(aij) * static_cast<long double>(xi);
                    temp2_used = true;
                }
            }

            if (temp2_used) {
                const T temp2 = static_cast<T>(temp2_acc);
                if (!is_zero(temp2)) {
                    update_y(j, alpha * temp2);
                }
            }
        }
    } else {
        for (std::int64_t j = 0; j < n; ++j) {
            const T xj = x_value_at(j);
            const T temp1 = alpha * xj;

            long double temp2_acc = 0.0L;
            bool temp2_used = false;

            const std::size_t column_offset = packed_triangle_offset<T>(static_cast<std::size_t>(n), static_cast<std::size_t>(j), false);
            const T ajj = ap[column_offset];
            if (!is_zero(temp1) && !is_zero(ajj)) {
                update_y(j, temp1 * ajj);
            }

            for (std::int64_t i = j + 1; i < n; ++i) {
                const std::size_t row_index = static_cast<std::size_t>(i - j);
                const T aij = ap[column_offset + row_index];
                if (is_zero(aij)) {
                    continue;
                }
                const T xi = x_value_at(i);
                if (!is_zero(temp1)) {
                    update_y(i, temp1 * aij);
                }
                if (!is_zero(xi)) {
                    temp2_acc += static_cast<long double>(aij) * static_cast<long double>(xi);
                    temp2_used = true;
                }
            }

            if (temp2_used) {
                const T temp2 = static_cast<T>(temp2_acc);
                if (!is_zero(temp2)) {
                    update_y(j, alpha * temp2);
                }
            }
        }
    }
}

template <typename T>
void trsv_impl(const char *routine,
               const char *uplo,
               const char *trans,
               const char *diag,
               std::int64_t n,
               const T *a,
               std::size_t lda,
               T *x,
               std::ptrdiff_t incx) {
    const char uplo_value = uplo ? *uplo : '\0';
    const bool upper = (uplo_value == 'U' || uplo_value == 'u');
    const bool lower = (uplo_value == 'L' || uplo_value == 'l');

    const char trans_value = trans ? *trans : '\0';
    const bool nota = (trans_value == 'N' || trans_value == 'n');
    const bool transpose = (trans_value == 'T' || trans_value == 't');

    const char diag_value = diag ? *diag : '\0';
    const bool unit_diagonal = (diag_value == 'U' || diag_value == 'u');
    const bool non_unit = (diag_value == 'N' || diag_value == 'n');

    std::int64_t info = 0;
    if (!(upper || lower)) {
        info = 1;
    } else if (!(nota || transpose)) {
        info = 2;
    } else if (!(unit_diagonal || non_unit)) {
        info = 3;
    } else if (n < 0) {
        info = 4;
    } else if (lda < static_cast<std::size_t>(std::max<std::int64_t>(std::int64_t{1}, n))) {
        info = 6;
    } else if (incx == 0) {
        info = 8;
    }

    if (info != 0) {
        xerbla_(routine, &info);
        return;
    }

    if (n == 0) {
        return;
    }

    const std::size_t global_terms = compensated_blas::runtime::compensation_terms();
    std::vector<T> &local_bins = thread_local_bins<T>(global_terms);
    T *sum_bins = local_bins.empty() ? nullptr : local_bins.data();

    auto x_metadata = fetch_deferred_vector(x, incx > 0 ? incx : -incx);
    const bool x_has_bins = x_metadata.compensation != nullptr && x_metadata.terms > 0;
    const std::size_t x_terms = x_metadata.terms;

    T *x_base = adjust_pointer(x, n, incx);

    auto element_ptr = [&](std::int64_t index) -> T * {
        return x_base + index * incx;
    };

    auto element_bins = [&](std::int64_t index) -> T * {
        return x_has_bins ? deferred_bins_at<T>(x_metadata, static_cast<std::size_t>(index)) : nullptr;
    };

    auto element_value = [&](std::int64_t index) -> T {
        T *ptr = element_ptr(index);
        T *bins = element_bins(index);
        if (bins) {
            return reconstruct_from_bins(*ptr, bins, x_terms);
        }
        return *ptr;
    };

    auto mat_at = [&](std::int64_t row, std::int64_t col) -> T {
        return matrix_at(a, lda, false, static_cast<std::size_t>(row), static_cast<std::size_t>(col));
    };

    if (nota) {
        if (upper) {
            for (std::int64_t i = n - 1; i >= 0; --i) {
                T sum_primary{};
                if (sum_bins) {
                    zero_bins(sum_bins, global_terms);
                }
                for (std::int64_t j = i + 1; j < n; ++j) {
                    const T a_value = mat_at(i, j);
                    if (is_zero(a_value)) {
                        continue;
                    }
                    const T x_value = element_value(j);
                    if (is_zero(x_value)) {
                        continue;
                    }
                    const T contribution = a_value * x_value;
                    accumulate_value(sum_primary, sum_bins, global_terms, contribution);
                }
                T dot = sum_primary;
                if (sum_bins) {
                    dot = finalize_value(sum_primary, sum_bins, global_terms);
                }
                T numerator = element_value(i) - dot;
                T *target_ptr = element_ptr(i);
                T *target_bins = element_bins(i);
                if (!unit_diagonal) {
                    const T diag_entry = mat_at(i, i);
                    const T result = kdiv(numerator, diag_entry, *target_ptr, target_bins, x_terms);
                    if (!target_bins) {
                        *target_ptr = result;
                    }
                } else {
                    if (target_bins && x_terms > 0) {
                        zero_bins(target_bins, x_terms);
                    }
                    *target_ptr = numerator;
                }
            }
        } else {
            for (std::int64_t i = 0; i < n; ++i) {
                T sum_primary{};
                if (sum_bins) {
                    zero_bins(sum_bins, global_terms);
                }
                for (std::int64_t j = 0; j < i; ++j) {
                    const T a_value = mat_at(i, j);
                    if (is_zero(a_value)) {
                        continue;
                    }
                    const T x_value = element_value(j);
                    if (is_zero(x_value)) {
                        continue;
                    }
                    const T contribution = a_value * x_value;
                    accumulate_value(sum_primary, sum_bins, global_terms, contribution);
                }
                T dot = sum_primary;
                if (sum_bins) {
                    dot = finalize_value(sum_primary, sum_bins, global_terms);
                }
                T numerator = element_value(i) - dot;
                T *target_ptr = element_ptr(i);
                T *target_bins = element_bins(i);
                if (!unit_diagonal) {
                    const T diag_entry = mat_at(i, i);
                    const T result = kdiv(numerator, diag_entry, *target_ptr, target_bins, x_terms);
                    if (!target_bins) {
                        *target_ptr = result;
                    }
                } else {
                    if (target_bins && x_terms > 0) {
                        zero_bins(target_bins, x_terms);
                    }
                    *target_ptr = numerator;
                }
            }
        }
    } else {  // transpose
        if (upper) {
            for (std::int64_t i = 0; i < n; ++i) {
                T sum_primary{};
                if (sum_bins) {
                    zero_bins(sum_bins, global_terms);
                }
                for (std::int64_t j = 0; j < i; ++j) {
                    const T a_value = mat_at(j, i);
                    if (is_zero(a_value)) {
                        continue;
                    }
                    const T x_value = element_value(j);
                    if (is_zero(x_value)) {
                        continue;
                    }
                    const T contribution = a_value * x_value;
                    accumulate_value(sum_primary, sum_bins, global_terms, contribution);
                }
                T dot = sum_primary;
                if (sum_bins) {
                    dot = finalize_value(sum_primary, sum_bins, global_terms);
                }
                T numerator = element_value(i) - dot;
                T *target_ptr = element_ptr(i);
                T *target_bins = element_bins(i);
                if (!unit_diagonal) {
                    const T diag_entry = mat_at(i, i);
                    const T result = kdiv(numerator, diag_entry, *target_ptr, target_bins, x_terms);
                    if (!target_bins) {
                        *target_ptr = result;
                    }
                } else {
                    if (target_bins && x_terms > 0) {
                        zero_bins(target_bins, x_terms);
                    }
                    *target_ptr = numerator;
                }
            }
        } else {
            for (std::int64_t i = n - 1; i >= 0; --i) {
                T sum_primary{};
                if (sum_bins) {
                    zero_bins(sum_bins, global_terms);
                }
                for (std::int64_t j = i + 1; j < n; ++j) {
                    const T a_value = mat_at(j, i);
                    if (is_zero(a_value)) {
                        continue;
                    }
                    const T x_value = element_value(j);
                    if (is_zero(x_value)) {
                        continue;
                    }
                    const T contribution = a_value * x_value;
                    accumulate_value(sum_primary, sum_bins, global_terms, contribution);
                }
                T dot = sum_primary;
                if (sum_bins) {
                    dot = finalize_value(sum_primary, sum_bins, global_terms);
                }
                T numerator = element_value(i) - dot;
                T *target_ptr = element_ptr(i);
                T *target_bins = element_bins(i);
                if (!unit_diagonal) {
                    const T diag_entry = mat_at(i, i);
                    const T result = kdiv(numerator, diag_entry, *target_ptr, target_bins, x_terms);
                    if (!target_bins) {
                        *target_ptr = result;
                    }
                } else {
                    if (target_bins && x_terms > 0) {
                        zero_bins(target_bins, x_terms);
                    }
                    *target_ptr = numerator;
                }
            }
        }
    }
}

template <typename T>
void trmv_impl(const char *routine,
               const char *uplo,
               const char *trans,
               const char *diag,
               std::int64_t n,
               const T *a,
               std::size_t lda,
               T *x,
               std::ptrdiff_t incx) {
    const char uplo_value = uplo ? *uplo : '\0';
    const bool upper = (uplo_value == 'U' || uplo_value == 'u');
    const bool lower = (uplo_value == 'L' || uplo_value == 'l');

    const char trans_value = trans ? *trans : '\0';
    const bool nota = (trans_value == 'N' || trans_value == 'n');
    const bool transpose = (trans_value == 'T' || trans_value == 't');

    const char diag_value = diag ? *diag : '\0';
    const bool unit_diagonal = (diag_value == 'U' || diag_value == 'u');
    const bool non_unit = (diag_value == 'N' || diag_value == 'n');

    std::int64_t info = 0;
    if (!(upper || lower)) {
        info = 1;
    } else if (!(nota || transpose)) {
        info = 2;
    } else if (!(unit_diagonal || non_unit)) {
        info = 3;
    } else if (n < 0) {
        info = 4;
    } else if (lda < static_cast<std::size_t>(std::max<std::int64_t>(std::int64_t{1}, n))) {
        info = 6;
    } else if (incx == 0) {
        info = 8;
    }

    if (info != 0) {
        xerbla_(routine, &info);
        return;
    }

    if (n == 0) {
        return;
    }

    const std::size_t global_terms = compensated_blas::runtime::compensation_terms();
    std::vector<T> &local_bins = thread_local_bins<T>(global_terms);

    auto x_metadata = fetch_deferred_vector(x, incx > 0 ? incx : -incx);
    const bool x_has_bins = x_metadata.compensation != nullptr && x_metadata.terms > 0;
    const std::size_t x_terms = x_metadata.terms;

    T *x_base = adjust_pointer(x, n, incx);

    auto element_ptr = [&](std::int64_t index) -> T * {
        return x_base + index * incx;
    };

    auto element_bins = [&](std::int64_t index) -> T * {
        return x_has_bins ? deferred_bins_at<T>(x_metadata, static_cast<std::size_t>(index)) : nullptr;
    };

    auto mat_at = [&](std::int64_t row, std::int64_t col) -> T {
        return matrix_at(a, lda, false, static_cast<std::size_t>(row), static_cast<std::size_t>(col));
    };

    auto process_element = [&](std::int64_t index, auto range_begin, auto range_end, auto step) {
        T sum_primary{};
        T *sum_bins = local_bins.empty() ? nullptr : local_bins.data();
        if (sum_bins) {
            zero_bins(sum_bins, global_terms);
        }

        for (std::int64_t j = range_begin; j != range_end; j += step) {
            const T a_value = mat_at(index, j);
            if (is_zero(a_value)) {
                continue;
            }
            T *x_ptr = element_ptr(j);
            T *x_bins = element_bins(j);
            T x_value = *x_ptr;
            if (x_bins) {
                x_value = reconstruct_from_bins(x_value, x_bins, x_terms);
            }
            if (is_zero(x_value)) {
                continue;
            }
            accumulate_value(sum_primary, sum_bins, global_terms, a_value * x_value);
        }

        if (sum_bins) {
            sum_primary = finalize_value(sum_primary, sum_bins, global_terms);
        }

        T *target_ptr = element_ptr(index);
        T *target_bins = element_bins(index);
        T base_value = *target_ptr;
        if (target_bins) {
            base_value = reconstruct_from_bins(base_value, target_bins, x_terms);
            zero_bins(target_bins, x_terms);
        }

        T result = base_value;
        if (!unit_diagonal) {
            const T diag_entry = mat_at(index, index);
            result *= diag_entry;
        }
        result += sum_primary;

        *target_ptr = result;
        if (target_bins) {
            zero_bins(target_bins, x_terms);
        }
    };

    if (nota) {
        if (upper) {
            for (std::int64_t i = 0; i < n; ++i) {
                process_element(i, i + 1, n, 1);
            }
        } else {
            for (std::int64_t i = n - 1; i >= 0; --i) {
                process_element(i, 0, i, 1);
            }
        }
    } else {
        if (upper) {
            for (std::int64_t i = 0; i < n; ++i) {
                process_element(i, 0, i, 1);
            }
        } else {
            for (std::int64_t i = n - 1; i >= 0; --i) {
                process_element(i, i + 1, n, 1);
            }
        }
    }
}

template <typename T>
T nrm2_impl(std::int64_t count, const T *x, std::ptrdiff_t incx) {
    if (count <= 0) {
        return T{};
    }
    const T *x_ptr = adjust_pointer(x, count, incx);
    std::ptrdiff_t step = incx;

    long double scale = 0.0L;
    long double sumsq = 1.0L;
    bool nonzero = false;
    bool found_infinity = false;
    bool found_nan = false;

    for (std::int64_t i = 0; i < count; ++i) {
        const long double value = static_cast<long double>(*x_ptr);
        if (std::isnan(value)) {
            found_nan = true;
        }
        const long double abs_val = std::abs(value);
        if (std::isinf(abs_val)) {
            found_infinity = true;
        } else if (abs_val != 0.0L) {
            if (!nonzero) {
                scale = abs_val;
                nonzero = true;
            } else if (abs_val > scale) {
                const long double ratio = scale / abs_val;
                sumsq = sumsq * ratio * ratio + 1.0L;
                scale = abs_val;
            } else {
                const long double ratio = abs_val / scale;
                sumsq += ratio * ratio;
            }
        }
        x_ptr += step;
    }

    if (found_nan) {
        return std::numeric_limits<T>::quiet_NaN();
    }
    if (found_infinity) {
        return std::numeric_limits<T>::infinity();
    }
    if (!nonzero) {
        return T{};
    }
    return static_cast<T>(scale * std::sqrt(sumsq));
}

template <typename T>
T complex_nrm2_impl(std::int64_t count, const std::complex<T> *x, std::ptrdiff_t incx) {
    if (count <= 0) {
        return T{};
    }
    const std::complex<T> *x_ptr = adjust_pointer(x, count, incx);
    std::ptrdiff_t step = incx;

    long double scale = 0.0L;
    long double sumsq = 1.0L;
    bool nonzero = false;
    bool found_infinity = false;
    bool found_nan = false;

    for (std::int64_t i = 0; i < count; ++i) {
        const std::complex<T> value = *x_ptr;
        const T magnitude = std::abs(value);
        if (std::isnan(value.real()) || std::isnan(value.imag()) || std::isnan(magnitude)) {
            found_nan = true;
        }
        if (std::isinf(magnitude)) {
            found_infinity = true;
        } else if (magnitude != T(0)) {
            const long double abs_val = static_cast<long double>(magnitude);
            if (!nonzero) {
                scale = abs_val;
                nonzero = true;
            } else if (abs_val > scale) {
                const long double ratio = scale / abs_val;
                sumsq = sumsq * ratio * ratio + 1.0L;
                scale = abs_val;
            } else {
                const long double ratio = abs_val / scale;
                sumsq += ratio * ratio;
            }
        }
        x_ptr += step;
    }

    if (found_nan) {
        return std::numeric_limits<T>::quiet_NaN();
    }
    if (found_infinity) {
        return std::numeric_limits<T>::infinity();
    }
    if (!nonzero) {
        return T{};
    }
    return static_cast<T>(scale * std::sqrt(sumsq));
}

template <typename T>
T asum_impl(std::int64_t count, const T *x, std::ptrdiff_t incx) {
    if (count <= 0) {
        return T{};
    }
    const std::size_t terms = compensated_blas::runtime::compensation_terms();
    std::vector<T> &bins = thread_local_bins<T>(terms);
    T *bins_ptr = bins.empty() ? nullptr : bins.data();
    if (bins_ptr) {
        zero_bins(bins_ptr, terms);
    }

    T sum{};
    const T *x_ptr = adjust_pointer(x, count, incx);
    for (std::int64_t i = 0; i < count; ++i) {
        const T value = absolute_value(*x_ptr);
        accumulate_value(sum, bins_ptr, terms, value);
        x_ptr += incx;
    }
    return finalize_value(sum, bins_ptr, terms);
}

template <typename T>
T complex_asum_impl(std::int64_t count, const std::complex<T> *x, std::ptrdiff_t incx) {
    if (count <= 0) {
        return T{};
    }
    const std::size_t terms = compensated_blas::runtime::compensation_terms();
    std::vector<T> &bins = thread_local_bins<T>(terms);
    T *bins_ptr = bins.empty() ? nullptr : bins.data();
    if (bins_ptr) {
        zero_bins(bins_ptr, terms);
    }

    T sum{};
    const std::complex<T> *x_ptr = adjust_pointer(x, count, incx);
    for (std::int64_t i = 0; i < count; ++i) {
        const T value = complex_abs1(*x_ptr);
        accumulate_value(sum, bins_ptr, terms, value);
        x_ptr += incx;
    }
    return finalize_value(sum, bins_ptr, terms);
}

template <typename T>
std::int64_t iamax_impl(std::int64_t count, const T *x, std::ptrdiff_t incx) {
    if (count <= 0 || incx == 0) {
        return 0;
    }

    const T *x_ptr = adjust_pointer(x, count, incx);
    std::ptrdiff_t step = incx;
    std::int64_t index = 1;
    T max_value = T(0);

    for (std::int64_t i = 0; i < count; ++i) {
        const T value = absolute_value(*x_ptr);
        if (value > max_value) {
            max_value = value;
            index = i + 1;
        }
        x_ptr += step;
    }

    return index;
}

template <typename T>
std::int64_t complex_iamax_impl(std::int64_t count, const std::complex<T> *x, std::ptrdiff_t incx) {
    if (count <= 0 || incx == 0) {
        return 0;
    }

    const std::complex<T> *x_ptr = adjust_pointer(x, count, incx);
    std::ptrdiff_t step = incx;
    std::int64_t index = 1;
    T max_value = T(0);

    for (std::int64_t i = 0; i < count; ++i) {
        const T value = complex_abs1(*x_ptr);
        if (value > max_value) {
            max_value = value;
            index = i + 1;
        }
        x_ptr += step;
    }

    return index;
}

void naive_blas_backend_t::sswap(const std::int64_t *n,
                             float *x,
                             const std::int64_t *incx,
                             float *y,
                             const std::int64_t *incy) {
    swap_impl<float>(*n, x, to_stride(incx), y, to_stride(incy));
}

void naive_blas_backend_t::dswap(const std::int64_t *n,
                             double *x,
                             const std::int64_t *incx,
                             double *y,
                             const std::int64_t *incy) {
    swap_impl<double>(*n, x, to_stride(incx), y, to_stride(incy));
}

void naive_blas_backend_t::cswap(const std::int64_t *n,
                             compensated_blas_complex_float *x,
                             const std::int64_t *incx,
                             compensated_blas_complex_float *y,
                             const std::int64_t *incy) {
    swap_impl<std::complex<float>>(*n,
                                   reinterpret_cast<std::complex<float> *>(x),
                                   to_stride(incx),
                                   reinterpret_cast<std::complex<float> *>(y),
                                   to_stride(incy));
}

void naive_blas_backend_t::zswap(const std::int64_t *n,
                             compensated_blas_complex_double *x,
                             const std::int64_t *incx,
                             compensated_blas_complex_double *y,
                             const std::int64_t *incy) {
    swap_impl<std::complex<double>>(*n,
                                    reinterpret_cast<std::complex<double> *>(x),
                                    to_stride(incx),
                                    reinterpret_cast<std::complex<double> *>(y),
                                    to_stride(incy));
}

void naive_blas_backend_t::srotg(float *a,
                             float *b,
                             float *c,
                             float *s) {
    rotg_impl(a, b, c, s);
}

void naive_blas_backend_t::drotg(double *a,
                             double *b,
                             double *c,
                             double *s) {
    rotg_impl(a, b, c, s);
}

void naive_blas_backend_t::crotg(compensated_blas_complex_float *a,
                             const compensated_blas_complex_float *b,
                             float *c,
                             compensated_blas_complex_float *s) {
    complex_rotg_impl<float>(reinterpret_cast<std::complex<float> *>(a),
                             reinterpret_cast<const std::complex<float> *>(b),
                             c,
                             reinterpret_cast<std::complex<float> *>(s));
}

void naive_blas_backend_t::zrotg(compensated_blas_complex_double *a,
                             const compensated_blas_complex_double *b,
                             double *c,
                             compensated_blas_complex_double *s) {
    complex_rotg_impl<double>(reinterpret_cast<std::complex<double> *>(a),
                              reinterpret_cast<const std::complex<double> *>(b),
                              c,
                              reinterpret_cast<std::complex<double> *>(s));
}

void naive_blas_backend_t::srot(const std::int64_t *n,
                             float *x,
                             const std::int64_t *incx,
                             float *y,
                             const std::int64_t *incy,
                             const float *c,
                             const float *s) {
    if (!c || !s) {
        return;
    }
    rot_impl<float>(*n, x, to_stride(incx), y, to_stride(incy), *c, *s);
}

void naive_blas_backend_t::drot(const std::int64_t *n,
                             double *x,
                             const std::int64_t *incx,
                             double *y,
                             const std::int64_t *incy,
                             const double *c,
                             const double *s) {
    if (!c || !s) {
        return;
    }
    rot_impl<double>(*n, x, to_stride(incx), y, to_stride(incy), *c, *s);
}

void naive_blas_backend_t::csrot(const std::int64_t *n,
                             compensated_blas_complex_float *x,
                             const std::int64_t *incx,
                             compensated_blas_complex_float *y,
                             const std::int64_t *incy,
                             const float *c,
                             const float *s) {
    if (!c || !s) {
        return;
    }
    const std::complex<float> c_value(*c, 0.0f);
    const std::complex<float> s_value(*s, 0.0f);
    rot_impl<std::complex<float>>(*n,
                                  reinterpret_cast<std::complex<float> *>(x),
                                  to_stride(incx),
                                  reinterpret_cast<std::complex<float> *>(y),
                                  to_stride(incy),
                                  c_value,
                                  s_value);
}

void naive_blas_backend_t::zdrot(const std::int64_t *n,
                             compensated_blas_complex_double *x,
                             const std::int64_t *incx,
                             compensated_blas_complex_double *y,
                             const std::int64_t *incy,
                             const double *c,
                             const double *s) {
    if (!c || !s) {
        return;
    }
    const std::complex<double> c_value(*c, 0.0);
    const std::complex<double> s_value(*s, 0.0);
    rot_impl<std::complex<double>>(*n,
                                   reinterpret_cast<std::complex<double> *>(x),
                                   to_stride(incx),
                                   reinterpret_cast<std::complex<double> *>(y),
                                   to_stride(incy),
                                   c_value,
                                   s_value);
}

void naive_blas_backend_t::srotm(const std::int64_t *n,
                             float *x,
                             const std::int64_t *incx,
                             float *y,
                             const std::int64_t *incy,
                             const float *param) {
    rotm_impl<float>(*n, x, to_stride(incx), y, to_stride(incy), param);
}

void naive_blas_backend_t::drotm(const std::int64_t *n,
                             double *x,
                             const std::int64_t *incx,
                             double *y,
                             const std::int64_t *incy,
                             const double *param) {
    rotm_impl<double>(*n, x, to_stride(incx), y, to_stride(incy), param);
}

void naive_blas_backend_t::srotmg(float *d1,
                              float *d2,
                              float *x1,
                              const float *y1,
                              float *param) {
    rotmg_impl(d1, d2, x1, y1, param);
}

void naive_blas_backend_t::drotmg(double *d1,
                              double *d2,
                              double *x1,
                              const double *y1,
                              double *param) {
    rotmg_impl(d1, d2, x1, y1, param);
}

void naive_blas_backend_t::scopy(const std::int64_t *n,
                             const float *x,
                             const std::int64_t *incx,
                             float *y,
                             const std::int64_t *incy) {
    copy_impl<float>(*n, x, to_stride(incx), y, to_stride(incy));
}

void naive_blas_backend_t::dcopy(const std::int64_t *n,
                             const double *x,
                             const std::int64_t *incx,
                             double *y,
                             const std::int64_t *incy) {
    copy_impl<double>(*n, x, to_stride(incx), y, to_stride(incy));
}

void naive_blas_backend_t::ccopy(const std::int64_t *n,
                             const compensated_blas_complex_float *x,
                             const std::int64_t *incx,
                             compensated_blas_complex_float *y,
                             const std::int64_t *incy) {
    copy_impl<std::complex<float>>(*n,
                                    reinterpret_cast<const std::complex<float> *>(x),
                                    to_stride(incx),
                                    reinterpret_cast<std::complex<float> *>(y),
                                    to_stride(incy));
}

void naive_blas_backend_t::zcopy(const std::int64_t *n,
                             const compensated_blas_complex_double *x,
                             const std::int64_t *incx,
                             compensated_blas_complex_double *y,
                             const std::int64_t *incy) {
    copy_impl<std::complex<double>>(*n,
                                     reinterpret_cast<const std::complex<double> *>(x),
                                     to_stride(incx),
                                     reinterpret_cast<std::complex<double> *>(y),
                                     to_stride(incy));
}

void naive_blas_backend_t::sscal(const std::int64_t *n,
                             const float *alpha,
                             float *x,
                             const std::int64_t *incx) {
    scal_impl<float>(*n, *alpha, x, to_stride(incx));
}

void naive_blas_backend_t::dscal(const std::int64_t *n,
                             const double *alpha,
                             double *x,
                             const std::int64_t *incx) {
    scal_impl<double>(*n, *alpha, x, to_stride(incx));
}

void naive_blas_backend_t::csscal(const std::int64_t *n,
                              const float *alpha,
                              compensated_blas_complex_float *x,
                              const std::int64_t *incx) {
    const std::complex<float> scale(*alpha, 0.0f);
    scal_impl<std::complex<float>>(*n,
                                   scale,
                                   reinterpret_cast<std::complex<float> *>(x),
                                   to_stride(incx));
}

void naive_blas_backend_t::cscal(const std::int64_t *n,
                             const compensated_blas_complex_float *alpha,
                             compensated_blas_complex_float *x,
                             const std::int64_t *incx) {
    scal_impl<std::complex<float>>(*n,
                                   to_complex(*alpha),
                                   reinterpret_cast<std::complex<float> *>(x),
                                   to_stride(incx));
}

void naive_blas_backend_t::zdscal(const std::int64_t *n,
                              const double *alpha,
                              compensated_blas_complex_double *x,
                              const std::int64_t *incx) {
    const std::complex<double> scale(*alpha, 0.0);
    scal_impl<std::complex<double>>(*n,
                                    scale,
                                    reinterpret_cast<std::complex<double> *>(x),
                                    to_stride(incx));
}

void naive_blas_backend_t::zscal(const std::int64_t *n,
                             const compensated_blas_complex_double *alpha,
                             compensated_blas_complex_double *x,
                             const std::int64_t *incx) {
    scal_impl<std::complex<double>>(*n,
                                    to_complex(*alpha),
                                    reinterpret_cast<std::complex<double> *>(x),
                                    to_stride(incx));
}

float naive_blas_backend_t::snrm2(const std::int64_t *n,
                              const float *x,
                              const std::int64_t *incx) {
    return nrm2_impl<float>(*n, x, to_stride(incx));
}

double naive_blas_backend_t::dnrm2(const std::int64_t *n,
                               const double *x,
                               const std::int64_t *incx) {
    return nrm2_impl<double>(*n, x, to_stride(incx));
}

float naive_blas_backend_t::scnrm2(const std::int64_t *n,
                               const compensated_blas_complex_float *x,
                               const std::int64_t *incx) {
    return complex_nrm2_impl<float>(*n,
                                    reinterpret_cast<const std::complex<float> *>(x),
                                    to_stride(incx));
}

double naive_blas_backend_t::dznrm2(const std::int64_t *n,
                                const compensated_blas_complex_double *x,
                                const std::int64_t *incx) {
    return complex_nrm2_impl<double>(*n,
                                     reinterpret_cast<const std::complex<double> *>(x),
                                     to_stride(incx));
}

float naive_blas_backend_t::sasum(const std::int64_t *n,
                              const float *x,
                              const std::int64_t *incx) {
    return asum_impl<float>(*n, x, to_stride(incx));
}

double naive_blas_backend_t::dasum(const std::int64_t *n,
                               const double *x,
                               const std::int64_t *incx) {
    return asum_impl<double>(*n, x, to_stride(incx));
}

float naive_blas_backend_t::scasum(const std::int64_t *n,
                               const compensated_blas_complex_float *x,
                               const std::int64_t *incx) {
    return complex_asum_impl<float>(*n,
                                    reinterpret_cast<const std::complex<float> *>(x),
                                    to_stride(incx));
}

double naive_blas_backend_t::dzasum(const std::int64_t *n,
                                const compensated_blas_complex_double *x,
                                const std::int64_t *incx) {
    return complex_asum_impl<double>(*n,
                                     reinterpret_cast<const std::complex<double> *>(x),
                                     to_stride(incx));
}

std::int64_t naive_blas_backend_t::isamax(const std::int64_t *n,
                                      const float *x,
                                      const std::int64_t *incx) {
    return iamax_impl<float>(*n, x, to_stride(incx));
}

std::int64_t naive_blas_backend_t::idamax(const std::int64_t *n,
                                      const double *x,
                                      const std::int64_t *incx) {
    return iamax_impl<double>(*n, x, to_stride(incx));
}

std::int64_t naive_blas_backend_t::icamax(const std::int64_t *n,
                                      const compensated_blas_complex_float *x,
                                      const std::int64_t *incx) {
    return complex_iamax_impl<float>(*n,
                                     reinterpret_cast<const std::complex<float> *>(x),
                                     to_stride(incx));
}

std::int64_t naive_blas_backend_t::izamax(const std::int64_t *n,
                                      const compensated_blas_complex_double *x,
                                      const std::int64_t *incx) {
    return complex_iamax_impl<double>(*n,
                                      reinterpret_cast<const std::complex<double> *>(x),
                                      to_stride(incx));
}

// Helper to compute dot product with optional conjugation of X
template <typename T, bool ConjugateX>
T dot_impl(std::int64_t count, const T *x, std::ptrdiff_t incx, const T *y, std::ptrdiff_t incy) {
    if (count <= 0) {
        return T{};
    }
    const std::size_t terms = compensated_blas::runtime::compensation_terms();
    T primary{};
    std::vector<T> &compensation = thread_local_bins<T>(terms);
    auto *bins = compensation.empty() ? nullptr : compensation.data();

    const T *x_ptr = adjust_pointer(x, count, incx);
    const T *y_ptr = adjust_pointer(y, count, incy);
    std::ptrdiff_t x_step = incx;
    std::ptrdiff_t y_step = incy;
    for (std::int64_t i = 0; i < count; ++i) {
        T left = *x_ptr;
        if constexpr (scalar_traits_t<T>::is_complex && ConjugateX) {
            left = std::conj(left);
        }
        const T right = *y_ptr;
        accumulate_value(primary, bins, terms, left * right);
        x_ptr += x_step;
        y_ptr += y_step;
    }
    return finalize_value(primary, bins, terms);
}

// Specialized dot for mixed precision returning float.
float sdsdot_impl(std::int64_t count,
                  float initial_bias,
                  const float *x,
                  std::ptrdiff_t incx,
                  const float *y,
                  std::ptrdiff_t incy) {
    if (count <= 0) {
        return initial_bias;
    }
    const std::size_t terms = compensated_blas::runtime::compensation_terms();
    double primary = static_cast<double>(initial_bias);
    std::vector<double> &compensation = thread_local_bins<double>(terms);
    auto *bins = compensation.empty() ? nullptr : compensation.data();

    const float *x_ptr = adjust_pointer(x, count, incx);
    const float *y_ptr = adjust_pointer(y, count, incy);
    const std::ptrdiff_t x_step = incx;
    const std::ptrdiff_t y_step = incy;
    for (std::int64_t i = 0; i < count; ++i) {
        const double product = static_cast<double>(*x_ptr) * static_cast<double>(*y_ptr);
        accumulate_value(primary, bins, terms, product);
        x_ptr += x_step;
        y_ptr += y_step;
    }
    const double result = finalize_value(primary, bins, terms);
    return static_cast<float>(result);
}

// Mixed-precision dot returning double from float inputs.
double dsdot_impl(std::int64_t count,
                  const float *x,
                  std::ptrdiff_t incx,
                  const float *y,
                  std::ptrdiff_t incy) {
    if (count <= 0) {
        return 0.0;
    }
    const std::size_t terms = compensated_blas::runtime::compensation_terms();
    double primary{};
    std::vector<double> &compensation = thread_local_bins<double>(terms);
    auto *bins = compensation.empty() ? nullptr : compensation.data();
    
    const float *x_ptr = adjust_pointer(x, count, incx);
    const float *y_ptr = adjust_pointer(y, count, incy);
    const std::ptrdiff_t x_step = incx;
    const std::ptrdiff_t y_step = incy;
    for (std::int64_t i = 0; i < count; ++i) {
        const double product = static_cast<double>(*x_ptr) * static_cast<double>(*y_ptr);
        accumulate_value(primary, bins, terms, product);
        x_ptr += x_step;
        y_ptr += y_step;
    }
    return finalize_value(primary, bins, terms);
}

// Generic axpy implementation
template <typename T>
void axpy_impl(std::int64_t count,
               const T &alpha,
               const T *x,
               std::ptrdiff_t incx,
               T *y,
               std::ptrdiff_t incy) {
    if (count <= 0 || is_zero(alpha)) {
        return;
    }
    const std::size_t global_terms = compensated_blas::runtime::compensation_terms();

    const deferred_vector_metadata_t deferred = fetch_deferred_vector(y, incy);
    const bool use_deferred = deferred.compensation != nullptr && deferred.terms > 0;

    std::vector<T> &local_bins = thread_local_bins<T>(global_terms);
    T *local_bins_ptr = local_bins.empty() ? nullptr : local_bins.data();

    const T *x_ptr = adjust_pointer(x, count, incx);
    T *y_ptr = adjust_pointer(y, count, incy);
    const std::ptrdiff_t x_step = incx;
    const std::ptrdiff_t y_step = incy;

    for (std::int64_t i = 0; i < count; ++i) {
        const T contribution = alpha * (*x_ptr);
        T &destination = *y_ptr;

        if (use_deferred) {
            auto *compensation = static_cast<T *>(deferred.compensation);
            const std::size_t offset = static_cast<std::size_t>(i * deferred.stride);
            if (offset >= deferred.element_span) {
                T *no_bins = nullptr;
                accumulate_value(destination, no_bins, 0, contribution);
                continue;
            }
            T *bins = compensation + offset * deferred.terms;
            accumulate_value(destination, bins, deferred.terms, contribution);
        } else {
            accumulate_value(destination, local_bins_ptr, global_terms, contribution);
            destination = finalize_value(destination, local_bins_ptr, global_terms);
        }
        x_ptr += x_step;
        y_ptr += y_step;
    }
}

// Access helper for matrices
template <typename T>
const T &matrix_at(const T *matrix,
                   std::size_t leading_dimension,
                   bool row_major,
                   std::size_t row,
                   std::size_t column) {
    if (row_major) {
        return matrix[row * leading_dimension + column];
    }
    return matrix[column * leading_dimension + row];
}

template <typename T>
T &matrix_at(T *matrix,
             std::size_t leading_dimension,
             bool row_major,
             std::size_t row,
             std::size_t column) {
    if (row_major) {
        return matrix[row * leading_dimension + column];
    }
    return matrix[column * leading_dimension + row];
}

// SYRK helper
template <typename T>
void syrk_impl(const char *uplo,
               const char *trans,
               std::int64_t n,
               std::int64_t k,
               const T &alpha,
               const T *a,
               std::size_t lda,
               const T &beta,
               T *c,
               std::size_t ldc,
               bool row_major_a,
               deferred_matrix_metadata_t deferred) {
    if (n <= 0) {
        return;
    }
    const bool upper = (uplo && (*uplo == 'U' || *uplo == 'u'));
    const bool transposed = (trans && (*trans == 'T' || *trans == 't' || *trans == 'C' || *trans == 'c'));
    const bool conj_first = scalar_traits_t<T>::is_complex && trans && (*trans == 'C' || *trans == 'c');

    const std::size_t global_terms = compensated_blas::runtime::compensation_terms();
    std::vector<T> &local_bins = thread_local_bins<T>(global_terms);
    T *local_bins_ptr = local_bins.empty() ? nullptr : local_bins.data();

    const bool use_deferred = deferred.compensation != nullptr && deferred.terms > 0;
    const bool c_row_major = use_deferred ? deferred.row_major : false;

    auto value_from_a = [&](std::size_t row, std::size_t col) -> T {
        return matrix_at(a, lda, row_major_a, row, col);
    };

    for (std::int64_t j = 0; j < n; ++j) {
        const std::int64_t i_begin = upper ? 0 : j;
        const std::int64_t i_end = upper ? j : n - 1;
        for (std::int64_t i = i_begin; i <= i_end; ++i) {
            T &destination = matrix_at(c, ldc, c_row_major, static_cast<std::size_t>(i), static_cast<std::size_t>(j));
            T *bins = nullptr;
            if (use_deferred) {
                auto *comp = static_cast<T *>(deferred.compensation);
                const std::size_t index = c_row_major ?
                                              static_cast<std::size_t>(i) * ldc + static_cast<std::size_t>(j) :
                                              static_cast<std::size_t>(j) * ldc + static_cast<std::size_t>(i);
                if (index < deferred.element_span) {
                    bins = comp + index * deferred.terms;
                }
            }

            if (is_zero(beta)) {
                destination = T{};
                if (bins) {
                    zero_bins(bins, deferred.terms);
                }
                zero_bins(local_bins_ptr, global_terms);
            } else {
                scale_compensated(destination, bins ? bins : local_bins_ptr, bins ? deferred.terms : global_terms, beta);
            }

            if (k <= 0 || is_zero(alpha)) {
                if (!use_deferred) {
                    destination = finalize_value(destination, local_bins_ptr, global_terms);
                }
                continue;
            }

            for (std::int64_t l = 0; l < k; ++l) {
                T left;
                T right;
                if (!transposed) {
                    left = value_from_a(static_cast<std::size_t>(i), static_cast<std::size_t>(l));
                    right = value_from_a(static_cast<std::size_t>(j), static_cast<std::size_t>(l));
                } else {
                    left = value_from_a(static_cast<std::size_t>(l), static_cast<std::size_t>(i));
                    right = value_from_a(static_cast<std::size_t>(l), static_cast<std::size_t>(j));
                }
                if constexpr (scalar_traits_t<T>::is_complex) {
                    if (conj_first) {
                        left = std::conj(left);
                    }
                }
                const T contribution = alpha * left * right;
                accumulate_value(destination, bins ? bins : local_bins_ptr, bins ? deferred.terms : global_terms, contribution);
            }

            if (!use_deferred) {
                destination = finalize_value(destination, local_bins_ptr, global_terms);
            }
        }
    }
}

float naive_blas_backend_t::sdot(const std::int64_t *n,
                             const float *x,
                             const std::int64_t *incx,
                             const float *y,
                             const std::int64_t *incy) {
    return dot_impl<float, false>(*n, x, to_stride(incx), y, to_stride(incy));
}

float naive_blas_backend_t::sdsdot(const std::int64_t *n,
                               const float *sb,
                               const float *x,
                               const std::int64_t *incx,
                               const float *y,
                               const std::int64_t *incy) {
    return sdsdot_impl(*n, *sb, x, to_stride(incx), y, to_stride(incy));
}

double naive_blas_backend_t::ddot(const std::int64_t *n,
                              const double *x,
                              const std::int64_t *incx,
                              const double *y,
                              const std::int64_t *incy) {
    return dot_impl<double, false>(*n, x, to_stride(incx), y, to_stride(incy));
}

double naive_blas_backend_t::dsdot(const std::int64_t *n,
                               const float *x,
                               const std::int64_t *incx,
                               const float *y,
                               const std::int64_t *incy) {
    return dsdot_impl(*n, x, to_stride(incx), y, to_stride(incy));
}

compensated_blas_complex_float naive_blas_backend_t::cdotu(const std::int64_t *n,
                                                       const compensated_blas_complex_float *x,
                                                       const std::int64_t *incx,
                                                       const compensated_blas_complex_float *y,
                                                       const std::int64_t *incy) {
    const auto result = dot_impl<std::complex<float>, false>(*n,
                                                             reinterpret_cast<const std::complex<float> *>(x),
                                                             to_stride(incx),
                                                             reinterpret_cast<const std::complex<float> *>(y),
                                                             to_stride(incy));
    return from_complex(result);
}

compensated_blas_complex_float naive_blas_backend_t::cdotc(const std::int64_t *n,
                                                       const compensated_blas_complex_float *x,
                                                       const std::int64_t *incx,
                                                       const compensated_blas_complex_float *y,
                                                       const std::int64_t *incy) {
    const auto result = dot_impl<std::complex<float>, true>(*n,
                                                            reinterpret_cast<const std::complex<float> *>(x),
                                                            to_stride(incx),
                                                            reinterpret_cast<const std::complex<float> *>(y),
                                                            to_stride(incy));
    return from_complex(result);
}

compensated_blas_complex_double naive_blas_backend_t::zdotu(const std::int64_t *n,
                                                        const compensated_blas_complex_double *x,
                                                        const std::int64_t *incx,
                                                        const compensated_blas_complex_double *y,
                                                        const std::int64_t *incy) {
    const auto result = dot_impl<std::complex<double>, false>(*n,
                                                              reinterpret_cast<const std::complex<double> *>(x),
                                                              to_stride(incx),
                                                              reinterpret_cast<const std::complex<double> *>(y),
                                                              to_stride(incy));
    return from_complex(result);
}

compensated_blas_complex_double naive_blas_backend_t::zdotc(const std::int64_t *n,
                                                        const compensated_blas_complex_double *x,
                                                        const std::int64_t *incx,
                                                        const compensated_blas_complex_double *y,
                                                        const std::int64_t *incy) {
    const auto result = dot_impl<std::complex<double>, true>(*n,
                                                             reinterpret_cast<const std::complex<double> *>(x),
                                                             to_stride(incx),
                                                             reinterpret_cast<const std::complex<double> *>(y),
                                                             to_stride(incy));
    return from_complex(result);
}

void naive_blas_backend_t::saxpy(const std::int64_t *n,
                             const float *alpha,
                             const float *x,
                             const std::int64_t *incx,
                             float *y,
                             const std::int64_t *incy) {
    axpy_impl(*n, *alpha, x, to_stride(incx), y, to_stride(incy));
}

void naive_blas_backend_t::daxpy(const std::int64_t *n,
                             const double *alpha,
                             const double *x,
                             const std::int64_t *incx,
                             double *y,
                             const std::int64_t *incy) {
    axpy_impl(*n, *alpha, x, to_stride(incx), y, to_stride(incy));
}

void naive_blas_backend_t::caxpy(const std::int64_t *n,
                             const compensated_blas_complex_float *alpha,
                             const compensated_blas_complex_float *x,
                             const std::int64_t *incx,
                             compensated_blas_complex_float *y,
                             const std::int64_t *incy) {
    axpy_impl(*n,
              to_complex(*alpha),
              reinterpret_cast<const std::complex<float> *>(x),
              to_stride(incx),
              reinterpret_cast<std::complex<float> *>(y),
              to_stride(incy));
}

void naive_blas_backend_t::zaxpy(const std::int64_t *n,
                             const compensated_blas_complex_double *alpha,
                             const compensated_blas_complex_double *x,
                             const std::int64_t *incx,
                             compensated_blas_complex_double *y,
                             const std::int64_t *incy) {
    axpy_impl(*n,
              to_complex(*alpha),
              reinterpret_cast<const std::complex<double> *>(x),
              to_stride(incx),
              reinterpret_cast<std::complex<double> *>(y),
              to_stride(incy));
}

void naive_blas_backend_t::sgemv(const char *trans,
                             const std::int64_t *m,
                             const std::int64_t *n,
                             const float *alpha,
                             const float *a,
                             const std::int64_t *lda,
                             const float *x,
                             const std::int64_t *incx,
                             const float *beta,
                             float *y,
                             const std::int64_t *incy) {
    gemv_impl<float>("SGEMV ",
                     trans,
                     *m,
                     *n,
                     *alpha,
                     a,
                     *lda,
                     x,
                     to_stride(incx),
                     *beta,
                     y,
                     to_stride(incy));
}

void naive_blas_backend_t::dgemv(const char *trans,
                             const std::int64_t *m,
                             const std::int64_t *n,
                             const double *alpha,
                             const double *a,
                             const std::int64_t *lda,
                             const double *x,
                             const std::int64_t *incx,
                             const double *beta,
                             double *y,
                             const std::int64_t *incy) {
    gemv_impl<double>("DGEMV ",
                      trans,
                      *m,
                      *n,
                      *alpha,
                      a,
                      *lda,
                      x,
                      to_stride(incx),
                      *beta,
                      y,
                      to_stride(incy));
}

void naive_blas_backend_t::sgbmv(const char *trans,
                             const std::int64_t *m,
                             const std::int64_t *n,
                             const std::int64_t *kl,
                             const std::int64_t *ku,
                             const float *alpha,
                             const float *a,
                             const std::int64_t *lda,
                             const float *x,
                             const std::int64_t *incx,
                             const float *beta,
                             float *y,
                             const std::int64_t *incy) {
    gbmv_impl<float>("SGBMV ",
                     trans,
                     *m,
                     *n,
                     *kl,
                     *ku,
                     *alpha,
                     a,
                     *lda,
                     x,
                     to_stride(incx),
                     *beta,
                     y,
                     to_stride(incy));
}

void naive_blas_backend_t::dgbmv(const char *trans,
                             const std::int64_t *m,
                             const std::int64_t *n,
                             const std::int64_t *kl,
                             const std::int64_t *ku,
                             const double *alpha,
                             const double *a,
                             const std::int64_t *lda,
                             const double *x,
                             const std::int64_t *incx,
                             const double *beta,
                             double *y,
                             const std::int64_t *incy) {
    gbmv_impl<double>("DGBMV ",
                      trans,
                      *m,
                      *n,
                      *kl,
                      *ku,
                      *alpha,
                      a,
                      *lda,
                      x,
                      to_stride(incx),
                      *beta,
                      y,
                      to_stride(incy));
}

void naive_blas_backend_t::ssymv(const char *uplo,
                             const std::int64_t *n,
                             const float *alpha,
                             const float *a,
                             const std::int64_t *lda,
                             const float *x,
                             const std::int64_t *incx,
                             const float *beta,
                             float *y,
                             const std::int64_t *incy) {
    symv_impl<float>("SSYMV ",
                     uplo,
                     *n,
                     *alpha,
                     a,
                     *lda,
                     x,
                     to_stride(incx),
                     *beta,
                     y,
                     to_stride(incy));
}

void naive_blas_backend_t::dsymv(const char *uplo,
                             const std::int64_t *n,
                             const double *alpha,
                             const double *a,
                             const std::int64_t *lda,
                             const double *x,
                             const std::int64_t *incx,
                             const double *beta,
                             double *y,
                             const std::int64_t *incy) {
    symv_impl<double>("DSYMV ",
                      uplo,
                      *n,
                      *alpha,
                      a,
                      *lda,
                      x,
                      to_stride(incx),
                      *beta,
                      y,
                      to_stride(incy));
}

void naive_blas_backend_t::ssbmv(const char *uplo,
                             const std::int64_t *n,
                             const std::int64_t *k,
                             const float *alpha,
                             const float *a,
                             const std::int64_t *lda,
                             const float *x,
                             const std::int64_t *incx,
                             const float *beta,
                             float *y,
                             const std::int64_t *incy) {
    sbmv_impl<float>("SSBMV ",
                     uplo,
                     *n,
                     *k,
                     *alpha,
                     a,
                     *lda,
                     x,
                     to_stride(incx),
                     *beta,
                     y,
                     to_stride(incy));
}

void naive_blas_backend_t::dsbmv(const char *uplo,
                             const std::int64_t *n,
                             const std::int64_t *k,
                             const double *alpha,
                             const double *a,
                             const std::int64_t *lda,
                             const double *x,
                             const std::int64_t *incx,
                             const double *beta,
                             double *y,
                             const std::int64_t *incy) {
    sbmv_impl<double>("DSBMV ",
                      uplo,
                      *n,
                      *k,
                      *alpha,
                      a,
                      *lda,
                      x,
                      to_stride(incx),
                      *beta,
                      y,
                      to_stride(incy));
}

void naive_blas_backend_t::sspmv(const char *uplo,
                             const std::int64_t *n,
                             const float *alpha,
                             const float *ap,
                             const float *x,
                             const std::int64_t *incx,
                             const float *beta,
                             float *y,
                             const std::int64_t *incy) {
    spmv_impl<float>("SSPMV ",
                     uplo,
                     *n,
                     *alpha,
                     ap,
                     x,
                     to_stride(incx),
                     *beta,
                     y,
                     to_stride(incy));
}

void naive_blas_backend_t::dspmv(const char *uplo,
                             const std::int64_t *n,
                             const double *alpha,
                             const double *ap,
                             const double *x,
                             const std::int64_t *incx,
                             const double *beta,
                             double *y,
                             const std::int64_t *incy) {
    spmv_impl<double>("DSPMV ",
                      uplo,
                      *n,
                      *alpha,
                      ap,
                      x,
                      to_stride(incx),
                      *beta,
                      y,
                      to_stride(incy));
}

void naive_blas_backend_t::strsv(const char *uplo,
                             const char *trans,
                             const char *diag,
                             const std::int64_t *n,
                             const float *a,
                             const std::int64_t *lda,
                             float *x,
                             const std::int64_t *incx) {
    trsv_impl<float>("STRSV ",
                     uplo,
                     trans,
                     diag,
                     *n,
                     a,
                     static_cast<std::size_t>(*lda),
                     x,
                     to_stride(incx));
}

void naive_blas_backend_t::dtrsv(const char *uplo,
                             const char *trans,
                             const char *diag,
                             const std::int64_t *n,
                             const double *a,
                             const std::int64_t *lda,
                             double *x,
                             const std::int64_t *incx) {
    trsv_impl<double>("DTRSV ",
                      uplo,
                      trans,
                      diag,
                      *n,
                      a,
                      static_cast<std::size_t>(*lda),
                      x,
                      to_stride(incx));
}

void naive_blas_backend_t::strmv(const char *uplo,
                             const char *trans,
                             const char *diag,
                             const std::int64_t *n,
                             const float *a,
                             const std::int64_t *lda,
                             float *x,
                             const std::int64_t *incx) {
    trmv_impl<float>("STRMV ",
                     uplo,
                     trans,
                     diag,
                     *n,
                     a,
                     static_cast<std::size_t>(*lda),
                     x,
                     to_stride(incx));
}

void naive_blas_backend_t::dtrmv(const char *uplo,
                             const char *trans,
                             const char *diag,
                             const std::int64_t *n,
                             const double *a,
                             const std::int64_t *lda,
                             double *x,
                             const std::int64_t *incx) {
    trmv_impl<double>("DTRMV ",
                      uplo,
                      trans,
                      diag,
                      *n,
                      a,
                      static_cast<std::size_t>(*lda),
                      x,
                      to_stride(incx));
}

void naive_blas_backend_t::ssyrk(const char *uplo,
                             const char *trans,
                             const std::int64_t *n,
                             const std::int64_t *k,
                             const float *alpha,
                             const float *a,
                             const std::int64_t *lda,
                             const float *beta,
                             float *c,
                             const std::int64_t *ldc) {
    const deferred_matrix_metadata_t deferred = fetch_deferred_matrix(c, static_cast<std::size_t>(*ldc), false);
    syrk_impl(uplo, trans, *n, *k, *alpha, a, static_cast<std::size_t>(*lda), *beta, c, static_cast<std::size_t>(*ldc), false, deferred);
}

void naive_blas_backend_t::dsyrk(const char *uplo,
                             const char *trans,
                             const std::int64_t *n,
                             const std::int64_t *k,
                             const double *alpha,
                             const double *a,
                             const std::int64_t *lda,
                             const double *beta,
                             double *c,
                             const std::int64_t *ldc) {
    const deferred_matrix_metadata_t deferred = fetch_deferred_matrix(c, static_cast<std::size_t>(*ldc), false);
    syrk_impl(uplo, trans, *n, *k, *alpha, a, static_cast<std::size_t>(*lda), *beta, c, static_cast<std::size_t>(*ldc), false, deferred);
}

void naive_blas_backend_t::csyrk(const char *uplo,
                             const char *trans,
                             const std::int64_t *n,
                             const std::int64_t *k,
                             const compensated_blas_complex_float *alpha,
                             const compensated_blas_complex_float *a,
                             const std::int64_t *lda,
                             const compensated_blas_complex_float *beta,
                             compensated_blas_complex_float *c,
                             const std::int64_t *ldc) {
    const deferred_matrix_metadata_t deferred = fetch_deferred_matrix(c, static_cast<std::size_t>(*ldc), false);
    syrk_impl(uplo,
              trans,
              *n,
              *k,
              to_complex(*alpha),
              reinterpret_cast<const std::complex<float> *>(a),
              static_cast<std::size_t>(*lda),
              to_complex(*beta),
              reinterpret_cast<std::complex<float> *>(c),
              static_cast<std::size_t>(*ldc),
              false,
              deferred);
}

void naive_blas_backend_t::zsyrk(const char *uplo,
                             const char *trans,
                             const std::int64_t *n,
                             const std::int64_t *k,
                             const compensated_blas_complex_double *alpha,
                             const compensated_blas_complex_double *a,
                             const std::int64_t *lda,
                             const compensated_blas_complex_double *beta,
                             compensated_blas_complex_double *c,
                             const std::int64_t *ldc) {
    const deferred_matrix_metadata_t deferred = fetch_deferred_matrix(c, static_cast<std::size_t>(*ldc), false);
    syrk_impl(uplo,
              trans,
              *n,
              *k,
              to_complex(*alpha),
              reinterpret_cast<const std::complex<double> *>(a),
              static_cast<std::size_t>(*lda),
              to_complex(*beta),
              reinterpret_cast<std::complex<double> *>(c),
              static_cast<std::size_t>(*ldc),
              false,
              deferred);
}

naive_blas_backend_t naive_backend_instance{};

compensated_blas::impl::blas_backend_t *acquire_naive_backend_impl() {
    return &naive_backend_instance;
}

compensated_blas::impl::blas_backend_t *acquire_naive_backend() {
    return acquire_naive_backend_impl();
}

}  // namespace compensated_blas::impl::detail
