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
T reconstruct_compensated_value(const T &primary, const T *bins, std::size_t terms) {
    if (!bins || terms == 0) {
        return primary;
    }
    long double sum = static_cast<long double>(primary);
    for (std::size_t i = 0; i < terms; ++i) {
        sum += static_cast<long double>(bins[i]);
    }
    return static_cast<T>(sum);
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
