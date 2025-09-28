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

#include "compensated_blas.hpp"
#include "impl/backend_stub.hpp"
#include "impl/compensated_arithmetic.hpp"

namespace compensated_blas::impl::detail {

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

class naive_blas_backend_t final : public stub_backend_t {
public:
    float sdot(const std::int64_t *n, const float *x, const std::int64_t *incx, const float *y, const std::int64_t *incy) override;
    float sdsdot(const std::int64_t *n, const float *sb, const float *x, const std::int64_t *incx, const float *y, const std::int64_t *incy) override;
    double ddot(const std::int64_t *n, const double *x, const std::int64_t *incx, const double *y, const std::int64_t *incy) override;
    double dsdot(const std::int64_t *n, const float *x, const std::int64_t *incx, const float *y, const std::int64_t *incy) override;
    compensated_blas_complex_float cdotu(const std::int64_t *n, const compensated_blas_complex_float *x, const std::int64_t *incx, const compensated_blas_complex_float *y, const std::int64_t *incy) override;
    compensated_blas_complex_float cdotc(const std::int64_t *n, const compensated_blas_complex_float *x, const std::int64_t *incx, const compensated_blas_complex_float *y, const std::int64_t *incy) override;
    compensated_blas_complex_double zdotu(const std::int64_t *n, const compensated_blas_complex_double *x, const std::int64_t *incx, const compensated_blas_complex_double *y, const std::int64_t *incy) override;
    compensated_blas_complex_double zdotc(const std::int64_t *n, const compensated_blas_complex_double *x, const std::int64_t *incx, const compensated_blas_complex_double *y, const std::int64_t *incy) override;

    void saxpy(const std::int64_t *n, const float *alpha, const float *x, const std::int64_t *incx, float *y, const std::int64_t *incy) override;
    void daxpy(const std::int64_t *n, const double *alpha, const double *x, const std::int64_t *incx, double *y, const std::int64_t *incy) override;
    void caxpy(const std::int64_t *n, const compensated_blas_complex_float *alpha, const compensated_blas_complex_float *x, const std::int64_t *incx, compensated_blas_complex_float *y, const std::int64_t *incy) override;
    void zaxpy(const std::int64_t *n, const compensated_blas_complex_double *alpha, const compensated_blas_complex_double *x, const std::int64_t *incx, compensated_blas_complex_double *y, const std::int64_t *incy) override;

    void ssyrk(const char *uplo, const char *trans, const std::int64_t *n, const std::int64_t *k, const float *alpha, const float *a, const std::int64_t *lda, const float *beta, float *c, const std::int64_t *ldc) override;
    void dsyrk(const char *uplo, const char *trans, const std::int64_t *n, const std::int64_t *k, const double *alpha, const double *a, const std::int64_t *lda, const double *beta, double *c, const std::int64_t *ldc) override;
    void csyrk(const char *uplo, const char *trans, const std::int64_t *n, const std::int64_t *k, const compensated_blas_complex_float *alpha, const compensated_blas_complex_float *a, const std::int64_t *lda, const compensated_blas_complex_float *beta, compensated_blas_complex_float *c, const std::int64_t *ldc) override;
    void zsyrk(const char *uplo, const char *trans, const std::int64_t *n, const std::int64_t *k, const compensated_blas_complex_double *alpha, const compensated_blas_complex_double *a, const std::int64_t *lda, const compensated_blas_complex_double *beta, compensated_blas_complex_double *c, const std::int64_t *ldc) override;
};

// Helper to compute dot product with optional conjugation of X
template <typename T, bool ConjugateX>
T dot_impl(std::int64_t count, const T *x, std::ptrdiff_t incx, const T *y, std::ptrdiff_t incy) {
    if (count <= 0) {
        return T{};
    }
    const std::size_t terms = compensated_blas::runtime::compensation_terms();
    T primary{};
    std::vector<T> compensation(terms, T{});
    auto *bins = compensation.data();

    const T *x_ptr = x;
    const T *y_ptr = y;
    for (std::int64_t i = 0; i < count; ++i) {
        T left = x_ptr[i * incx];
        if constexpr (scalar_traits_t<T>::is_complex && ConjugateX) {
            left = std::conj(left);
        }
        const T right = y_ptr[i * incy];
        accumulate_value(primary, bins, terms, left * right);
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
    std::vector<double> compensation(terms, 0.0);
    auto *bins = compensation.data();

    for (std::int64_t i = 0; i < count; ++i) {
        const double product = static_cast<double>(x[i * incx]) * static_cast<double>(y[i * incy]);
        accumulate_value(primary, bins, terms, product);
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
    std::vector<double> compensation(terms, 0.0);
    auto *bins = compensation.data();
    for (std::int64_t i = 0; i < count; ++i) {
        const double product = static_cast<double>(x[i * incx]) * static_cast<double>(y[i * incy]);
        accumulate_value(primary, bins, terms, product);
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

    std::vector<T> local_bins(global_terms, T{});
    T *local_bins_ptr = global_terms > 0 ? local_bins.data() : nullptr;

    for (std::int64_t i = 0; i < count; ++i) {
        const T contribution = alpha * x[i * incx];
        T &destination = y[i * incy];

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
    std::vector<T> local_bins(global_terms, T{});
    T *local_bins_ptr = global_terms > 0 ? local_bins.data() : nullptr;

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
