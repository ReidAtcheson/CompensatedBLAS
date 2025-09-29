#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <complex>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "impl/compensated_arithmetic.hpp"
#include "compensated_blas.hpp"
#include "impl/compensated_accumulator.hpp"
#include "impl/compensated_blas_backend_ilp64.hpp"
#include "impl/naive_blas_backend.hpp"

TEST(compensated_blas_sanity, identity_returns_input) {
    constexpr compensated_blas::index_t value = 123;
    EXPECT_EQ(compensated_blas::identity(value), value);
}

namespace {

template <compensated_blas::accumulator_layout Layout>
void check_accumulates_without_compensation() {
    std::array<double, 3> working{};
    compensated_blas::compensated_accumulator_t<double, Layout> acc(nullptr, working.size(), 0);
    acc.accumulate(0, working[0], 1.0);
    acc.accumulate(0, working[0], 2.0);
    EXPECT_DOUBLE_EQ(acc.round(0, working[0]), 3.0);
}

template <compensated_blas::accumulator_layout Layout>
void check_accumulates_with_single_compensation() {
    std::array<double, 1> working{};
    std::array<double, 1> bins{};
    compensated_blas::compensated_accumulator_t<double, Layout> acc(bins.data(), working.size(), 1);
    acc.accumulate(0, working[0], 1e16);
    acc.accumulate(0, working[0], 1.0);
    acc.round(0, working[0]);
    EXPECT_DOUBLE_EQ(working[0], 1e16 + 1.0);
}

template <compensated_blas::accumulator_layout Layout>
void check_uses_external_storage() {
    std::array<double, 4> working{};
    std::array<double, 8> bins{};
    compensated_blas::compensated_accumulator_t<double, Layout> acc(bins.data(), working.size(), 2);
    acc.accumulate(1, working[1], 3.5);
    EXPECT_DOUBLE_EQ(acc.round(1, working[1]), 3.5);
}

template <compensated_blas::accumulator_layout Layout>
void check_multiple_compensation_bins() {
    std::array<double, 1> working{};
    std::array<double, 3> bins{};
    compensated_blas::compensated_accumulator_t<double, Layout> acc(bins.data(), working.size(), 3);
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

TEST(compensated_accumulator_test, uses_external_storage_soa) {
    check_uses_external_storage<compensated_blas::accumulator_layout::soa>();
}

TEST(compensated_accumulator_test, uses_external_storage_aos) {
    check_uses_external_storage<compensated_blas::accumulator_layout::aos>();
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
    compensated_blas::runtime::arena_config_t cfg;
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

    compensated_blas::runtime::preallocation_request_t request{};
    request.deferred_rounding_matrices = 2;
    request.deferred_rounding_vectors = 2;
    compensated_blas::runtime::preallocate(request);

    compensated_blas::runtime::deferred_rounding_matrix_t matrix{};
    matrix.data = reinterpret_cast<void *>(0x1234);
    matrix.rows = 8;
    matrix.cols = 8;
    matrix.leading_dimension = 8;
    matrix.element_size = sizeof(double);
    matrix.alignment = alignof(double);
    matrix.type = compensated_blas::runtime::scalar_type_t::real64;
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

    compensated_blas::runtime::deferred_rounding_vector_t vector{};
    vector.data = reinterpret_cast<void *>(0x5678);
    vector.length = 16;
    vector.stride = 1;
    vector.element_size = sizeof(double);
    vector.alignment = alignof(double);
    vector.type = compensated_blas::runtime::scalar_type_t::real64;

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

namespace {

template <typename T>
T finalize_with_bins(T &primary, T *bins, std::size_t terms) {
    if (!bins || terms == 0) {
        return primary;
    }
    if constexpr (std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>) {
        using real_t = typename T::value_type;
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
            if (carry == T{}) {
                continue;
            }
            compensated_blas::two_sum(sum, carry);
        }
        if (carry != T{}) {
            sum += carry;
        }
        primary = sum;
        return sum;
    }
}

}  // namespace

TEST(compensated_runtime_config, updates_compensation_terms) {
    compensated_blas::runtime::set_default_allocatr();
    compensated_blas::runtime::clear_deferred_rounding_registrations();
    const std::size_t previous_terms = compensated_blas::runtime::compensation_terms();
    compensated_blas::runtime::set_compensation_terms(2);
    EXPECT_EQ(compensated_blas::runtime::compensation_terms(), 2u);

    compensated_blas::runtime::deferred_rounding_vector_t vector_desc{};
    std::array<float, 4> data{};
    vector_desc.data = data.data();
    vector_desc.length = data.size();
    vector_desc.stride = 1;
    vector_desc.element_size = sizeof(float);
    vector_desc.alignment = alignof(float);
    vector_desc.type = compensated_blas::runtime::scalar_type_t::real32;
    compensated_blas::runtime::register_deferred_rounding_vector(vector_desc);

    compensated_blas::runtime::set_compensation_terms(3);
    auto updated = compensated_blas::runtime::find_deferred_rounding_vector(data.data());
    ASSERT_TRUE(updated.has_value());
    EXPECT_EQ(updated->compensation_terms, 3u);

    compensated_blas::runtime::clear_deferred_rounding_registrations();
}

TEST(compensated_naive_backend, dot_and_axpy_immediate) {
    compensated_blas::runtime::set_backend(compensated_blas::runtime::backend_kind_t::naive);
    std::int64_t n = 3;
    std::int64_t inc = 1;
    float alpha = 2.0f;
    std::array<float, 3> x{1.0f, 2.0f, -3.0f};
    std::array<float, 3> y{0.5f, -1.0f, 4.0f};

    auto &backend = compensated_blas::impl::get_active_ilp64_backend();
    float dot = backend.sdot(&n, x.data(), &inc, y.data(), &inc);
    EXPECT_NEAR(dot, -13.5f, 1e-6f);

    backend.saxpy(&n, &alpha, x.data(), &inc, y.data(), &inc);
    EXPECT_NEAR(y[0], 2.5f, 1e-6f);
    EXPECT_NEAR(y[1], 3.0f, 1e-6f);
    EXPECT_NEAR(y[2], -2.0f, 1e-6f);

    compensated_blas::runtime::set_backend(compensated_blas::runtime::backend_kind_t::empty);
}

TEST(compensated_naive_backend, axpy_deferred_rounding) {
    compensated_blas::runtime::set_backend(compensated_blas::runtime::backend_kind_t::naive);
    compensated_blas::runtime::set_compensation_terms(2);
    compensated_blas::runtime::clear_deferred_rounding_registrations();

    constexpr std::int64_t n = 4;
    std::int64_t inc = 1;
    double alpha = 1.0;
    std::array<double, n> x{1.0, 1e16, -1e16, 2.0};
    std::array<double, n> y{0.0, 0.0, 0.0, 0.0};

    compensated_blas::runtime::deferred_rounding_vector_t vec{};
    vec.data = y.data();
    vec.length = y.size();
    vec.stride = 1;
    vec.element_size = sizeof(double);
    vec.alignment = alignof(double);
    vec.type = compensated_blas::runtime::scalar_type_t::real64;
    compensated_blas::runtime::register_deferred_rounding_vector(vec);

    auto &backend = compensated_blas::impl::get_active_ilp64_backend();
    backend.daxpy(&n, &alpha, x.data(), &inc, y.data(), &inc);

    auto descriptor = compensated_blas::runtime::find_deferred_rounding_vector(y.data());
    ASSERT_TRUE(descriptor.has_value());
    auto *bins = static_cast<double *>(descriptor->compensation);
    ASSERT_NE(bins, nullptr);

    for (std::size_t i = 0; i < y.size(); ++i) {
        double &primary = y[i];
        double *element_bins = bins + i * descriptor->compensation_terms;
        (void)finalize_with_bins(primary, element_bins, descriptor->compensation_terms);
    }

    EXPECT_NEAR(y[0], 1.0, 1e-9);
    EXPECT_NEAR(y[1], 1e16, 1e-3);
    EXPECT_NEAR(y[2], -1e16, 1e-3);
    EXPECT_NEAR(y[3], 2.0, 1e-9);

    compensated_blas::runtime::clear_deferred_rounding_registrations();
    compensated_blas::runtime::set_backend(compensated_blas::runtime::backend_kind_t::empty);
}

TEST(compensated_naive_backend, syrk_immediate_and_deferred) {
    compensated_blas::runtime::set_backend(compensated_blas::runtime::backend_kind_t::naive);
    compensated_blas::runtime::set_compensation_terms(2);
    compensated_blas::runtime::clear_deferred_rounding_registrations();

    const char uplo = 'L';
    const char trans = 'N';
    std::int64_t n = 2;
    std::int64_t k = 3;
    std::int64_t lda = 2;
    std::int64_t ldc = 2;
    double alpha = 1.0;
    double beta = 0.0;
    double a[] = {1.0, 2.0, -1.0, 3.0, 0.5, -0.5};
    double c[] = {0.0, 0.0, 0.0, 0.0};

    auto &backend = compensated_blas::impl::get_active_ilp64_backend();
    backend.dsyrk(&uplo, &trans, &n, &k, &alpha, a, &lda, &beta, c, &ldc);

    EXPECT_NEAR(c[0], 1.0 * 1.0 + (-1.0) * (-1.0) + 0.5 * 0.5, 1e-9);
    EXPECT_NEAR(c[1], 1.0 * 2.0 + (-1.0) * 3.0 + 0.5 * (-0.5), 1e-9);
    EXPECT_NEAR(c[3], 2.0 * 2.0 + 3.0 * 3.0 + (-0.5) * (-0.5), 1e-9);

    compensated_blas::runtime::deferred_rounding_matrix_t matrix_desc{};
    matrix_desc.data = c;
    matrix_desc.rows = 2;
    matrix_desc.cols = 2;
    matrix_desc.leading_dimension = ldc;
    matrix_desc.element_size = sizeof(double);
    matrix_desc.alignment = alignof(double);
    matrix_desc.type = compensated_blas::runtime::scalar_type_t::real64;
    matrix_desc.row_major = false;
    compensated_blas::runtime::register_deferred_rounding_matrix(matrix_desc);

    double beta_one = 1.0;
    backend.dsyrk(&uplo, &trans, &n, &k, &alpha, a, &lda, &beta_one, c, &ldc);

    auto matrix_descriptor = compensated_blas::runtime::find_deferred_rounding_matrix(c);
    ASSERT_TRUE(matrix_descriptor.has_value());
    auto *matrix_bins = static_cast<double *>(matrix_descriptor->compensation);
    ASSERT_NE(matrix_bins, nullptr);
    for (std::size_t col = 0; col < static_cast<std::size_t>(n); ++col) {
        for (std::size_t row = col; row < static_cast<std::size_t>(n); ++row) {
            double &primary = c[col * ldc + row];
            double *element_bins = matrix_bins + (col * ldc + row) * matrix_descriptor->compensation_terms;
            (void)finalize_with_bins(primary, element_bins, matrix_descriptor->compensation_terms);
        }
    }

    EXPECT_NEAR(c[0], 2.0 * (1.0 * 1.0 + (-1.0) * (-1.0) + 0.5 * 0.5), 1e-6);
    EXPECT_NEAR(c[1], 2.0 * (1.0 * 2.0 + (-1.0) * 3.0 + 0.5 * (-0.5)), 1e-6);
    EXPECT_NEAR(c[3], 2.0 * (2.0 * 2.0 + 3.0 * 3.0 + (-0.5) * (-0.5)), 1e-6);

    compensated_blas::runtime::clear_deferred_rounding_registrations();
    compensated_blas::runtime::set_backend(compensated_blas::runtime::backend_kind_t::empty);
}

TEST(compensated_naive_backend, level1_real_immediate) {
    compensated_blas::runtime::set_backend(compensated_blas::runtime::backend_kind_t::naive);
    compensated_blas::runtime::set_compensation_terms(2);
    compensated_blas::runtime::clear_deferred_rounding_registrations();

    std::int64_t n = 3;
    std::int64_t inc = 1;

    std::array<float, 3> xf{1.0f, -2.0f, 3.0f};
    std::array<float, 3> yf{4.0f, 5.0f, -6.0f};

    auto &backend = compensated_blas::impl::get_active_ilp64_backend();
    backend.sswap(&n, xf.data(), &inc, yf.data(), &inc);
    EXPECT_FLOAT_EQ(xf[0], 4.0f);
    EXPECT_FLOAT_EQ(xf[1], 5.0f);
    EXPECT_FLOAT_EQ(xf[2], -6.0f);
    EXPECT_FLOAT_EQ(yf[0], 1.0f);
    EXPECT_FLOAT_EQ(yf[1], -2.0f);
    EXPECT_FLOAT_EQ(yf[2], 3.0f);

    float alpha = 0.5f;
    backend.sscal(&n, &alpha, xf.data(), &inc);
    EXPECT_FLOAT_EQ(xf[0], 2.0f);
    EXPECT_FLOAT_EQ(xf[1], 2.5f);
    EXPECT_FLOAT_EQ(xf[2], -3.0f);

    backend.scopy(&n, yf.data(), &inc, xf.data(), &inc);
    EXPECT_FLOAT_EQ(xf[0], 1.0f);
    EXPECT_FLOAT_EQ(xf[1], -2.0f);
    EXPECT_FLOAT_EQ(xf[2], 3.0f);

    std::array<double, 4> xd{1.0, -4.0, 3.0, -2.0};
    std::array<double, 4> yd{2.0, 5.0, -1.0, 4.0};
    n = 2;
    std::int64_t inc2 = 2;  // test non-unit stride
    backend.dswap(&n, xd.data(), &inc2, yd.data(), &inc2);
    EXPECT_DOUBLE_EQ(xd[0], 2.0);
    EXPECT_DOUBLE_EQ(xd[2], -1.0);
    EXPECT_DOUBLE_EQ(yd[0], 1.0);
    EXPECT_DOUBLE_EQ(yd[2], 3.0);

    double beta = -2.0;
    backend.dscal(&n, &beta, xd.data(), &inc2);
    EXPECT_DOUBLE_EQ(xd[0], -4.0);
    EXPECT_DOUBLE_EQ(xd[2], 2.0);

    backend.dcopy(&n, yd.data(), &inc2, xd.data(), &inc2);
    EXPECT_DOUBLE_EQ(xd[0], 1.0);
    EXPECT_DOUBLE_EQ(xd[2], 3.0);

    compensated_blas::runtime::clear_deferred_rounding_registrations();
    compensated_blas::runtime::set_backend(compensated_blas::runtime::backend_kind_t::empty);
}

TEST(compensated_naive_backend, level1_real_deferred) {
    compensated_blas::runtime::set_backend(compensated_blas::runtime::backend_kind_t::naive);
    compensated_blas::runtime::set_compensation_terms(2);
    compensated_blas::runtime::clear_deferred_rounding_registrations();

    std::int64_t n = 3;
    std::int64_t inc = 1;

    std::array<double, 3> x{1.0, 1e16, -1e16};
    std::array<double, 3> y{-2.0, 4.0, 5.0};

    compensated_blas::runtime::deferred_rounding_vector_t desc{};
    desc.data = x.data();
    desc.length = x.size();
    desc.stride = 1;
    desc.element_size = sizeof(double);
    desc.alignment = alignof(double);
    desc.type = compensated_blas::runtime::scalar_type_t::real64;
    compensated_blas::runtime::register_deferred_rounding_vector(desc);

    desc.data = y.data();
    compensated_blas::runtime::register_deferred_rounding_vector(desc);

    auto &backend = compensated_blas::impl::get_active_ilp64_backend();
    backend.dswap(&n, x.data(), &inc, y.data(), &inc);
    EXPECT_DOUBLE_EQ(x[0], -2.0);
    EXPECT_DOUBLE_EQ(y[1], 1e16);

    double alpha = 1e-6;
    backend.dscal(&n, &alpha, x.data(), &inc);
    backend.dcopy(&n, y.data(), &inc, x.data(), &inc);

    auto x_descriptor = compensated_blas::runtime::find_deferred_rounding_vector(x.data());
    ASSERT_TRUE(x_descriptor.has_value());
    ASSERT_NE(x_descriptor->compensation, nullptr);
    double *x_bins = static_cast<double *>(x_descriptor->compensation);
    for (std::size_t i = 0; i < x_descriptor->compensation_elements * x_descriptor->compensation_terms; ++i) {
        EXPECT_DOUBLE_EQ(x_bins[i], 0.0);
    }

    compensated_blas::runtime::clear_deferred_rounding_registrations();
    compensated_blas::runtime::set_backend(compensated_blas::runtime::backend_kind_t::empty);
}

TEST(compensated_naive_backend, level1_complex_immediate) {
    compensated_blas::runtime::set_backend(compensated_blas::runtime::backend_kind_t::naive);
    compensated_blas::runtime::set_compensation_terms(2);
    compensated_blas::runtime::clear_deferred_rounding_registrations();

    std::array<compensated_blas_complex_float, 3> x{{{1.0f, 2.0f}, {3.0f, -4.0f}, {5.0f, 0.5f}}};
    std::array<compensated_blas_complex_float, 3> y{{{7.0f, -1.0f}, {2.0f, 6.0f}, {-3.0f, 4.0f}}};
    std::array<std::complex<float>, 3> expected_x{std::complex<float>(1.0f, 2.0f),
                                                  std::complex<float>(3.0f, -4.0f),
                                                  std::complex<float>(5.0f, 0.5f)};
    std::array<std::complex<float>, 3> expected_y{std::complex<float>(7.0f, -1.0f),
                                                  std::complex<float>(2.0f, 6.0f),
                                                  std::complex<float>(-3.0f, 4.0f)};

    auto expect_match = [](const std::array<compensated_blas_complex_float, 3> &actual,
                           const std::array<std::complex<float>, 3> &expected,
                           float tol) {
        for (std::size_t i = 0; i < actual.size(); ++i) {
            EXPECT_NEAR(actual[i].real, expected[i].real(), tol);
            EXPECT_NEAR(actual[i].imag, expected[i].imag(), tol);
        }
    };

    std::int64_t n = 3;
    std::int64_t inc = 1;
    auto &backend = compensated_blas::impl::get_active_ilp64_backend();

    backend.cswap(&n, x.data(), &inc, y.data(), &inc);
    for (std::size_t i = 0; i < expected_x.size(); ++i) {
        std::swap(expected_x[i], expected_y[i]);
    }
    expect_match(x, expected_x, 1e-6f);
    expect_match(y, expected_y, 1e-6f);

    float csscal_alpha = 2.0f;
    backend.csscal(&n, &csscal_alpha, x.data(), &inc);
    for (auto &value : expected_x) {
        value *= csscal_alpha;
    }
    expect_match(x, expected_x, 1e-6f);

    compensated_blas_complex_float cscal_alpha{0.0f, -1.0f};
    backend.cscal(&n, &cscal_alpha, y.data(), &inc);
    const std::complex<float> cscal_alpha_complex(0.0f, -1.0f);
    for (auto &value : expected_y) {
        value *= cscal_alpha_complex;
    }
    expect_match(y, expected_y, 1e-6f);

    backend.ccopy(&n, y.data(), &inc, x.data(), &inc);
    expected_x = expected_y;
    expect_match(x, expected_x, 1e-6f);

    compensated_blas_complex_float axpy_alpha{1.0f, 1.0f};
    backend.caxpy(&n, &axpy_alpha, y.data(), &inc, x.data(), &inc);
    const std::complex<float> axpy_alpha_complex(1.0f, 1.0f);
    for (std::size_t i = 0; i < expected_x.size(); ++i) {
        expected_x[i] += axpy_alpha_complex * expected_y[i];
    }
    expect_match(x, expected_x, 1e-5f);

    float norm = backend.scnrm2(&n, x.data(), &inc);
    double expected_norm = 0.0;
    for (const auto &value : expected_x) {
        expected_norm += static_cast<double>(std::norm(value));
    }
    expected_norm = std::sqrt(expected_norm);
    EXPECT_NEAR(norm, static_cast<float>(expected_norm), 1e-5f);

    float asum = backend.scasum(&n, x.data(), &inc);
    double expected_asum = 0.0;
    for (const auto &value : expected_x) {
        expected_asum += std::abs(value.real()) + std::abs(value.imag());
    }
    EXPECT_NEAR(asum, static_cast<float>(expected_asum), 1e-5f);

    std::int64_t icamax_index = backend.icamax(&n, x.data(), &inc);
    std::int64_t expected_icamax = 1;
    float max_abs1 = -1.0f;
    for (std::size_t i = 0; i < expected_x.size(); ++i) {
        const float abs1 = std::abs(expected_x[i].real()) + std::abs(expected_x[i].imag());
        if (abs1 > max_abs1) {
            max_abs1 = abs1;
            expected_icamax = static_cast<std::int64_t>(i + 1);
        }
    }
    EXPECT_EQ(icamax_index, expected_icamax);

    float rot_c = 0.6f;
    float rot_s = -0.8f;
    backend.csrot(&n, x.data(), &inc, y.data(), &inc, &rot_c, &rot_s);
    for (std::size_t i = 0; i < expected_x.size(); ++i) {
        const std::complex<float> old_x = expected_x[i];
        const std::complex<float> old_y = expected_y[i];
        expected_x[i] = rot_c * old_x + rot_s * old_y;
        expected_y[i] = rot_c * old_y - rot_s * old_x;
    }
    expect_match(x, expected_x, 1e-5f);
    expect_match(y, expected_y, 1e-5f);

    compensated_blas_complex_float crotg_a{3.0f, 4.0f};
    compensated_blas_complex_float crotg_b{1.0f, -2.0f};
    float crotg_c = 0.0f;
    compensated_blas_complex_float crotg_s{};
    backend.crotg(&crotg_a, &crotg_b, &crotg_c, &crotg_s);

    std::complex<float> a_complex(3.0f, 4.0f);
    std::complex<float> b_complex(1.0f, -2.0f);
    float expected_crotg_c;
    std::complex<float> expected_crotg_s;
    std::complex<float> expected_crotg_r;
    const float abs_a = std::abs(a_complex);
    const float abs_b = std::abs(b_complex);
    if (abs_a == 0.0f && abs_b == 0.0f) {
        expected_crotg_c = 1.0f;
        expected_crotg_s = std::complex<float>(0.0f, 0.0f);
        expected_crotg_r = std::complex<float>(0.0f, 0.0f);
    } else if (abs_a == 0.0f) {
        expected_crotg_c = 0.0f;
        expected_crotg_s = std::complex<float>(1.0f, 0.0f);
        expected_crotg_r = b_complex;
    } else {
        const float scale = abs_a + abs_b;
        const float norm = scale * std::sqrt((abs_a / scale) * (abs_a / scale) + (abs_b / scale) * (abs_b / scale));
        const std::complex<float> alpha = a_complex / abs_a;
        expected_crotg_c = abs_a / norm;
        expected_crotg_s = alpha * std::conj(b_complex) / norm;
        expected_crotg_r = alpha * norm;
    }
    EXPECT_NEAR(crotg_c, expected_crotg_c, 1e-5f);
    EXPECT_NEAR(crotg_s.real, expected_crotg_s.real(), 1e-5f);
    EXPECT_NEAR(crotg_s.imag, expected_crotg_s.imag(), 1e-5f);
    EXPECT_NEAR(crotg_a.real, expected_crotg_r.real(), 1e-5f);
    EXPECT_NEAR(crotg_a.imag, expected_crotg_r.imag(), 1e-5f);

    compensated_blas::runtime::set_backend(compensated_blas::runtime::backend_kind_t::empty);
}

TEST(compensated_naive_backend, level1_complex_deferred) {
    compensated_blas::runtime::set_backend(compensated_blas::runtime::backend_kind_t::naive);
    compensated_blas::runtime::set_compensation_terms(2);
    compensated_blas::runtime::clear_deferred_rounding_registrations();

    std::array<compensated_blas_complex_double, 3> x{{{1.0, -2.0}, {1e16, -1e16}, {-1e16, 2e16}}};
    std::array<compensated_blas_complex_double, 3> y{{{3.0, 4.0}, {-5.0, 6.0}, {7.5, -8.5}}};
    std::array<std::complex<double>, 3> expected_x{std::complex<double>(1.0, -2.0),
                                                   std::complex<double>(1e16, -1e16),
                                                   std::complex<double>(-1e16, 2e16)};
    std::array<std::complex<double>, 3> expected_y{std::complex<double>(3.0, 4.0),
                                                   std::complex<double>(-5.0, 6.0),
                                                   std::complex<double>(7.5, -8.5)};

    compensated_blas::runtime::deferred_rounding_vector_t desc{};
    desc.data = x.data();
    desc.length = x.size();
    desc.stride = 1;
    desc.element_size = sizeof(compensated_blas_complex_double);
    desc.alignment = alignof(compensated_blas_complex_double);
    desc.type = compensated_blas::runtime::scalar_type_t::complex128;
    compensated_blas::runtime::register_deferred_rounding_vector(desc);

    desc.data = y.data();
    compensated_blas::runtime::register_deferred_rounding_vector(desc);

    auto reconstruct_value = [](const compensated_blas_complex_double &primary,
                                const compensated_blas_complex_double *bins,
                                std::size_t terms) {
        long double sum_real = primary.real;
        long double sum_imag = primary.imag;
        if (bins && terms > 0) {
            for (std::size_t i = 0; i < terms; ++i) {
                sum_real += bins[i].real;
                sum_imag += bins[i].imag;
            }
        }
        return std::complex<double>(static_cast<double>(sum_real), static_cast<double>(sum_imag));
    };

    auto gather = [&](compensated_blas_complex_double *data,
                      std::size_t length) {
        std::vector<std::complex<double>> result(length);
        auto descriptor = compensated_blas::runtime::find_deferred_rounding_vector(data);
        if (descriptor.has_value() && descriptor->compensation != nullptr && descriptor->compensation_terms > 0) {
            auto *bins = static_cast<compensated_blas_complex_double *>(descriptor->compensation);
            for (std::size_t i = 0; i < length; ++i) {
                const std::size_t offset = i * descriptor->stride;
                const bool within = offset < descriptor->compensation_elements;
                const compensated_blas_complex_double *element_bins = within ? bins + offset * descriptor->compensation_terms : nullptr;
                result[i] = reconstruct_value(data[offset], element_bins, descriptor->compensation_terms);
            }
        } else {
            for (std::size_t i = 0; i < length; ++i) {
                result[i] = std::complex<double>(data[i].real, data[i].imag);
            }
        }
        return result;
    };

    auto &backend = compensated_blas::impl::get_active_ilp64_backend();
    std::int64_t n = static_cast<std::int64_t>(x.size());
    std::int64_t inc = 1;

    backend.zswap(&n, x.data(), &inc, y.data(), &inc);
    for (std::size_t i = 0; i < expected_x.size(); ++i) {
        std::swap(expected_x[i], expected_y[i]);
    }
    auto actual_x = gather(x.data(), x.size());
    auto actual_y = gather(y.data(), y.size());
    for (std::size_t i = 0; i < expected_x.size(); ++i) {
        EXPECT_NEAR(actual_x[i].real(), expected_x[i].real(), 1e-6);
        EXPECT_NEAR(actual_x[i].imag(), expected_x[i].imag(), 1e-6);
        EXPECT_NEAR(actual_y[i].real(), expected_y[i].real(), 1e-6);
        EXPECT_NEAR(actual_y[i].imag(), expected_y[i].imag(), 1e-6);
    }

    double real_scale = 0.5;
    backend.zdscal(&n, &real_scale, x.data(), &inc);
    for (auto &value : expected_x) {
        value *= real_scale;
    }

    compensated_blas_complex_double complex_scale{0.25, -0.5};
    backend.zscal(&n, &complex_scale, y.data(), &inc);
    const std::complex<double> complex_scale_std(0.25, -0.5);
    for (auto &value : expected_y) {
        value *= complex_scale_std;
    }

    backend.zcopy(&n, y.data(), &inc, x.data(), &inc);
    expected_x = expected_y;

    compensated_blas_complex_double zaxpy_alpha{-1.0, 0.5};
    backend.zaxpy(&n, &zaxpy_alpha, y.data(), &inc, x.data(), &inc);
    const std::complex<double> zaxpy_alpha_std(-1.0, 0.5);
    for (std::size_t i = 0; i < expected_x.size(); ++i) {
        expected_x[i] += zaxpy_alpha_std * expected_y[i];
    }

    double zdrot_c = 0.8;
    double zdrot_s = 0.6;
    backend.zdrot(&n, x.data(), &inc, y.data(), &inc, &zdrot_c, &zdrot_s);
    for (std::size_t i = 0; i < expected_x.size(); ++i) {
        const std::complex<double> old_x = expected_x[i];
        const std::complex<double> old_y = expected_y[i];
        expected_x[i] = zdrot_c * old_x + zdrot_s * old_y;
        expected_y[i] = zdrot_c * old_y - zdrot_s * old_x;
    }

    actual_x = gather(x.data(), x.size());
    actual_y = gather(y.data(), y.size());
    for (std::size_t i = 0; i < expected_x.size(); ++i) {
        EXPECT_NEAR(actual_x[i].real(), expected_x[i].real(), 1e-5);
        EXPECT_NEAR(actual_x[i].imag(), expected_x[i].imag(), 1e-5);
        EXPECT_NEAR(actual_y[i].real(), expected_y[i].real(), 1e-5);
        EXPECT_NEAR(actual_y[i].imag(), expected_y[i].imag(), 1e-5);
    }

    compensated_blas_complex_double zrotg_a{-2.0, 1.5};
    compensated_blas_complex_double zrotg_b{4.0, -3.0};
    double zrotg_c = 0.0;
    compensated_blas_complex_double zrotg_s{};
    backend.zrotg(&zrotg_a, &zrotg_b, &zrotg_c, &zrotg_s);

    std::complex<double> za(-2.0, 1.5);
    std::complex<double> zb(4.0, -3.0);
    double expected_zc;
    std::complex<double> expected_zs;
    std::complex<double> expected_zr;
    const double abs_za = std::abs(za);
    const double abs_zb = std::abs(zb);
    if (abs_za == 0.0 && abs_zb == 0.0) {
        expected_zc = 1.0;
        expected_zs = std::complex<double>(0.0, 0.0);
        expected_zr = std::complex<double>(0.0, 0.0);
    } else if (abs_za == 0.0) {
        expected_zc = 0.0;
        expected_zs = std::complex<double>(1.0, 0.0);
        expected_zr = zb;
    } else {
        const double scale = abs_za + abs_zb;
        const double norm = scale * std::sqrt((abs_za / scale) * (abs_za / scale) + (abs_zb / scale) * (abs_zb / scale));
        const std::complex<double> alpha = za / abs_za;
        expected_zc = abs_za / norm;
        expected_zs = alpha * std::conj(zb) / norm;
        expected_zr = alpha * norm;
    }
    EXPECT_NEAR(zrotg_c, expected_zc, 1e-12);
    EXPECT_NEAR(zrotg_s.real, expected_zs.real(), 1e-12);
    EXPECT_NEAR(zrotg_s.imag, expected_zs.imag(), 1e-12);
    EXPECT_NEAR(zrotg_a.real, expected_zr.real(), 1e-12);
    EXPECT_NEAR(zrotg_a.imag, expected_zr.imag(), 1e-12);

    auto x_descriptor = compensated_blas::runtime::find_deferred_rounding_vector(x.data());
    ASSERT_TRUE(x_descriptor.has_value());
    if (x_descriptor->compensation != nullptr) {
        auto *bins = static_cast<compensated_blas_complex_double *>(x_descriptor->compensation);
        for (std::size_t i = 0; i < x_descriptor->compensation_elements * x_descriptor->compensation_terms; ++i) {
            EXPECT_TRUE(std::abs(bins[i].real) < 1e-9);
            EXPECT_TRUE(std::abs(bins[i].imag) < 1e-9);
        }
    }

    compensated_blas::runtime::clear_deferred_rounding_registrations();
    compensated_blas::runtime::set_backend(compensated_blas::runtime::backend_kind_t::empty);
}

TEST(compensated_naive_backend, level1_reductions) {
    compensated_blas::runtime::set_backend(compensated_blas::runtime::backend_kind_t::naive);
    compensated_blas::runtime::set_compensation_terms(2);
    compensated_blas::runtime::clear_deferred_rounding_registrations();

    std::array<float, 4> xf{3.0f, -4.0f, 1.0f, -2.0f};
    std::int64_t n = static_cast<std::int64_t>(xf.size());
    std::int64_t inc = 1;

    auto &backend = compensated_blas::impl::get_active_ilp64_backend();
    float norm = backend.snrm2(&n, xf.data(), &inc);
    EXPECT_NEAR(norm, std::sqrt(3.0f * 3.0f + (-4.0f) * (-4.0f) + 1.0f * 1.0f + (-2.0f) * (-2.0f)), 1e-6f);

    float asum = backend.sasum(&n, xf.data(), &inc);
    EXPECT_FLOAT_EQ(asum, 10.0f);

    std::int64_t idx = backend.isamax(&n, xf.data(), &inc);
    EXPECT_EQ(idx, 2);

    std::array<double, 5> xd{1.0, -2.0, 3.0, -4.0, 5.0};
    n = static_cast<std::int64_t>(xd.size());
    inc = 1;
    double dnorm = backend.dnrm2(&n, xd.data(), &inc);
    EXPECT_NEAR(dnorm, std::sqrt(55.0), 1e-12);

    double das = backend.dasum(&n, xd.data(), &inc);
    EXPECT_DOUBLE_EQ(das, 15.0);

    idx = backend.idamax(&n, xd.data(), &inc);
    EXPECT_EQ(idx, 5);

    compensated_blas::runtime::clear_deferred_rounding_registrations();
    compensated_blas::runtime::set_backend(compensated_blas::runtime::backend_kind_t::empty);
}

namespace {

template <typename T>
struct reference_rotmg_constants;

template <>
struct reference_rotmg_constants<float> {
    static constexpr float gam = 4096.0f;
    static constexpr float gamsq = gam * gam;
    static constexpr float rgamsq = 5.96046e-8f;
};

template <>
struct reference_rotmg_constants<double> {
    static constexpr double gam = 4096.0;
    static constexpr double gamsq = gam * gam;
    static constexpr double rgamsq = 5.9604645e-8;
};

template <typename T>
void reference_rotmg(T *d1, T *d2, T *x1, const T *y1, T *param) {
    if (!d1 || !d2 || !x1 || !y1 || !param) {
        return;
    }

    const T zero = T(0);
    const T one = T(1);
    const T neg_one = T(-1);
    const T neg_two = T(-2);
    const T gam = reference_rotmg_constants<T>::gam;
    const T gamsq = reference_rotmg_constants<T>::gamsq;
    const T rgamsq = reference_rotmg_constants<T>::rgamsq;

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
            }
        } else {
            if (q2 < zero) {
                flag = neg_one;
                *d1 = zero;
                *d2 = zero;
                *x1 = zero;
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

template <typename T>
void reference_rotm(std::int64_t n,
                    T *x,
                    std::int64_t incx,
                    T *y,
                    std::int64_t incy,
                    const T *param) {
    if (n <= 0 || incx == 0 || incy == 0 || !x || !y || !param) {
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

    T *x_ptr = x;
    T *y_ptr = y;
    const std::ptrdiff_t sx = static_cast<std::ptrdiff_t>(incx);
    const std::ptrdiff_t sy = static_cast<std::ptrdiff_t>(incy);

    if (sx < 0) {
        x_ptr += static_cast<std::ptrdiff_t>(n - 1) * (-sx);
    }
    if (sy < 0) {
        y_ptr += static_cast<std::ptrdiff_t>(n - 1) * (-sy);
    }

    for (std::int64_t i = 0; i < n; ++i) {
        const T x_val = *x_ptr;
        const T y_val = *y_ptr;

        T new_x;
        T new_y;

        if (flag == static_cast<T>(-1)) {
            new_x = h11 * x_val + h12 * y_val;
            new_y = h21 * x_val + h22 * y_val;
        } else if (flag == static_cast<T>(0)) {
            new_x = x_val + h12 * y_val;
            new_y = h21 * x_val + y_val;
        } else {
            new_x = h11 * x_val + y_val;
            new_y = -x_val + h22 * y_val;
        }

        *x_ptr = new_x;
        *y_ptr = new_y;

        x_ptr += sx;
        y_ptr += sy;
    }
}

}  // namespace

TEST(compensated_naive_backend, level1_real_rotations) {
    compensated_blas::runtime::set_backend(compensated_blas::runtime::backend_kind_t::naive);
    compensated_blas::runtime::set_compensation_terms(2);
    compensated_blas::runtime::clear_deferred_rounding_registrations();

    auto &backend = compensated_blas::impl::get_active_ilp64_backend();

    float a_f = 3.0f;
    float b_f = 4.0f;
    float c_f = 0.0f;
    float s_f = 0.0f;
    backend.srotg(&a_f, &b_f, &c_f, &s_f);
    EXPECT_NEAR(a_f, 5.0f, 1e-6f);
    EXPECT_FLOAT_EQ(c_f * 3.0f + s_f * 4.0f, a_f);
    EXPECT_NEAR(c_f * 4.0f - s_f * 3.0f, 0.0f, 1e-6f);

    std::array<double, 2> xd{1.0, 2.0};
    std::array<double, 2> yd{3.0, -4.0};
    std::int64_t n = 2;
    std::int64_t inc = 1;
    double c = 0.6;
    double s = -0.8;
    backend.drot(&n, xd.data(), &inc, yd.data(), &inc, &c, &s);
    EXPECT_NEAR(xd[0], c * 1.0 + s * 3.0, 1e-12);
    EXPECT_NEAR(yd[0], c * 3.0 - s * 1.0, 1e-12);

    compensated_blas::runtime::deferred_rounding_vector_t desc{};
    std::array<double, 3> x_vec{1.0, 1e16, -1e16};
    std::array<double, 3> y_vec{-1.0, 2.0, 3.0};
    n = 3;
    desc.data = x_vec.data();
    desc.length = x_vec.size();
    desc.stride = 1;
    desc.element_size = sizeof(double);
    desc.alignment = alignof(double);
    desc.type = compensated_blas::runtime::scalar_type_t::real64;
    compensated_blas::runtime::register_deferred_rounding_vector(desc);
    desc.data = y_vec.data();
    compensated_blas::runtime::register_deferred_rounding_vector(desc);

    backend.drot(&n, x_vec.data(), &inc, y_vec.data(), &inc, &c, &s);
    auto descriptor = compensated_blas::runtime::find_deferred_rounding_vector(x_vec.data());
    ASSERT_TRUE(descriptor.has_value());
    ASSERT_NE(descriptor->compensation, nullptr);

    compensated_blas::runtime::clear_deferred_rounding_registrations();
    compensated_blas::runtime::set_backend(compensated_blas::runtime::backend_kind_t::empty);
}

TEST(compensated_naive_backend, level1_rotmg_and_rotm) {
    compensated_blas::runtime::set_backend(compensated_blas::runtime::backend_kind_t::naive);
    compensated_blas::runtime::set_compensation_terms(2);
    compensated_blas::runtime::clear_deferred_rounding_registrations();

    auto &backend = compensated_blas::impl::get_active_ilp64_backend();

    double d1 = 2.0;
    double d2 = 3.0;
    double x1 = 1.5;
    double y1 = -2.0;
    double param[5]{};

    backend.drotmg(&d1, &d2, &x1, &y1, param);

    double ref_param[5]{};
    double ref_d1 = 2.0;
    double ref_d2 = 3.0;
    double ref_x1 = 1.5;
    reference_rotmg(&ref_d1, &ref_d2, &ref_x1, &y1, ref_param);

    for (int i = 0; i < 5; ++i) {
        EXPECT_NEAR(param[i], ref_param[i], 1e-12);
    }
    EXPECT_NEAR(d1, ref_d1, 1e-12);
    EXPECT_NEAR(d2, ref_d2, 1e-12);
    EXPECT_NEAR(x1, ref_x1, 1e-12);

    std::array<double, 3> xv{1.0, -2.0, 3.0};
    std::array<double, 3> yv{4.0, 5.0, -6.0};
    std::int64_t n = 3;
    std::int64_t inc = 1;

    backend.drotm(&n, xv.data(), &inc, yv.data(), &inc, param);

    std::array<double, 3> ref_x = {1.0, -2.0, 3.0};
    std::array<double, 3> ref_y = {4.0, 5.0, -6.0};
    reference_rotm(n, ref_x.data(), inc, ref_y.data(), inc, ref_param);

    for (std::size_t i = 0; i < xv.size(); ++i) {
        EXPECT_NEAR(xv[i], ref_x[i], 1e-12);
        EXPECT_NEAR(yv[i], ref_y[i], 1e-12);
    }

    compensated_blas::runtime::set_backend(compensated_blas::runtime::backend_kind_t::empty);
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
namespace {

template <typename T>
T collapse_bins(T primary, const T *bins, std::size_t terms) {
    long double acc = static_cast<long double>(primary);
    for (std::size_t i = 0; i < terms; ++i) {
        acc += static_cast<long double>(bins[i]);
    }
    return static_cast<T>(acc);
}

template <typename T>
std::array<long double, 3> forward_substitution_lower(const std::array<T, 9> &a,
                                                      const std::array<T, 3> &b) {
    const std::size_t lda = 3;
    std::array<long double, 3> x{};
    for (std::size_t i = 0; i < 3; ++i) {
        long double sum = 0.0L;
        for (std::size_t j = 0; j < i; ++j) {
            const std::size_t index = j * lda + i;
            sum += static_cast<long double>(a[index]) * x[j];
        }
        const std::size_t diag_index = i * lda + i;
        const long double diag = static_cast<long double>(a[diag_index]);
        x[i] = (static_cast<long double>(b[i]) - sum) / diag;
    }
    return x;
}

}  // namespace

TEST(CompensatedSymv, DeferredStoresAccurately) {
    const std::int64_t n = 3;
    const float alpha = 2.0f;
    const float beta = 0.5f;
    const float a[9] = {
        1.0f, 0.0f, 2.0f,
        0.0f, 3.0f, -4.0f,
        2.0f, -4.0f, 5.0f,
    };

    float x_host[n] = {1.0f, -2.0f, 0.5f};
    float y_host[n] = {0.25f, -1.0f, 2.0f};

    // Register deferred rounding buffers for y.
    std::vector<float> compensation(3 * 2, 0.0f);
    const std::size_t previous_terms = compensated_blas::runtime::compensation_terms();
    compensated_blas::runtime::set_compensation_terms(2);
    compensated_blas::runtime::deferred_rounding_vector_t descriptor{};
    descriptor.data = y_host;
    descriptor.length = static_cast<std::size_t>(n);
    descriptor.stride = 1;
    descriptor.element_size = sizeof(float);
    descriptor.alignment = alignof(float);
    descriptor.type = compensated_blas::runtime::scalar_type_t::real32;
    descriptor.compensation = compensation.data();
    descriptor.compensation_elements = static_cast<std::size_t>(n);
    descriptor.compensation_terms = 2;
    compensated_blas::runtime::register_deferred_rounding_vector(descriptor);

    std::int64_t lda = n;
    std::int64_t stride = 1;
    auto backend = compensated_blas::impl::detail::acquire_naive_backend();
    backend->ssymv("U", &n, &alpha, a, &lda, x_host, &stride, &beta, y_host, &stride);

    // Collapse compensated result and compare against reference computed in long double.
    const long double ref0 = static_cast<long double>(alpha) * (1.0L * 1.0L + 0.0L * -2.0L + 2.0L * 0.5L) + static_cast<long double>(beta) * 0.25L;
    const long double ref1 = static_cast<long double>(alpha) * (3.0L * -2.0L + -4.0L * 0.5L) + static_cast<long double>(beta) * -1.0L;
    const long double ref2 = static_cast<long double>(alpha) * (5.0L * 0.5L + -4.0L * -2.0L + 2.0L * 1.0L) + static_cast<long double>(beta) * 2.0L;

    float y0 = collapse_bins(y_host[0], compensation.data() + 0 * 2, 2);
    float y1 = collapse_bins(y_host[1], compensation.data() + 1 * 2, 2);
    float y2 = collapse_bins(y_host[2], compensation.data() + 2 * 2, 2);

    EXPECT_NEAR(static_cast<long double>(y0), ref0, 1e-6L);
    EXPECT_NEAR(static_cast<long double>(y1), ref1, 1e-6L);
    EXPECT_NEAR(static_cast<long double>(y2), ref2, 1e-6L);

    compensated_blas::runtime::clear_deferred_rounding_registrations();
    compensated_blas::runtime::set_compensation_terms(previous_terms);
}

TEST(CompensatedTrsv, LowerNoTransposeDeferred) {
    const std::int64_t n = 3;
    const std::array<float, 9> a = {
        3.0f, -1.0f, 4.0f,
        0.0f, 2.5f, -3.0f,
        0.0f, 0.0f, -1.5f,
    };
    std::array<float, 3> x = {5.0f, -2.0f, 1.0f};
    const std::array<float, 3> rhs = x;

    std::vector<float> compensation(3 * 2, 0.0f);
    const std::size_t previous_terms = compensated_blas::runtime::compensation_terms();
    compensated_blas::runtime::set_compensation_terms(2);

    compensated_blas::runtime::deferred_rounding_vector_t descriptor{};
    descriptor.data = x.data();
    descriptor.length = static_cast<std::size_t>(n);
    descriptor.stride = 1;
    descriptor.element_size = sizeof(float);
    descriptor.alignment = alignof(float);
    descriptor.type = compensated_blas::runtime::scalar_type_t::real32;
    descriptor.compensation = compensation.data();
    descriptor.compensation_elements = static_cast<std::size_t>(n);
    descriptor.compensation_terms = 2;
    compensated_blas::runtime::register_deferred_rounding_vector(descriptor);

    std::int64_t lda = n;
    std::int64_t inc = 1;
    auto backend = compensated_blas::impl::detail::acquire_naive_backend();
    backend->strsv("L", "N", "N", &n, a.data(), &lda, x.data(), &inc);

    const auto reference = forward_substitution_lower(a, rhs);

    for (std::size_t i = 0; i < 3; ++i) {
        float collapsed = collapse_bins(x[i], compensation.data() + i * 2, 2);
        EXPECT_NEAR(static_cast<long double>(collapsed), reference[i], 1e-5L);
    }

    compensated_blas::runtime::clear_deferred_rounding_registrations();
    compensated_blas::runtime::set_compensation_terms(previous_terms);
}

TEST(CompensatedTrmv, UpperNoTransposeDeferred) {
    const std::int64_t n = 3;
    const std::array<double, 9> a = {
        2.0, 0.0, 0.0,
        -1.0, -1.5, 0.0,
        3.5, 4.0, 0.75,
    };
    std::array<double, 3> x = {1.0, -2.0, 0.5};
    const std::array<double, 3> x_initial = x;

    std::vector<double> bins(3 * 2, 0.0);
    const std::size_t previous_terms = compensated_blas::runtime::compensation_terms();
    compensated_blas::runtime::set_compensation_terms(2);

    compensated_blas::runtime::deferred_rounding_vector_t descriptor{};
    descriptor.data = x.data();
    descriptor.length = static_cast<std::size_t>(n);
    descriptor.stride = 1;
    descriptor.element_size = sizeof(double);
    descriptor.alignment = alignof(double);
    descriptor.type = compensated_blas::runtime::scalar_type_t::real64;
    descriptor.compensation = bins.data();
    descriptor.compensation_elements = static_cast<std::size_t>(n);
    descriptor.compensation_terms = 2;
    compensated_blas::runtime::register_deferred_rounding_vector(descriptor);

    std::int64_t lda = n;
    std::int64_t inc = 1;
    auto backend = compensated_blas::impl::detail::acquire_naive_backend();
    backend->dtrmv("U", "N", "N", &n, a.data(), &lda, x.data(), &inc);

    std::array<long double, 3> reference{};
    reference[0] = static_cast<long double>(a[0]) * x_initial[0] +
                  static_cast<long double>(a[3]) * x_initial[1] +
                  static_cast<long double>(a[6]) * x_initial[2];
    reference[1] = static_cast<long double>(a[4]) * x_initial[1] +
                  static_cast<long double>(a[7]) * x_initial[2];
    reference[2] = static_cast<long double>(a[8]) * x_initial[2];

    for (std::size_t i = 0; i < 3; ++i) {
        double collapsed = collapse_bins(x[i], bins.data() + i * 2, 2);
        EXPECT_NEAR(static_cast<long double>(collapsed), reference[i], 1e-12L);
    }

    compensated_blas::runtime::clear_deferred_rounding_registrations();
    compensated_blas::runtime::set_compensation_terms(previous_terms);
}
