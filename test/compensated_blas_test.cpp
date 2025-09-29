#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <complex>
#include <cblas.h>
#include <type_traits>

#include <gtest/gtest.h>

#include "impl/compensated_arithmetic.hpp"
#include "compensated_blas.hpp"
#include "impl/compensated_accumulator.hpp"
#include "impl/compensated_blas_backend_ilp64.hpp"

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
    cblas_drotmg(&ref_d1, &ref_d2, &ref_x1, y1, ref_param);

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
    cblas_drotm(n, ref_x.data(), inc, ref_y.data(), inc, ref_param);

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
