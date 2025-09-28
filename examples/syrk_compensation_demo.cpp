#include "compensated_blas.hpp"
#include "impl/compensated_arithmetic.hpp"
#include "impl/compensated_blas_backend_ilp64.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

namespace {

struct demo_config_t {
    std::size_t size = 64;
    std::size_t rank = 48;
    std::size_t updates = 50;
    std::size_t compensation_terms = 4;
};

std::map<std::string, std::string> parse_args(int argc, char **argv) {
    std::map<std::string, std::string> values;
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        auto pos = arg.find('=');
        if (pos == std::string::npos) {
            values[arg] = "";
        } else {
            values[arg.substr(0, pos)] = arg.substr(pos + 1);
        }
    }
    return values;
}

void print_help(std::ostream &out) {
    out << "SYRK compensation demo\n"
        << "Options:\n"
        << "  --help                    Show this message\n"
        << "  --size=<n>               Matrix dimension (default 64)\n"
        << "  --rank=<n>               Rank parameter k (default 48)\n"
        << "  --updates=<n>            Number of repeated SYRK updates (default 50)\n"
        << "  --compensation-terms=<n> Compensation bins when enabled (default 4)\n";
}

demo_config_t build_config(const std::map<std::string, std::string> &args) {
    demo_config_t config;
    auto fetch = [&](const std::string &key) -> std::optional<std::string> {
        auto it = args.find(key);
        if (it == args.end()) {
            return std::nullopt;
        }
        return it->second;
    };

    if (auto value = fetch("--size")) {
        config.size = static_cast<std::size_t>(std::stoul(*value));
    }
    if (auto value = fetch("--rank")) {
        config.rank = static_cast<std::size_t>(std::stoul(*value));
    }
    if (auto value = fetch("--updates")) {
        config.updates = static_cast<std::size_t>(std::stoul(*value));
    }
    if (auto value = fetch("--compensation-terms")) {
        config.compensation_terms = static_cast<std::size_t>(std::stoul(*value));
    }
    return config;
}

void fill_test_matrix(std::vector<double> &a, std::size_t rows, std::size_t cols, std::size_t lda) {
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (std::size_t j = 0; j < cols; ++j) {
        for (std::size_t i = 0; i < rows; ++i) {
            a[j * lda + i] = dist(rng);
        }
    }
}

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

void flush_matrix_if_deferred(double *data, std::size_t size, std::size_t leading_dimension) {
    using namespace compensated_blas::runtime;
    auto descriptor = find_deferred_rounding_matrix(static_cast<const void *>(data));
    if (!descriptor.has_value() || descriptor->compensation_terms == 0 || descriptor->compensation == nullptr) {
        return;
    }
    auto *bins = static_cast<double *>(descriptor->compensation);
    for (std::size_t col = 0; col < descriptor->cols; ++col) {
        for (std::size_t row = 0; row < descriptor->rows; ++row) {
            const std::size_t index = descriptor->row_major ? row * descriptor->leading_dimension + col
                                                            : col * descriptor->leading_dimension + row;
            double &primary = data[index];
            double *element_bins = bins + index * descriptor->compensation_terms;
            finalize_with_bins(primary, element_bins, descriptor->compensation_terms);
        }
    }
}

void symmetrize_lower_to_full(std::vector<double> &c, std::size_t n, std::size_t ldc) {
    for (std::size_t col = 0; col < n; ++col) {
        for (std::size_t row = col + 1; row < n; ++row) {
            const double value = c[row + col * ldc];
            c[col + row * ldc] = value;
        }
    }
}

double cholesky_min_pivot(std::vector<double> c, std::size_t n, std::size_t ldc) {
    double min_pivot = std::numeric_limits<double>::infinity();
    for (std::size_t j = 0; j < n; ++j) {
        double diag = c[j * ldc + j];
        for (std::size_t k = 0; k < j; ++k) {
            const double l_jk = c[j * ldc + k];
            diag -= l_jk * l_jk;
        }
        if (diag <= 0.0) {
            return -std::sqrt(std::abs(diag));
        }
        diag = std::sqrt(diag);
        min_pivot = std::min(min_pivot, diag);
        c[j * ldc + j] = diag;
        for (std::size_t i = j + 1; i < n; ++i) {
            double value = c[i * ldc + j];
            for (std::size_t k = 0; k < j; ++k) {
                value -= c[i * ldc + k] * c[j * ldc + k];
            }
            value /= diag;
            c[i * ldc + j] = value;
        }
    }
    return min_pivot;
}

struct syrk_result_t {
    double min_pivot = 0.0;
};

syrk_result_t run_syrk_demo(std::vector<double> c,
                             const std::vector<double> &a,
                             std::size_t n,
                             std::size_t k,
                             std::size_t ldc,
                             std::size_t lda,
                             std::size_t updates,
                             bool enable_compensation,
                             std::size_t compensation_terms) {
    using namespace compensated_blas::runtime;

    set_backend(backend_kind_t::naive);
    set_compensation_terms(enable_compensation ? compensation_terms : 0);
    clear_deferred_rounding_registrations();

    if (enable_compensation && compensation_terms > 0) {
        deferred_rounding_matrix_t descriptor{};
        descriptor.data = c.data();
        descriptor.rows = n;
        descriptor.cols = n;
        descriptor.leading_dimension = ldc;
        descriptor.element_size = sizeof(double);
        descriptor.alignment = alignof(double);
        descriptor.type = scalar_type_t::real64;
        descriptor.row_major = false;
        register_deferred_rounding_matrix(descriptor);
    }

    auto &backend = compensated_blas::impl::get_active_ilp64_backend();
    const std::int64_t n64 = static_cast<std::int64_t>(n);
    const std::int64_t k64 = static_cast<std::int64_t>(k);
    const std::int64_t lda64 = static_cast<std::int64_t>(lda);
    const std::int64_t ldc64 = static_cast<std::int64_t>(ldc);
    const double alpha = 1.0;
    const double beta = 1.0;
    const char uplo = 'L';
    const char trans = 'N';

    for (std::size_t step = 0; step < updates; ++step) {
        backend.dsyrk(&uplo,
                      &trans,
                      &n64,
                      &k64,
                      &alpha,
                      a.data(),
                      &lda64,
                      &beta,
                      c.data(),
                      &ldc64);
    }

    if (enable_compensation && compensation_terms > 0) {
        flush_matrix_if_deferred(c.data(), n, ldc);
    }

    symmetrize_lower_to_full(c, n, ldc);
    const double min_pivot = cholesky_min_pivot(c, n, ldc);

    clear_deferred_rounding_registrations();
    set_backend(backend_kind_t::empty);

    return {min_pivot};
}

}  // namespace

int main(int argc, char **argv) {
    const auto args = parse_args(argc, argv);
    if (args.count("--help")) {
        print_help(std::cout);
        return 0;
    }
    const auto config = build_config(args);

    std::cout << "SYRK compensation demo with size=" << config.size
              << " rank=" << config.rank
              << " updates=" << config.updates
              << '\n';

    const std::size_t n = config.size;
    const std::size_t k = config.rank;
    const std::size_t lda = n;
    const std::size_t ldc = n;

    std::vector<double> a(k * lda);
    std::vector<double> c(n * ldc, 0.0);
    std::vector<double> c_initial = c;

    fill_test_matrix(a, n, k, lda);

    auto plain = run_syrk_demo(c_initial,
                                a,
                                n,
                                k,
                                ldc,
                                lda,
                                config.updates,
                                false,
                                0);

    auto compensated = run_syrk_demo(c_initial,
                                      a,
                                      n,
                                      k,
                                      ldc,
                                      lda,
                                      config.updates,
                                      true,
                                      config.compensation_terms);

    std::cout << std::fixed << std::setprecision(10);
    std::cout << "Plain min Cholesky pivot:        " << plain.min_pivot << '\n';
    std::cout << "Rolling-compensated min pivot:  " << compensated.min_pivot << '\n';

    return 0;
}
