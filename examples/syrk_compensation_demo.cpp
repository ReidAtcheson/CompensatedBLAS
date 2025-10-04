#include "compensated_blas.hpp"
#include "impl/compensated_arithmetic.hpp"
#include "impl/compensated_blas_backend_ilp64.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace {

struct demo_config_t {
    std::size_t size = 64;
    std::size_t rank = 48;
    std::size_t updates = 50;
    std::size_t compensation_terms = 4;
    std::optional<std::string> dump_prefix;
    std::uint64_t random_seed = 1337;
    std::size_t dependent_basis = 8;
    double initial_noise_scale = 1e-2;
    double noise_decay = 0.5;
};

std::optional<std::string> build_dump_path(const std::optional<std::string> &prefix,
                                           const std::string &tag) {
    if (!prefix) {
        return std::nullopt;
    }
    std::string path = *prefix;
    path += '_';
    path += tag;
    path += ".txt";
    return path;
}

bool dump_matrix_to_file(const std::string &path,
                         const std::vector<double> &matrix,
                         std::size_t n,
                         std::size_t ldc,
                         std::string &error_message) {
    std::ofstream out(path, std::ios::out | std::ios::trunc);
    if (!out) {
        std::ostringstream oss;
        oss << "unable to open '" << path << "' for writing";
        error_message = oss.str();
        return false;
    }
    out << n << " " << n << '\n';
    out << std::setprecision(17);
    for (std::size_t row = 0; row < n; ++row) {
        for (std::size_t col = 0; col < n; ++col) {
            const double value = matrix[col * ldc + row];
            out << value;
            if (col + 1 < n) {
                out << ' ';
            }
        }
        out << '\n';
    }
    if (!out) {
        error_message = "failed while writing matrix data";
        return false;
    }
    return true;
}

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
        << "  --compensation-terms=<n> Compensation bins when enabled (default 4)\n"
        << "  --dump-prefix=<path>     Write matrices to <path>_{plain,compensated}.txt\n"
        << "  --seed=<n>               Seed for random updates (default 1337)\n"
        << "  --basis=<n>              Count of cached dependent basis columns (default 8)\n"
        << "  --noise-scale=<x>        Initial noise scale mixed into basis (default 1e-2)\n"
        << "  --noise-decay=<x>        Decay factor applied to noise per update (default 0.5)\n";
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
    if (auto value = fetch("--dump-prefix")) {
        if (!value->empty()) {
            config.dump_prefix = *value;
        }
    }
    if (auto value = fetch("--seed")) {
        if (!value->empty()) {
            config.random_seed = static_cast<std::uint64_t>(std::stoull(*value));
        }
    }
    if (auto value = fetch("--basis")) {
        if (!value->empty()) {
            config.dependent_basis = std::max<std::size_t>(1, std::stoul(*value));
        }
    }
    if (auto value = fetch("--noise-scale")) {
        if (!value->empty()) {
            config.initial_noise_scale = std::stod(*value);
        }
    }
    if (auto value = fetch("--noise-decay")) {
        if (!value->empty()) {
            config.noise_decay = std::stod(*value);
        }
    }
    return config;
}

void fill_random_matrix(std::vector<double> &a,
                        std::size_t rows,
                        std::size_t cols,
                        std::size_t lda,
                        std::mt19937_64 &rng,
                        std::uniform_real_distribution<double> &dist) {
    for (std::size_t col = 0; col < cols; ++col) {
        for (std::size_t row = 0; row < rows; ++row) {
            a[col * lda + row] = dist(rng);
        }
    }
}

void naive_dsyrk_lower(std::size_t n,
                       std::size_t k,
                       const double *a,
                       std::size_t lda,
                       double *c,
                       std::size_t ldc) {
    for (std::size_t col = 0; col < n; ++col) {
        for (std::size_t row = col; row < n; ++row) {
            double sum = 0.0;
            for (std::size_t inner = 0; inner < k; ++inner) {
                const double a_row = a[inner * lda + row];
                const double a_col = a[inner * lda + col];
                sum += a_row * a_col;
            }
            c[col * ldc + row] += sum;
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
    std::optional<std::string> dump_path;
    std::optional<std::string> dump_error;
};

syrk_result_t run_syrk_demo(std::vector<double> c,
                             std::vector<double> &a,
                             std::size_t n,
                             std::size_t k,
                             std::size_t ldc,
                             std::size_t lda,
                             std::size_t updates,
                             bool enable_compensation,
                             std::size_t compensation_terms,
                             const std::optional<std::string> &dump_path,
                             std::uint64_t rng_seed,
                             std::size_t dependent_basis_count,
                             double initial_noise_scale,
                             double noise_decay) {
    using namespace compensated_blas::runtime;

    syrk_result_t result;

    clear_deferred_rounding_registrations();

    const bool use_compensated_backend = enable_compensation && compensation_terms > 0;

    if (use_compensated_backend) {
        set_backend(backend_kind_t::naive);
        set_compensation_terms(compensation_terms);
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
    } else {
        set_backend(backend_kind_t::empty);
        set_compensation_terms(0);
    }

    std::mt19937_64 rng(rng_seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    const std::size_t basis_columns = std::max<std::size_t>(1, std::min(dependent_basis_count, k));
    std::vector<double> basis(lda * basis_columns);
    fill_random_matrix(basis, n, basis_columns, lda, rng, dist);

    double noise_scale = std::abs(initial_noise_scale);
    const double decay = std::abs(noise_decay);
    auto column_coeff = [&](std::mt19937_64 &engine) {
        return 1.0 + 0.05 * dist(engine);
    };

    auto mix_column = [&](std::size_t column_index, double factor, double noise_strength) {
        const std::size_t basis_index = column_index % basis_columns;
        const double *basis_col = &basis[basis_index * lda];
        double *target_col = &a[column_index * lda];
        for (std::size_t row = 0; row < n; ++row) {
            double value = factor * basis_col[row];
            if (noise_strength > 0.0) {
                value += noise_strength * dist(rng);
            }
            target_col[row] = value;
        }
    };

    const std::int64_t n64 = static_cast<std::int64_t>(n);
    const std::int64_t k64 = static_cast<std::int64_t>(k);
    const std::int64_t lda64 = static_cast<std::int64_t>(lda);
    const std::int64_t ldc64 = static_cast<std::int64_t>(ldc);

    const double alpha = 1.0;
    const double beta = 1.0;
    const char uplo = 'L';
    const char trans = 'N';

    auto *backend = use_compensated_backend ? &compensated_blas::impl::get_active_ilp64_backend() : nullptr;

    const double min_noise_scale = std::abs(initial_noise_scale) * 1e-6;

    for (std::size_t step = 0; step < updates; ++step) {
        for (std::size_t col = 0; col < k; ++col) {
            const double coeff = column_coeff(rng);
            mix_column(col, coeff, noise_scale);
        }

        if (use_compensated_backend && backend) {
            backend->dsyrk(&uplo,
                           &trans,
                           &n64,
                           &k64,
                           &alpha,
                           a.data(),
                           &lda64,
                           &beta,
                           c.data(),
                           &ldc64);
        } else {
            naive_dsyrk_lower(n, k, a.data(), lda, c.data(), ldc);
        }

        if (noise_scale > 0.0) {
            const double basis_perturb = 0.25 * noise_scale;
            for (std::size_t col = 0; col < basis_columns; ++col) {
                double *basis_col = &basis[col * lda];
                for (std::size_t row = 0; row < n; ++row) {
                    basis_col[row] += basis_perturb * dist(rng);
                }
            }
        }

        noise_scale = decay * noise_scale;
        if (noise_scale < min_noise_scale) {
            noise_scale = 0.0;
        }
    }

    if (use_compensated_backend) {
        flush_matrix_if_deferred(c.data(), n, ldc);
    }

    symmetrize_lower_to_full(c, n, ldc);

    if (dump_path) {
        std::string error_message;
        if (dump_matrix_to_file(*dump_path, c, n, ldc, error_message)) {
            result.dump_path = *dump_path;
        } else {
            result.dump_error = std::move(error_message);
        }
    }

    result.min_pivot = cholesky_min_pivot(std::move(c), n, ldc);

    clear_deferred_rounding_registrations();
    set_backend(backend_kind_t::empty);

    return result;
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

    auto plain_dump_path = build_dump_path(config.dump_prefix, "plain");
    auto compensated_dump_path = build_dump_path(config.dump_prefix, "compensated");

    auto plain = run_syrk_demo(c_initial,
                                a,
                                n,
                            k,
                            ldc,
                            lda,
                            config.updates,
                            false,
                            0,
                            plain_dump_path,
                            config.random_seed,
                            config.dependent_basis,
                            config.initial_noise_scale,
                            config.noise_decay);

    auto compensated = run_syrk_demo(c_initial,
                                      a,
                                      n,
                                      k,
                                      ldc,
                                      lda,
                                      config.updates,
                                      true,
                                      config.compensation_terms,
                                      compensated_dump_path,
                                      config.random_seed,
                                      config.dependent_basis,
                                      config.initial_noise_scale,
                                      config.noise_decay);

    std::cout << std::fixed << std::setprecision(10);
    std::cout << "Plain min Cholesky pivot:        " << plain.min_pivot << '\n';
    std::cout << "Rolling-compensated min pivot:  " << compensated.min_pivot << '\n';
    if (plain.dump_path) {
        std::cout << "Plain matrix written to:        " << *plain.dump_path << '\n';
    } else if (plain.dump_error) {
        std::cout << "Plain matrix dump failed:       " << *plain.dump_error << '\n';
    }
    if (compensated.dump_path) {
        std::cout << "Compensated matrix written to:  " << *compensated.dump_path << '\n';
    } else if (compensated.dump_error) {
        std::cout << "Compensated matrix dump failed: " << *compensated.dump_error << '\n';
    }

    return 0;
}
