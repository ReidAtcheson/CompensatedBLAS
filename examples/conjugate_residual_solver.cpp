#include "compensated_blas.hpp"
#include "impl/compensated_arithmetic.hpp"
#include "impl/compensated_blas_backend_ilp64.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <complex>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

struct solver_config_t {
    std::size_t grid_extent = 64;
    std::size_t max_iterations = 2000;
    double tolerance = 1e-10;
    std::size_t log_interval = 100;
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
    out << "Conjugate Residual solver demo\n"
        << "Options:\n"
        << "  --help                    Show this message\n"
        << "  --grid=<n>               Grid extent per dimension (default 64)\n"
        << "  --max-iter=<n>           Maximum iterations (default 2000)\n"
        << "  --tol=<value>            Convergence tolerance (default 1e-10)\n"
        << "  --log-interval=<n>       Logging interval in iterations (default 100)\n"
        << "  --compensation-terms=<n> Compensation bins when enabled (default 4)\n";
}

solver_config_t build_config(const std::map<std::string, std::string> &args) {
    solver_config_t config;
    auto fetch = [&](const std::string &key) -> std::optional<std::string> {
        auto it = args.find(key);
        if (it == args.end()) {
            return std::nullopt;
        }
        return it->second;
    };

    if (auto value = fetch("--grid")) {
        config.grid_extent = static_cast<std::size_t>(std::stoul(*value));
    }
    if (auto value = fetch("--max-iter")) {
        config.max_iterations = static_cast<std::size_t>(std::stoul(*value));
    }
    if (auto value = fetch("--tol")) {
        config.tolerance = std::stod(*value);
    }
    if (auto value = fetch("--log-interval")) {
        config.log_interval = static_cast<std::size_t>(std::stoul(*value));
    }
    if (auto value = fetch("--compensation-terms")) {
        config.compensation_terms = static_cast<std::size_t>(std::stoul(*value));
    }
    return config;
}

inline std::size_t linear_index(std::size_t grid, std::size_t row, std::size_t col) {
    return row * grid + col;
}

void apply_laplacian(const std::vector<double> &x, std::vector<double> &y, std::size_t grid) {
    const std::size_t total = grid * grid;
    if (y.size() != total) {
        y.resize(total);
    }
    std::fill(y.begin(), y.end(), 0.0);

    for (std::size_t row = 0; row < grid; ++row) {
        for (std::size_t col = 0; col < grid; ++col) {
            const std::size_t idx = linear_index(grid, row, col);
            double value = 4.0 * x[idx];
            if (row > 0) {
                value -= x[linear_index(grid, row - 1, col)];
            }
            if (row + 1 < grid) {
                value -= x[linear_index(grid, row + 1, col)];
            }
            if (col > 0) {
                value -= x[linear_index(grid, row, col - 1)];
            }
            if (col + 1 < grid) {
                value -= x[linear_index(grid, row, col + 1)];
            }
            y[idx] = value;
        }
    }
}

inline double dot_backend(const std::vector<double> &x,
                          const std::vector<double> &y) {
    const std::int64_t size = static_cast<std::int64_t>(x.size());
    const std::int64_t stride = 1;
    auto &backend = compensated_blas::impl::get_active_ilp64_backend();
    return backend.ddot(&size, x.data(), &stride, y.data(), &stride);
}

inline void axpy_backend(double alpha,
                         const std::vector<double> &x,
                         std::vector<double> &y) {
    const std::int64_t size = static_cast<std::int64_t>(x.size());
    const std::int64_t stride = 1;
    auto &backend = compensated_blas::impl::get_active_ilp64_backend();
    backend.daxpy(&size, &alpha, x.data(), &stride, y.data(), &stride);
}

inline void copy_backend(const std::vector<double> &src, std::vector<double> &dst) {
    if (dst.size() != src.size()) {
        dst.resize(src.size());
    }
    std::copy(src.begin(), src.end(), dst.begin());
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

void flush_vector_if_deferred(double *data, std::size_t length) {
    using namespace compensated_blas::runtime;
    auto descriptor = find_deferred_rounding_vector(static_cast<const void *>(data));
    if (!descriptor.has_value() || descriptor->compensation_terms == 0 || descriptor->compensation == nullptr) {
        return;
    }
    auto *bins = static_cast<double *>(descriptor->compensation);
    for (std::size_t i = 0; i < length; ++i) {
        double &primary = data[i];
        double *element_bins = bins + i * descriptor->compensation_terms;
        finalize_with_bins(primary, element_bins, descriptor->compensation_terms);
    }
}

struct solver_log_entry_t {
    std::size_t iteration = 0;
    double residual_norm = 0.0;
};

struct solver_result_t {
    bool converged = false;
    std::size_t iterations = 0;
    double final_residual_norm = 0.0;
    std::vector<solver_log_entry_t> log_entries;
};

void register_vector_for_compensation(double *data, std::size_t length) {
    using namespace compensated_blas::runtime;
    deferred_rounding_vector_t descriptor{};
    descriptor.data = data;
    descriptor.length = length;
    descriptor.stride = 1;
    descriptor.element_size = sizeof(double);
    descriptor.alignment = alignof(double);
    descriptor.type = scalar_type_t::real64;
    register_deferred_rounding_vector(descriptor);
}

solver_result_t solve_conjugate_residual(const solver_config_t &config,
                                         bool enable_compensation,
                                         std::size_t compensation_terms,
                                         std::ostream &out) {
    using namespace compensated_blas::runtime;

    const std::size_t grid = config.grid_extent;
    const std::size_t total_unknowns = grid * grid;

    std::vector<double> x(total_unknowns, 0.0);
    std::vector<double> r(total_unknowns, 0.0);
    std::vector<double> p(total_unknowns, 0.0);
    std::vector<double> ap(total_unknowns, 0.0);
    std::vector<double> ar(total_unknowns, 0.0);
    std::vector<double> b(total_unknowns, 1.0);

    set_backend(backend_kind_t::naive);
    set_compensation_terms(enable_compensation ? compensation_terms : 0);
    clear_deferred_rounding_registrations();

    if (enable_compensation && compensation_terms > 0) {
        register_vector_for_compensation(x.data(), x.size());
        register_vector_for_compensation(r.data(), r.size());
        register_vector_for_compensation(p.data(), p.size());
        register_vector_for_compensation(ap.data(), ap.size());
        register_vector_for_compensation(ar.data(), ar.size());
    }

    apply_laplacian(x, ap, grid);
    copy_backend(b, r);  // initial residual since x = 0
    copy_backend(r, p);
    apply_laplacian(p, ap, grid);

    double rho = dot_backend(r, r);

    solver_result_t result{};
    result.log_entries.reserve(config.max_iterations / config.log_interval + 1);

    for (std::size_t iteration = 0; iteration < config.max_iterations; ++iteration) {
        const double sigma = dot_backend(ap, ap);
        if (sigma == 0.0) {
            break;
        }
        const double alpha = rho / sigma;

        axpy_backend(alpha, p, x);
        const double neg_alpha = -alpha;
        axpy_backend(neg_alpha, ap, r);

        apply_laplacian(r, ar, grid);

        const double rho_next = dot_backend(r, r);
        const double orthogonal_component = dot_backend(ar, ap);
        const double beta = orthogonal_component / sigma;

        // Update search direction vectors explicitly to avoid aliasing issues.
        for (std::size_t i = 0; i < total_unknowns; ++i) {
            p[i] = r[i] + beta * p[i];
            ap[i] = ar[i] + beta * ap[i];
        }

        const double residual_norm = std::sqrt(rho_next);
        flush_vector_if_deferred(r.data(), r.size());

        if (iteration % config.log_interval == 0 || iteration + 1 == config.max_iterations) {
            result.log_entries.push_back({iteration, residual_norm});
        }

        if (residual_norm < config.tolerance) {
            result.converged = true;
            result.iterations = iteration + 1;
            result.final_residual_norm = residual_norm;
            break;
        }

        rho = rho_next;
    }

    if (!result.converged) {
        result.iterations = config.max_iterations;
        result.final_residual_norm = std::sqrt(rho);
    }

    if (enable_compensation && compensation_terms > 0) {
        flush_vector_if_deferred(x.data(), x.size());
        flush_vector_if_deferred(r.data(), r.size());
        flush_vector_if_deferred(p.data(), p.size());
        flush_vector_if_deferred(ap.data(), ap.size());
        flush_vector_if_deferred(ar.data(), ar.size());
    }

    clear_deferred_rounding_registrations();
    set_backend(backend_kind_t::empty);

    out << (enable_compensation ? "[compensated]" : "[plain]")
        << " iterations=" << result.iterations
        << " final_residual=" << result.final_residual_norm << '\n';
    out << "    residual norms:" << '\n';
    for (const auto &entry : result.log_entries) {
        out << "      iter=" << std::setw(6) << entry.iteration
            << " residual=" << std::setw(14) << entry.residual_norm << '\n';
    }

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

    std::cout << "Conjugate Residual solver on " << config.grid_extent << "x" << config.grid_extent
              << " grid (" << config.grid_extent * config.grid_extent << " unknowns)" << '\n';
    std::cout << "max iterations=" << config.max_iterations
              << " tolerance=" << config.tolerance
              << " log interval=" << config.log_interval << '\n';

    (void)solve_conjugate_residual(config, false, 0, std::cout);
    std::cout << '\n';
    (void)solve_conjugate_residual(config, true, config.compensation_terms, std::cout);

    return 0;
}
