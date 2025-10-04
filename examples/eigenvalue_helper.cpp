#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

extern "C" {
void dsyev_(const char *jobz,
            const char *uplo,
            int *n,
            double *a,
            int *lda,
            double *w,
            double *work,
            int *lwork,
            int *info);
}

namespace {

struct config_t {
    std::string file_path;
    std::size_t eigenvalues = 5;
};

void print_help() {
    std::cout << "Eigenvalue helper\n"
              << "Usage: eigenvalue_helper --file=<path> [--k=<count>]\n"
              << "Reads a symmetric matrix dump and reports the smallest eigenvalues\n";
}

std::optional<config_t> parse_args(int argc, char **argv) {
    config_t config;
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--help") {
            print_help();
            return std::nullopt;
        }
        auto pos = arg.find('=');
        std::string key = arg;
        std::string value;
        if (pos != std::string::npos) {
            key = arg.substr(0, pos);
            value = arg.substr(pos + 1);
        }
        if (key == "--file") {
            config.file_path = value;
        } else if (key == "--k") {
            config.eigenvalues = static_cast<std::size_t>(std::stoul(value));
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            return std::nullopt;
        }
    }
    if (config.file_path.empty()) {
        std::cerr << "Missing required --file argument.\n";
        return std::nullopt;
    }
    if (config.eigenvalues == 0) {
        config.eigenvalues = 1;
    }
    return config;
}

bool load_matrix(const std::string &path, std::size_t &n, std::vector<double> &matrix) {
    std::ifstream in(path);
    if (!in) {
        std::cerr << "Unable to open matrix file: " << path << "\n";
        return false;
    }

    std::size_t rows = 0;
    std::size_t cols = 0;
    if (!(in >> rows >> cols)) {
        std::cerr << "Failed to read matrix dimensions from " << path << "\n";
        return false;
    }
    if (rows != cols) {
        std::cerr << "Matrix in " << path << " is not square (" << rows << "x" << cols << ").\n";
        return false;
    }
    n = rows;
    matrix.assign(n * n, 0.0);
    for (std::size_t row = 0; row < n; ++row) {
        for (std::size_t col = 0; col < n; ++col) {
            double value = 0.0;
            if (!(in >> value)) {
                std::cerr << "Unexpected end of data while reading matrix." << "\n";
                return false;
            }
            matrix[col * n + row] = value;
        }
    }
    return true;
}

bool compute_eigenvalues(std::size_t n,
                         std::vector<double> &matrix,
                         std::vector<double> &eigenvalues) {
    if (n == 0) {
        return true;
    }
    eigenvalues.assign(n, 0.0);
    int n32 = static_cast<int>(n);
    int lda = n32;
    int lwork = -1;
    double work_query = 0.0;
    int info = 0;
    char jobz = 'N';
    char uplo = 'L';

    dsyev_(&jobz, &uplo, &n32, matrix.data(), &lda, eigenvalues.data(), &work_query, &lwork, &info);
    if (info != 0) {
        std::cerr << "dsyev_ workspace query failed with info=" << info << "\n";
        return false;
    }
    lwork = std::max(1, static_cast<int>(work_query));
    std::vector<double> work(static_cast<std::size_t>(lwork));
    dsyev_(&jobz, &uplo, &n32, matrix.data(), &lda, eigenvalues.data(), work.data(), &lwork, &info);
    if (info != 0) {
        std::cerr << "dsyev_ failed with info=" << info << "\n";
        return false;
    }
    return true;
}

}  // namespace

int main(int argc, char **argv) {
    auto config_opt = parse_args(argc, argv);
    if (!config_opt.has_value()) {
        return 1;
    }
    const config_t config = *config_opt;

    std::size_t n = 0;
    std::vector<double> matrix;
    if (!load_matrix(config.file_path, n, matrix)) {
        return 1;
    }
    if (n == 0) {
        std::cout << "Matrix is empty." << std::endl;
        return 0;
    }

    std::vector<double> eigenvalues;
    if (!compute_eigenvalues(n, matrix, eigenvalues)) {
        return 2;
    }

    const std::size_t k = std::min<std::size_t>(config.eigenvalues, n);
    std::cout << std::setprecision(17);
    std::cout << "Smallest " << k << " eigenvalues:" << std::endl;
    for (std::size_t i = 0; i < k; ++i) {
        std::cout << "  " << eigenvalues[i] << std::endl;
    }
    return 0;
}
