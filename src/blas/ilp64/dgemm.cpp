#include "reidblas_blas_ilp64.h"
#include "reidblas_blas_ilp64_internal.hpp"

namespace reidblas::ilp64 {

void dgemm(
    const char *transa,
    const char *transb,
    const std::int64_t *m,
    const std::int64_t *n,
    const std::int64_t *k,
    const double *alpha,
    const double *a,
    const std::int64_t *lda,
    const double *b,
    const std::int64_t *ldb,
    const double *beta,
    double *c,
    const std::int64_t *ldc
) {
    (void)transa;
    (void)transb;
    (void)m;
    (void)n;
    (void)k;
    (void)alpha;
    (void)a;
    (void)lda;
    (void)b;
    (void)ldb;
    (void)beta;
    (void)c;
    (void)ldc;
}

}  // namespace reidblas::ilp64

#if REIDBLAS_USE_ILP64
extern "C" void dgemm_(
    const char *transa,
    const char *transb,
    const reidblas_blas_int *m,
    const reidblas_blas_int *n,
    const reidblas_blas_int *k,
    const double *alpha,
    const double *a,
    const reidblas_blas_int *lda,
    const double *b,
    const reidblas_blas_int *ldb,
    const double *beta,
    double *c,
    const reidblas_blas_int *ldc
) {
    reidblas::ilp64::dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
#endif  // REIDBLAS_USE_ILP64
