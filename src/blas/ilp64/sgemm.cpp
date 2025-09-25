#include "reidblas_blas_ilp64.h"
#include "reidblas_blas_ilp64_internal.hpp"

namespace reidblas::ilp64 {

void sgemm(
    const char *transa,
    const char *transb,
    const std::int64_t *m,
    const std::int64_t *n,
    const std::int64_t *k,
    const float *alpha,
    const float *a,
    const std::int64_t *lda,
    const float *b,
    const std::int64_t *ldb,
    const float *beta,
    float *c,
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
extern "C" void sgemm_(
    const char *transa,
    const char *transb,
    const reidblas_blas_int *m,
    const reidblas_blas_int *n,
    const reidblas_blas_int *k,
    const float *alpha,
    const float *a,
    const reidblas_blas_int *lda,
    const float *b,
    const reidblas_blas_int *ldb,
    const float *beta,
    float *c,
    const reidblas_blas_int *ldc
) {
    reidblas::ilp64::sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
#endif  // REIDBLAS_USE_ILP64
