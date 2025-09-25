#include "reidblas_blas_ilp64.h"
#include "reidblas_blas_ilp64_internal.hpp"

namespace reidblas::ilp64 {

void ssymm(
    const char *side,
    const char *uplo,
    const std::int64_t *m,
    const std::int64_t *n,
    const float *alpha,
    const float *a,
    const std::int64_t *lda,
    const float *b,
    const std::int64_t *ldb,
    const float *beta,
    float *c,
    const std::int64_t *ldc
) {
    (void)side;
    (void)uplo;
    (void)m;
    (void)n;
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
extern "C" void ssymm_(
    const char *side,
    const char *uplo,
    const reidblas_blas_int *m,
    const reidblas_blas_int *n,
    const float *alpha,
    const float *a,
    const reidblas_blas_int *lda,
    const float *b,
    const reidblas_blas_int *ldb,
    const float *beta,
    float *c,
    const reidblas_blas_int *ldc
) {
    reidblas::ilp64::ssymm(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}
#endif  // REIDBLAS_USE_ILP64
