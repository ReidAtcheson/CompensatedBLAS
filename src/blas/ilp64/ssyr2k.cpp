#include "reidblas_blas_ilp64.h"
#include "reidblas_blas_ilp64_internal.hpp"

namespace reidblas::ilp64 {

void ssyr2k(
    const char *uplo,
    const char *trans,
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
    (void)uplo;
    (void)trans;
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
extern "C" void ssyr2k_(
    const char *uplo,
    const char *trans,
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
    reidblas::ilp64::ssyr2k(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
#endif  // REIDBLAS_USE_ILP64
