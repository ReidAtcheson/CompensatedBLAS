#include "reidblas_blas_ilp64.h"
#include "reidblas_blas_ilp64_internal.hpp"

namespace reidblas::ilp64 {

void csyr2k(
    const char *uplo,
    const char *trans,
    const std::int64_t *n,
    const std::int64_t *k,
    const reidblas_complex_float *alpha,
    const reidblas_complex_float *a,
    const std::int64_t *lda,
    const reidblas_complex_float *b,
    const std::int64_t *ldb,
    const reidblas_complex_float *beta,
    reidblas_complex_float *c,
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
extern "C" void csyr2k_(
    const char *uplo,
    const char *trans,
    const reidblas_blas_int *n,
    const reidblas_blas_int *k,
    const reidblas_complex_float *alpha,
    const reidblas_complex_float *a,
    const reidblas_blas_int *lda,
    const reidblas_complex_float *b,
    const reidblas_blas_int *ldb,
    const reidblas_complex_float *beta,
    reidblas_complex_float *c,
    const reidblas_blas_int *ldc
) {
    reidblas::ilp64::csyr2k(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
#endif  // REIDBLAS_USE_ILP64
