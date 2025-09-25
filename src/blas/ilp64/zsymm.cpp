#include "reidblas_blas_ilp64.h"
#include "reidblas_blas_ilp64_internal.hpp"

namespace reidblas::ilp64 {

void zsymm(
    const char *side,
    const char *uplo,
    const std::int64_t *m,
    const std::int64_t *n,
    const reidblas_complex_double *alpha,
    const reidblas_complex_double *a,
    const std::int64_t *lda,
    const reidblas_complex_double *b,
    const std::int64_t *ldb,
    const reidblas_complex_double *beta,
    reidblas_complex_double *c,
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
extern "C" void zsymm_(
    const char *side,
    const char *uplo,
    const reidblas_blas_int *m,
    const reidblas_blas_int *n,
    const reidblas_complex_double *alpha,
    const reidblas_complex_double *a,
    const reidblas_blas_int *lda,
    const reidblas_complex_double *b,
    const reidblas_blas_int *ldb,
    const reidblas_complex_double *beta,
    reidblas_complex_double *c,
    const reidblas_blas_int *ldc
) {
    reidblas::ilp64::zsymm(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}
#endif  // REIDBLAS_USE_ILP64
