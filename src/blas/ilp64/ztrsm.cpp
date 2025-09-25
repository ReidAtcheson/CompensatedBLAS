#include "reidblas_blas_ilp64.h"
#include "reidblas_blas_ilp64_internal.hpp"

namespace reidblas::ilp64 {

void ztrsm(
    const char *side,
    const char *uplo,
    const char *transa,
    const char *diag,
    const std::int64_t *m,
    const std::int64_t *n,
    const reidblas_complex_double *alpha,
    const reidblas_complex_double *a,
    const std::int64_t *lda,
    reidblas_complex_double *b,
    const std::int64_t *ldb
) {
    (void)side;
    (void)uplo;
    (void)transa;
    (void)diag;
    (void)m;
    (void)n;
    (void)alpha;
    (void)a;
    (void)lda;
    (void)b;
    (void)ldb;
}

}  // namespace reidblas::ilp64

#if REIDBLAS_USE_ILP64
extern "C" void ztrsm_(
    const char *side,
    const char *uplo,
    const char *transa,
    const char *diag,
    const reidblas_blas_int *m,
    const reidblas_blas_int *n,
    const reidblas_complex_double *alpha,
    const reidblas_complex_double *a,
    const reidblas_blas_int *lda,
    reidblas_complex_double *b,
    const reidblas_blas_int *ldb
) {
    reidblas::ilp64::ztrsm(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}
#endif  // REIDBLAS_USE_ILP64
