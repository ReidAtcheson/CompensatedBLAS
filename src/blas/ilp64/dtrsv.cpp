#include "reidblas_blas_ilp64.h"
#include "reidblas_blas_ilp64_internal.hpp"

namespace reidblas::ilp64 {

void dtrsv(
    const char *uplo,
    const char *trans,
    const char *diag,
    const std::int64_t *n,
    const double *a,
    const std::int64_t *lda,
    double *x,
    const std::int64_t *incx
) {
    (void)uplo;
    (void)trans;
    (void)diag;
    (void)n;
    (void)a;
    (void)lda;
    (void)x;
    (void)incx;
}

}  // namespace reidblas::ilp64

#if REIDBLAS_USE_ILP64
extern "C" void dtrsv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const reidblas_blas_int *n,
    const double *a,
    const reidblas_blas_int *lda,
    double *x,
    const reidblas_blas_int *incx
) {
    reidblas::ilp64::dtrsv(uplo, trans, diag, n, a, lda, x, incx);
}
#endif  // REIDBLAS_USE_ILP64
