#include "reidblas_blas_ilp64.h"
#include "reidblas_blas_ilp64_internal.hpp"

namespace reidblas::ilp64 {

void dsyr2(
    const char *uplo,
    const std::int64_t *n,
    const double *alpha,
    const double *x,
    const std::int64_t *incx,
    const double *y,
    const std::int64_t *incy,
    double *a,
    const std::int64_t *lda
) {
    (void)uplo;
    (void)n;
    (void)alpha;
    (void)x;
    (void)incx;
    (void)y;
    (void)incy;
    (void)a;
    (void)lda;
}

}  // namespace reidblas::ilp64

#if REIDBLAS_USE_ILP64
extern "C" void dsyr2_(
    const char *uplo,
    const reidblas_blas_int *n,
    const double *alpha,
    const double *x,
    const reidblas_blas_int *incx,
    const double *y,
    const reidblas_blas_int *incy,
    double *a,
    const reidblas_blas_int *lda
) {
    reidblas::ilp64::dsyr2(uplo, n, alpha, x, incx, y, incy, a, lda);
}
#endif  // REIDBLAS_USE_ILP64
