#include "reidblas_blas_ilp64.h"
#include "reidblas_blas_ilp64_internal.hpp"

namespace reidblas::ilp64 {

void zhemv(
    const char *uplo,
    const std::int64_t *n,
    const reidblas_complex_double *alpha,
    const reidblas_complex_double *a,
    const std::int64_t *lda,
    const reidblas_complex_double *x,
    const std::int64_t *incx,
    const reidblas_complex_double *beta,
    reidblas_complex_double *y,
    const std::int64_t *incy
) {
    (void)uplo;
    (void)n;
    (void)alpha;
    (void)a;
    (void)lda;
    (void)x;
    (void)incx;
    (void)beta;
    (void)y;
    (void)incy;
}

}  // namespace reidblas::ilp64

#if REIDBLAS_USE_ILP64
extern "C" void zhemv_(
    const char *uplo,
    const reidblas_blas_int *n,
    const reidblas_complex_double *alpha,
    const reidblas_complex_double *a,
    const reidblas_blas_int *lda,
    const reidblas_complex_double *x,
    const reidblas_blas_int *incx,
    const reidblas_complex_double *beta,
    reidblas_complex_double *y,
    const reidblas_blas_int *incy
) {
    reidblas::ilp64::zhemv(uplo, n, alpha, a, lda, x, incx, beta, y, incy);
}
#endif  // REIDBLAS_USE_ILP64
