#include "reidblas_blas_ilp64.h"
#include "reidblas_blas_ilp64_internal.hpp"

namespace reidblas::ilp64 {

void zhpmv(
    const char *uplo,
    const std::int64_t *n,
    const reidblas_complex_double *alpha,
    const reidblas_complex_double *ap,
    const reidblas_complex_double *x,
    const std::int64_t *incx,
    const reidblas_complex_double *beta,
    reidblas_complex_double *y,
    const std::int64_t *incy
) {
    (void)uplo;
    (void)n;
    (void)alpha;
    (void)ap;
    (void)x;
    (void)incx;
    (void)beta;
    (void)y;
    (void)incy;
}

}  // namespace reidblas::ilp64

#if REIDBLAS_USE_ILP64
extern "C" void zhpmv_(
    const char *uplo,
    const reidblas_blas_int *n,
    const reidblas_complex_double *alpha,
    const reidblas_complex_double *ap,
    const reidblas_complex_double *x,
    const reidblas_blas_int *incx,
    const reidblas_complex_double *beta,
    reidblas_complex_double *y,
    const reidblas_blas_int *incy
) {
    reidblas::ilp64::zhpmv(uplo, n, alpha, ap, x, incx, beta, y, incy);
}
#endif  // REIDBLAS_USE_ILP64
