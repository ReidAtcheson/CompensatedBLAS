#include "reidblas_blas_ilp64.h"
#include "reidblas_blas_ilp64_internal.hpp"

namespace reidblas::ilp64 {

void zhpr2(
    const char *uplo,
    const std::int64_t *n,
    const reidblas_complex_double *alpha,
    const reidblas_complex_double *x,
    const std::int64_t *incx,
    const reidblas_complex_double *y,
    const std::int64_t *incy,
    reidblas_complex_double *ap
) {
    (void)uplo;
    (void)n;
    (void)alpha;
    (void)x;
    (void)incx;
    (void)y;
    (void)incy;
    (void)ap;
}

}  // namespace reidblas::ilp64

#if REIDBLAS_USE_ILP64
extern "C" void zhpr2_(
    const char *uplo,
    const reidblas_blas_int *n,
    const reidblas_complex_double *alpha,
    const reidblas_complex_double *x,
    const reidblas_blas_int *incx,
    const reidblas_complex_double *y,
    const reidblas_blas_int *incy,
    reidblas_complex_double *ap
) {
    reidblas::ilp64::zhpr2(uplo, n, alpha, x, incx, y, incy, ap);
}
#endif  // REIDBLAS_USE_ILP64
