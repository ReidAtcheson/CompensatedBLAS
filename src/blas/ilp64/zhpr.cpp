#include "reidblas_blas_ilp64.h"
#include "reidblas_blas_ilp64_internal.hpp"

namespace reidblas::ilp64 {

void zhpr(
    const char *uplo,
    const std::int64_t *n,
    const double *alpha,
    const reidblas_complex_double *x,
    const std::int64_t *incx,
    reidblas_complex_double *ap
) {
    (void)uplo;
    (void)n;
    (void)alpha;
    (void)x;
    (void)incx;
    (void)ap;
}

}  // namespace reidblas::ilp64

#if REIDBLAS_USE_ILP64
extern "C" void zhpr_(
    const char *uplo,
    const reidblas_blas_int *n,
    const double *alpha,
    const reidblas_complex_double *x,
    const reidblas_blas_int *incx,
    reidblas_complex_double *ap
) {
    reidblas::ilp64::zhpr(uplo, n, alpha, x, incx, ap);
}
#endif  // REIDBLAS_USE_ILP64
