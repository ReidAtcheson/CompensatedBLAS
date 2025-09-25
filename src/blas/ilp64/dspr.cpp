#include "reidblas_blas_ilp64.h"
#include "reidblas_blas_ilp64_internal.hpp"

namespace reidblas::ilp64 {

void dspr(
    const char *uplo,
    const std::int64_t *n,
    const double *alpha,
    const double *x,
    const std::int64_t *incx,
    double *ap
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
extern "C" void dspr_(
    const char *uplo,
    const reidblas_blas_int *n,
    const double *alpha,
    const double *x,
    const reidblas_blas_int *incx,
    double *ap
) {
    reidblas::ilp64::dspr(uplo, n, alpha, x, incx, ap);
}
#endif  // REIDBLAS_USE_ILP64
