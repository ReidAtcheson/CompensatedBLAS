#include "reidblas_blas_ilp64.h"
#include "reidblas_blas_ilp64_internal.hpp"

namespace reidblas::ilp64 {

void zher(
    const char *uplo,
    const std::int64_t *n,
    const double *alpha,
    const reidblas_complex_double *x,
    const std::int64_t *incx,
    reidblas_complex_double *a,
    const std::int64_t *lda
) {
    (void)uplo;
    (void)n;
    (void)alpha;
    (void)x;
    (void)incx;
    (void)a;
    (void)lda;
}

}  // namespace reidblas::ilp64

#if REIDBLAS_USE_ILP64
extern "C" void zher_(
    const char *uplo,
    const reidblas_blas_int *n,
    const double *alpha,
    const reidblas_complex_double *x,
    const reidblas_blas_int *incx,
    reidblas_complex_double *a,
    const reidblas_blas_int *lda
) {
    reidblas::ilp64::zher(uplo, n, alpha, x, incx, a, lda);
}
#endif  // REIDBLAS_USE_ILP64
