#include "reidblas_blas_ilp64.h"
#include "reidblas_blas_ilp64_internal.hpp"

namespace reidblas::ilp64 {

void cgerc(
    const std::int64_t *m,
    const std::int64_t *n,
    const reidblas_complex_float *alpha,
    const reidblas_complex_float *x,
    const std::int64_t *incx,
    const reidblas_complex_float *y,
    const std::int64_t *incy,
    reidblas_complex_float *a,
    const std::int64_t *lda
) {
    (void)m;
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
extern "C" void cgerc_(
    const reidblas_blas_int *m,
    const reidblas_blas_int *n,
    const reidblas_complex_float *alpha,
    const reidblas_complex_float *x,
    const reidblas_blas_int *incx,
    const reidblas_complex_float *y,
    const reidblas_blas_int *incy,
    reidblas_complex_float *a,
    const reidblas_blas_int *lda
) {
    reidblas::ilp64::cgerc(m, n, alpha, x, incx, y, incy, a, lda);
}
#endif  // REIDBLAS_USE_ILP64
