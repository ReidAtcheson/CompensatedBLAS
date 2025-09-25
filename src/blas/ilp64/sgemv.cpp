#include "reidblas_blas_ilp64.h"
#include "reidblas_blas_ilp64_internal.hpp"

namespace reidblas::ilp64 {

void sgemv(
    const char *trans,
    const std::int64_t *m,
    const std::int64_t *n,
    const float *alpha,
    const float *a,
    const std::int64_t *lda,
    const float *x,
    const std::int64_t *incx,
    const float *beta,
    float *y,
    const std::int64_t *incy
) {
    (void)trans;
    (void)m;
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
extern "C" void sgemv_(
    const char *trans,
    const reidblas_blas_int *m,
    const reidblas_blas_int *n,
    const float *alpha,
    const float *a,
    const reidblas_blas_int *lda,
    const float *x,
    const reidblas_blas_int *incx,
    const float *beta,
    float *y,
    const reidblas_blas_int *incy
) {
    reidblas::ilp64::sgemv(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}
#endif  // REIDBLAS_USE_ILP64
