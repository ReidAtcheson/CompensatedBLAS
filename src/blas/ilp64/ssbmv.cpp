#include "reidblas_blas_ilp64.h"
#include "reidblas_blas_ilp64_internal.hpp"

namespace reidblas::ilp64 {

void ssbmv(
    const char *uplo,
    const std::int64_t *n,
    const std::int64_t *k,
    const float *alpha,
    const float *a,
    const std::int64_t *lda,
    const float *x,
    const std::int64_t *incx,
    const float *beta,
    float *y,
    const std::int64_t *incy
) {
    (void)uplo;
    (void)n;
    (void)k;
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
extern "C" void ssbmv_(
    const char *uplo,
    const reidblas_blas_int *n,
    const reidblas_blas_int *k,
    const float *alpha,
    const float *a,
    const reidblas_blas_int *lda,
    const float *x,
    const reidblas_blas_int *incx,
    const float *beta,
    float *y,
    const reidblas_blas_int *incy
) {
    reidblas::ilp64::ssbmv(uplo, n, k, alpha, a, lda, x, incx, beta, y, incy);
}
#endif  // REIDBLAS_USE_ILP64
