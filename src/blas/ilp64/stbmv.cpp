#include "reidblas_blas_ilp64.h"
#include "reidblas_blas_ilp64_internal.hpp"

namespace reidblas::ilp64 {

void stbmv(
    const char *uplo,
    const char *trans,
    const char *diag,
    const std::int64_t *n,
    const std::int64_t *k,
    const float *a,
    const std::int64_t *lda,
    float *x,
    const std::int64_t *incx
) {
    (void)uplo;
    (void)trans;
    (void)diag;
    (void)n;
    (void)k;
    (void)a;
    (void)lda;
    (void)x;
    (void)incx;
}

}  // namespace reidblas::ilp64

#if REIDBLAS_USE_ILP64
extern "C" void stbmv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const reidblas_blas_int *n,
    const reidblas_blas_int *k,
    const float *a,
    const reidblas_blas_int *lda,
    float *x,
    const reidblas_blas_int *incx
) {
    reidblas::ilp64::stbmv(uplo, trans, diag, n, k, a, lda, x, incx);
}
#endif  // REIDBLAS_USE_ILP64
