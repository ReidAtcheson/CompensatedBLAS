#include "reidblas_blas_ilp64.h"
#include "reidblas_blas_ilp64_internal.hpp"

namespace reidblas::ilp64 {

void strmv(
    const char *uplo,
    const char *trans,
    const char *diag,
    const std::int64_t *n,
    const float *a,
    const std::int64_t *lda,
    float *x,
    const std::int64_t *incx
) {
    (void)uplo;
    (void)trans;
    (void)diag;
    (void)n;
    (void)a;
    (void)lda;
    (void)x;
    (void)incx;
}

}  // namespace reidblas::ilp64

#if REIDBLAS_USE_ILP64
extern "C" void strmv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const reidblas_blas_int *n,
    const float *a,
    const reidblas_blas_int *lda,
    float *x,
    const reidblas_blas_int *incx
) {
    reidblas::ilp64::strmv(uplo, trans, diag, n, a, lda, x, incx);
}
#endif  // REIDBLAS_USE_ILP64
