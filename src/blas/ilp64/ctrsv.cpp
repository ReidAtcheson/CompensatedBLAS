#include "reidblas_blas_ilp64.h"
#include "reidblas_blas_ilp64_internal.hpp"

namespace reidblas::ilp64 {

void ctrsv(
    const char *uplo,
    const char *trans,
    const char *diag,
    const std::int64_t *n,
    const reidblas_complex_float *a,
    const std::int64_t *lda,
    reidblas_complex_float *x,
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
extern "C" void ctrsv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const reidblas_blas_int *n,
    const reidblas_complex_float *a,
    const reidblas_blas_int *lda,
    reidblas_complex_float *x,
    const reidblas_blas_int *incx
) {
    reidblas::ilp64::ctrsv(uplo, trans, diag, n, a, lda, x, incx);
}
#endif  // REIDBLAS_USE_ILP64
