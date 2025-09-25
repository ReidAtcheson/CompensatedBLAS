#include "reidblas_blas_ilp64.h"
#include "reidblas_blas_ilp64_internal.hpp"

namespace reidblas::ilp64 {

void ztbsv(
    const char *uplo,
    const char *trans,
    const char *diag,
    const std::int64_t *n,
    const std::int64_t *k,
    const reidblas_complex_double *a,
    const std::int64_t *lda,
    reidblas_complex_double *x,
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
extern "C" void ztbsv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const reidblas_blas_int *n,
    const reidblas_blas_int *k,
    const reidblas_complex_double *a,
    const reidblas_blas_int *lda,
    reidblas_complex_double *x,
    const reidblas_blas_int *incx
) {
    reidblas::ilp64::ztbsv(uplo, trans, diag, n, k, a, lda, x, incx);
}
#endif  // REIDBLAS_USE_ILP64
