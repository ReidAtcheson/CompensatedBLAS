#include "reidblas_blas_ilp64.h"
#include "reidblas_blas_ilp64_internal.hpp"

namespace reidblas::ilp64 {

void ztpsv(
    const char *uplo,
    const char *trans,
    const char *diag,
    const std::int64_t *n,
    const reidblas_complex_double *ap,
    reidblas_complex_double *x,
    const std::int64_t *incx
) {
    (void)uplo;
    (void)trans;
    (void)diag;
    (void)n;
    (void)ap;
    (void)x;
    (void)incx;
}

}  // namespace reidblas::ilp64

#if REIDBLAS_USE_ILP64
extern "C" void ztpsv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const reidblas_blas_int *n,
    const reidblas_complex_double *ap,
    reidblas_complex_double *x,
    const reidblas_blas_int *incx
) {
    reidblas::ilp64::ztpsv(uplo, trans, diag, n, ap, x, incx);
}
#endif  // REIDBLAS_USE_ILP64
