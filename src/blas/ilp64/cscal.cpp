#include "reidblas_blas_ilp64.h"
#include "reidblas_blas_ilp64_internal.hpp"

namespace reidblas::ilp64 {

void cscal(
    const std::int64_t *n,
    const reidblas_complex_float *alpha,
    reidblas_complex_float *x,
    const std::int64_t *incx
) {
    (void)n;
    (void)alpha;
    (void)x;
    (void)incx;
}

}  // namespace reidblas::ilp64

#if REIDBLAS_USE_ILP64
extern "C" void cscal_(
    const reidblas_blas_int *n,
    const reidblas_complex_float *alpha,
    reidblas_complex_float *x,
    const reidblas_blas_int *incx
) {
    reidblas::ilp64::cscal(n, alpha, x, incx);
}
#endif  // REIDBLAS_USE_ILP64
