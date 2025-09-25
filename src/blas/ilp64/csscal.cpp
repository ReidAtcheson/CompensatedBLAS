#include "reidblas_blas_ilp64.h"
#include "reidblas_blas_ilp64_internal.hpp"

namespace reidblas::ilp64 {

void csscal(
    const std::int64_t *n,
    const float *alpha,
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
extern "C" void csscal_(
    const reidblas_blas_int *n,
    const float *alpha,
    reidblas_complex_float *x,
    const reidblas_blas_int *incx
) {
    reidblas::ilp64::csscal(n, alpha, x, incx);
}
#endif  // REIDBLAS_USE_ILP64
