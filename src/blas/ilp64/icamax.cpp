#include "reidblas_blas_ilp64.h"
#include "reidblas_blas_ilp64_internal.hpp"

namespace reidblas::ilp64 {

std::int64_t icamax(
    const std::int64_t *n,
    const reidblas_complex_float *x,
    const std::int64_t *incx
) {
    (void)n;
    (void)x;
    (void)incx;
    return 0;
}

}  // namespace reidblas::ilp64

#if REIDBLAS_USE_ILP64
extern "C" reidblas_blas_int icamax_(
    const reidblas_blas_int *n,
    const reidblas_complex_float *x,
    const reidblas_blas_int *incx
) {
    return reidblas::ilp64::icamax(n, x, incx);
}
#endif  // REIDBLAS_USE_ILP64
