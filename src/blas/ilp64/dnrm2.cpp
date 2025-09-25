#include "reidblas_blas_ilp64.h"
#include "reidblas_blas_ilp64_internal.hpp"

namespace reidblas::ilp64 {

double dnrm2(
    const std::int64_t *n,
    const double *x,
    const std::int64_t *incx
) {
    (void)n;
    (void)x;
    (void)incx;
    return 0.0;
}

}  // namespace reidblas::ilp64

#if REIDBLAS_USE_ILP64
extern "C" double dnrm2_(
    const reidblas_blas_int *n,
    const double *x,
    const reidblas_blas_int *incx
) {
    return reidblas::ilp64::dnrm2(n, x, incx);
}
#endif  // REIDBLAS_USE_ILP64
