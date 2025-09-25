#include "reidblas_blas_ilp64.h"
#include "reidblas_blas_ilp64_internal.hpp"

namespace reidblas::ilp64 {

void drotmg(
    double *d1,
    double *d2,
    double *x1,
    const double *y1,
    double *param
) {
    (void)d1;
    (void)d2;
    (void)x1;
    (void)y1;
    (void)param;
}

}  // namespace reidblas::ilp64

#if REIDBLAS_USE_ILP64
extern "C" void drotmg_(
    double *d1,
    double *d2,
    double *x1,
    const double *y1,
    double *param
) {
    reidblas::ilp64::drotmg(d1, d2, x1, y1, param);
}
#endif  // REIDBLAS_USE_ILP64
