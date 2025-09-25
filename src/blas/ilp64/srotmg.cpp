#include "reidblas_blas_ilp64.h"
#include "reidblas_blas_ilp64_internal.hpp"

namespace reidblas::ilp64 {

void srotmg(
    float *d1,
    float *d2,
    float *x1,
    const float *y1,
    float *param
) {
    (void)d1;
    (void)d2;
    (void)x1;
    (void)y1;
    (void)param;
}

}  // namespace reidblas::ilp64

#if REIDBLAS_USE_ILP64
extern "C" void srotmg_(
    float *d1,
    float *d2,
    float *x1,
    const float *y1,
    float *param
) {
    reidblas::ilp64::srotmg(d1, d2, x1, y1, param);
}
#endif  // REIDBLAS_USE_ILP64
