#include "reidblas_blas_ilp64.h"
#include "reidblas_blas_ilp64_internal.hpp"

namespace reidblas::ilp64 {

void srotg(
    float *a,
    float *b,
    float *c,
    float *s
) {
    (void)a;
    (void)b;
    (void)c;
    (void)s;
}

}  // namespace reidblas::ilp64

#if REIDBLAS_USE_ILP64
extern "C" void srotg_(
    float *a,
    float *b,
    float *c,
    float *s
) {
    reidblas::ilp64::srotg(a, b, c, s);
}
#endif  // REIDBLAS_USE_ILP64
