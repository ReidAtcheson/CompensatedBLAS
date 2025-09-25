#include "reidblas_blas_ilp64.h"
#include "reidblas_blas_ilp64_internal.hpp"

namespace reidblas::ilp64 {

void drotg(
    double *a,
    double *b,
    double *c,
    double *s
) {
    (void)a;
    (void)b;
    (void)c;
    (void)s;
}

}  // namespace reidblas::ilp64

#if REIDBLAS_USE_ILP64
extern "C" void drotg_(
    double *a,
    double *b,
    double *c,
    double *s
) {
    reidblas::ilp64::drotg(a, b, c, s);
}
#endif  // REIDBLAS_USE_ILP64
