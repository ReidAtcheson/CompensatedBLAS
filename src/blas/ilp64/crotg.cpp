#include "reidblas_blas_ilp64.h"
#include "reidblas_blas_ilp64_internal.hpp"

namespace reidblas::ilp64 {

void crotg(
    reidblas_complex_float *a,
    const reidblas_complex_float *b,
    float *c,
    reidblas_complex_float *s
) {
    (void)a;
    (void)b;
    (void)c;
    (void)s;
}

}  // namespace reidblas::ilp64

#if REIDBLAS_USE_ILP64
extern "C" void crotg_(
    reidblas_complex_float *a,
    const reidblas_complex_float *b,
    float *c,
    reidblas_complex_float *s
) {
    reidblas::ilp64::crotg(a, b, c, s);
}
#endif  // REIDBLAS_USE_ILP64
