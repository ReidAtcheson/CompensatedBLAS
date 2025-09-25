#include "reidblas_blas_ilp64.h"
#include "reidblas_blas_ilp64_internal.hpp"

namespace reidblas::ilp64 {

void zrotg(
    reidblas_complex_double *a,
    const reidblas_complex_double *b,
    double *c,
    reidblas_complex_double *s
) {
    (void)a;
    (void)b;
    (void)c;
    (void)s;
}

}  // namespace reidblas::ilp64

#if REIDBLAS_USE_ILP64
extern "C" void zrotg_(
    reidblas_complex_double *a,
    const reidblas_complex_double *b,
    double *c,
    reidblas_complex_double *s
) {
    reidblas::ilp64::zrotg(a, b, c, s);
}
#endif  // REIDBLAS_USE_ILP64
