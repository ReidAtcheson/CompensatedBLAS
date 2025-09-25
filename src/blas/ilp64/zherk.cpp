#include "reidblas_blas_ilp64.h"
#include "reidblas_blas_ilp64_internal.hpp"

namespace reidblas::ilp64 {

void zherk(
    const char *uplo,
    const char *trans,
    const std::int64_t *n,
    const std::int64_t *k,
    const double *alpha,
    const reidblas_complex_double *a,
    const std::int64_t *lda,
    const double *beta,
    reidblas_complex_double *c,
    const std::int64_t *ldc
) {
    (void)uplo;
    (void)trans;
    (void)n;
    (void)k;
    (void)alpha;
    (void)a;
    (void)lda;
    (void)beta;
    (void)c;
    (void)ldc;
}

}  // namespace reidblas::ilp64

#if REIDBLAS_USE_ILP64
extern "C" void zherk_(
    const char *uplo,
    const char *trans,
    const reidblas_blas_int *n,
    const reidblas_blas_int *k,
    const double *alpha,
    const reidblas_complex_double *a,
    const reidblas_blas_int *lda,
    const double *beta,
    reidblas_complex_double *c,
    const reidblas_blas_int *ldc
) {
    reidblas::ilp64::zherk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}
#endif  // REIDBLAS_USE_ILP64
