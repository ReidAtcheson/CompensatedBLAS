#pragma once

#include <cstdint>

namespace reidblas::ilp64 {

void srotg(
    float *a,
    float *b,
    float *c,
    float *s
);

void srotmg(
    float *d1,
    float *d2,
    float *x1,
    const float *y1,
    float *param
);

void srot(
    const std::int64_t *n,
    float *x,
    const std::int64_t *incx,
    float *y,
    const std::int64_t *incy,
    const float *c,
    const float *s
);

void srotm(
    const std::int64_t *n,
    float *x,
    const std::int64_t *incx,
    float *y,
    const std::int64_t *incy,
    const float *param
);

void sswap(
    const std::int64_t *n,
    float *x,
    const std::int64_t *incx,
    float *y,
    const std::int64_t *incy
);

void sscal(
    const std::int64_t *n,
    const float *alpha,
    float *x,
    const std::int64_t *incx
);

void scopy(
    const std::int64_t *n,
    const float *x,
    const std::int64_t *incx,
    float *y,
    const std::int64_t *incy
);

void saxpy(
    const std::int64_t *n,
    const float *alpha,
    const float *x,
    const std::int64_t *incx,
    float *y,
    const std::int64_t *incy
);

float sdot(
    const std::int64_t *n,
    const float *x,
    const std::int64_t *incx,
    const float *y,
    const std::int64_t *incy
);

float sdsdot(
    const std::int64_t *n,
    const float *sb,
    const float *x,
    const std::int64_t *incx,
    const float *y,
    const std::int64_t *incy
);

float snrm2(
    const std::int64_t *n,
    const float *x,
    const std::int64_t *incx
);

float sasum(
    const std::int64_t *n,
    const float *x,
    const std::int64_t *incx
);

std::int64_t isamax(
    const std::int64_t *n,
    const float *x,
    const std::int64_t *incx
);

void drotg(
    double *a,
    double *b,
    double *c,
    double *s
);

void drotmg(
    double *d1,
    double *d2,
    double *x1,
    const double *y1,
    double *param
);

void drot(
    const std::int64_t *n,
    double *x,
    const std::int64_t *incx,
    double *y,
    const std::int64_t *incy,
    const double *c,
    const double *s
);

void drotm(
    const std::int64_t *n,
    double *x,
    const std::int64_t *incx,
    double *y,
    const std::int64_t *incy,
    const double *param
);

void dswap(
    const std::int64_t *n,
    double *x,
    const std::int64_t *incx,
    double *y,
    const std::int64_t *incy
);

void dscal(
    const std::int64_t *n,
    const double *alpha,
    double *x,
    const std::int64_t *incx
);

void dcopy(
    const std::int64_t *n,
    const double *x,
    const std::int64_t *incx,
    double *y,
    const std::int64_t *incy
);

void daxpy(
    const std::int64_t *n,
    const double *alpha,
    const double *x,
    const std::int64_t *incx,
    double *y,
    const std::int64_t *incy
);

double ddot(
    const std::int64_t *n,
    const double *x,
    const std::int64_t *incx,
    const double *y,
    const std::int64_t *incy
);

double dsdot(
    const std::int64_t *n,
    const float *x,
    const std::int64_t *incx,
    const float *y,
    const std::int64_t *incy
);

double dnrm2(
    const std::int64_t *n,
    const double *x,
    const std::int64_t *incx
);

double dasum(
    const std::int64_t *n,
    const double *x,
    const std::int64_t *incx
);

std::int64_t idamax(
    const std::int64_t *n,
    const double *x,
    const std::int64_t *incx
);

void crotg(
    reidblas_complex_float *a,
    const reidblas_complex_float *b,
    float *c,
    reidblas_complex_float *s
);

void csrot(
    const std::int64_t *n,
    reidblas_complex_float *x,
    const std::int64_t *incx,
    reidblas_complex_float *y,
    const std::int64_t *incy,
    const float *c,
    const float *s
);

void csscal(
    const std::int64_t *n,
    const float *alpha,
    reidblas_complex_float *x,
    const std::int64_t *incx
);

void cscal(
    const std::int64_t *n,
    const reidblas_complex_float *alpha,
    reidblas_complex_float *x,
    const std::int64_t *incx
);

void cswap(
    const std::int64_t *n,
    reidblas_complex_float *x,
    const std::int64_t *incx,
    reidblas_complex_float *y,
    const std::int64_t *incy
);

void ccopy(
    const std::int64_t *n,
    const reidblas_complex_float *x,
    const std::int64_t *incx,
    reidblas_complex_float *y,
    const std::int64_t *incy
);

void caxpy(
    const std::int64_t *n,
    const reidblas_complex_float *alpha,
    const reidblas_complex_float *x,
    const std::int64_t *incx,
    reidblas_complex_float *y,
    const std::int64_t *incy
);

reidblas_complex_float cdotu(
    const std::int64_t *n,
    const reidblas_complex_float *x,
    const std::int64_t *incx,
    const reidblas_complex_float *y,
    const std::int64_t *incy
);

reidblas_complex_float cdotc(
    const std::int64_t *n,
    const reidblas_complex_float *x,
    const std::int64_t *incx,
    const reidblas_complex_float *y,
    const std::int64_t *incy
);

float scnrm2(
    const std::int64_t *n,
    const reidblas_complex_float *x,
    const std::int64_t *incx
);

float scasum(
    const std::int64_t *n,
    const reidblas_complex_float *x,
    const std::int64_t *incx
);

std::int64_t icamax(
    const std::int64_t *n,
    const reidblas_complex_float *x,
    const std::int64_t *incx
);

void zrotg(
    reidblas_complex_double *a,
    const reidblas_complex_double *b,
    double *c,
    reidblas_complex_double *s
);

void zdrot(
    const std::int64_t *n,
    reidblas_complex_double *x,
    const std::int64_t *incx,
    reidblas_complex_double *y,
    const std::int64_t *incy,
    const double *c,
    const double *s
);

void zdscal(
    const std::int64_t *n,
    const double *alpha,
    reidblas_complex_double *x,
    const std::int64_t *incx
);

void zscal(
    const std::int64_t *n,
    const reidblas_complex_double *alpha,
    reidblas_complex_double *x,
    const std::int64_t *incx
);

void zswap(
    const std::int64_t *n,
    reidblas_complex_double *x,
    const std::int64_t *incx,
    reidblas_complex_double *y,
    const std::int64_t *incy
);

void zcopy(
    const std::int64_t *n,
    const reidblas_complex_double *x,
    const std::int64_t *incx,
    reidblas_complex_double *y,
    const std::int64_t *incy
);

void zaxpy(
    const std::int64_t *n,
    const reidblas_complex_double *alpha,
    const reidblas_complex_double *x,
    const std::int64_t *incx,
    reidblas_complex_double *y,
    const std::int64_t *incy
);

reidblas_complex_double zdotu(
    const std::int64_t *n,
    const reidblas_complex_double *x,
    const std::int64_t *incx,
    const reidblas_complex_double *y,
    const std::int64_t *incy
);

reidblas_complex_double zdotc(
    const std::int64_t *n,
    const reidblas_complex_double *x,
    const std::int64_t *incx,
    const reidblas_complex_double *y,
    const std::int64_t *incy
);

double dznrm2(
    const std::int64_t *n,
    const reidblas_complex_double *x,
    const std::int64_t *incx
);

double dzasum(
    const std::int64_t *n,
    const reidblas_complex_double *x,
    const std::int64_t *incx
);

std::int64_t izamax(
    const std::int64_t *n,
    const reidblas_complex_double *x,
    const std::int64_t *incx
);

void sgemv(
    const char *trans,
    const std::int64_t *m,
    const std::int64_t *n,
    const float *alpha,
    const float *a,
    const std::int64_t *lda,
    const float *x,
    const std::int64_t *incx,
    const float *beta,
    float *y,
    const std::int64_t *incy
);

void sgbmv(
    const char *trans,
    const std::int64_t *m,
    const std::int64_t *n,
    const std::int64_t *kl,
    const std::int64_t *ku,
    const float *alpha,
    const float *a,
    const std::int64_t *lda,
    const float *x,
    const std::int64_t *incx,
    const float *beta,
    float *y,
    const std::int64_t *incy
);

void ssymv(
    const char *uplo,
    const std::int64_t *n,
    const float *alpha,
    const float *a,
    const std::int64_t *lda,
    const float *x,
    const std::int64_t *incx,
    const float *beta,
    float *y,
    const std::int64_t *incy
);

void ssbmv(
    const char *uplo,
    const std::int64_t *n,
    const std::int64_t *k,
    const float *alpha,
    const float *a,
    const std::int64_t *lda,
    const float *x,
    const std::int64_t *incx,
    const float *beta,
    float *y,
    const std::int64_t *incy
);

void sspmv(
    const char *uplo,
    const std::int64_t *n,
    const float *alpha,
    const float *ap,
    const float *x,
    const std::int64_t *incx,
    const float *beta,
    float *y,
    const std::int64_t *incy
);

void strmv(
    const char *uplo,
    const char *trans,
    const char *diag,
    const std::int64_t *n,
    const float *a,
    const std::int64_t *lda,
    float *x,
    const std::int64_t *incx
);

void stbmv(
    const char *uplo,
    const char *trans,
    const char *diag,
    const std::int64_t *n,
    const std::int64_t *k,
    const float *a,
    const std::int64_t *lda,
    float *x,
    const std::int64_t *incx
);

void stpmv(
    const char *uplo,
    const char *trans,
    const char *diag,
    const std::int64_t *n,
    const float *ap,
    float *x,
    const std::int64_t *incx
);

void strsv(
    const char *uplo,
    const char *trans,
    const char *diag,
    const std::int64_t *n,
    const float *a,
    const std::int64_t *lda,
    float *x,
    const std::int64_t *incx
);

void stbsv(
    const char *uplo,
    const char *trans,
    const char *diag,
    const std::int64_t *n,
    const std::int64_t *k,
    const float *a,
    const std::int64_t *lda,
    float *x,
    const std::int64_t *incx
);

void stpsv(
    const char *uplo,
    const char *trans,
    const char *diag,
    const std::int64_t *n,
    const float *ap,
    float *x,
    const std::int64_t *incx
);

void sger(
    const std::int64_t *m,
    const std::int64_t *n,
    const float *alpha,
    const float *x,
    const std::int64_t *incx,
    const float *y,
    const std::int64_t *incy,
    float *a,
    const std::int64_t *lda
);

void sspr(
    const char *uplo,
    const std::int64_t *n,
    const float *alpha,
    const float *x,
    const std::int64_t *incx,
    float *ap
);

void ssyr(
    const char *uplo,
    const std::int64_t *n,
    const float *alpha,
    const float *x,
    const std::int64_t *incx,
    float *a,
    const std::int64_t *lda
);

void sspr2(
    const char *uplo,
    const std::int64_t *n,
    const float *alpha,
    const float *x,
    const std::int64_t *incx,
    const float *y,
    const std::int64_t *incy,
    float *ap
);

void ssyr2(
    const char *uplo,
    const std::int64_t *n,
    const float *alpha,
    const float *x,
    const std::int64_t *incx,
    const float *y,
    const std::int64_t *incy,
    float *a,
    const std::int64_t *lda
);

void dgemv(
    const char *trans,
    const std::int64_t *m,
    const std::int64_t *n,
    const double *alpha,
    const double *a,
    const std::int64_t *lda,
    const double *x,
    const std::int64_t *incx,
    const double *beta,
    double *y,
    const std::int64_t *incy
);

void dgbmv(
    const char *trans,
    const std::int64_t *m,
    const std::int64_t *n,
    const std::int64_t *kl,
    const std::int64_t *ku,
    const double *alpha,
    const double *a,
    const std::int64_t *lda,
    const double *x,
    const std::int64_t *incx,
    const double *beta,
    double *y,
    const std::int64_t *incy
);

void dsymv(
    const char *uplo,
    const std::int64_t *n,
    const double *alpha,
    const double *a,
    const std::int64_t *lda,
    const double *x,
    const std::int64_t *incx,
    const double *beta,
    double *y,
    const std::int64_t *incy
);

void dsbmv(
    const char *uplo,
    const std::int64_t *n,
    const std::int64_t *k,
    const double *alpha,
    const double *a,
    const std::int64_t *lda,
    const double *x,
    const std::int64_t *incx,
    const double *beta,
    double *y,
    const std::int64_t *incy
);

void dspmv(
    const char *uplo,
    const std::int64_t *n,
    const double *alpha,
    const double *ap,
    const double *x,
    const std::int64_t *incx,
    const double *beta,
    double *y,
    const std::int64_t *incy
);

void dtrmv(
    const char *uplo,
    const char *trans,
    const char *diag,
    const std::int64_t *n,
    const double *a,
    const std::int64_t *lda,
    double *x,
    const std::int64_t *incx
);

void dtbmv(
    const char *uplo,
    const char *trans,
    const char *diag,
    const std::int64_t *n,
    const std::int64_t *k,
    const double *a,
    const std::int64_t *lda,
    double *x,
    const std::int64_t *incx
);

void dtpmv(
    const char *uplo,
    const char *trans,
    const char *diag,
    const std::int64_t *n,
    const double *ap,
    double *x,
    const std::int64_t *incx
);

void dtrsv(
    const char *uplo,
    const char *trans,
    const char *diag,
    const std::int64_t *n,
    const double *a,
    const std::int64_t *lda,
    double *x,
    const std::int64_t *incx
);

void dtbsv(
    const char *uplo,
    const char *trans,
    const char *diag,
    const std::int64_t *n,
    const std::int64_t *k,
    const double *a,
    const std::int64_t *lda,
    double *x,
    const std::int64_t *incx
);

void dtpsv(
    const char *uplo,
    const char *trans,
    const char *diag,
    const std::int64_t *n,
    const double *ap,
    double *x,
    const std::int64_t *incx
);

void dger(
    const std::int64_t *m,
    const std::int64_t *n,
    const double *alpha,
    const double *x,
    const std::int64_t *incx,
    const double *y,
    const std::int64_t *incy,
    double *a,
    const std::int64_t *lda
);

void dspr(
    const char *uplo,
    const std::int64_t *n,
    const double *alpha,
    const double *x,
    const std::int64_t *incx,
    double *ap
);

void dsyr(
    const char *uplo,
    const std::int64_t *n,
    const double *alpha,
    const double *x,
    const std::int64_t *incx,
    double *a,
    const std::int64_t *lda
);

void dspr2(
    const char *uplo,
    const std::int64_t *n,
    const double *alpha,
    const double *x,
    const std::int64_t *incx,
    const double *y,
    const std::int64_t *incy,
    double *ap
);

void dsyr2(
    const char *uplo,
    const std::int64_t *n,
    const double *alpha,
    const double *x,
    const std::int64_t *incx,
    const double *y,
    const std::int64_t *incy,
    double *a,
    const std::int64_t *lda
);

void cgemv(
    const char *trans,
    const std::int64_t *m,
    const std::int64_t *n,
    const reidblas_complex_float *alpha,
    const reidblas_complex_float *a,
    const std::int64_t *lda,
    const reidblas_complex_float *x,
    const std::int64_t *incx,
    const reidblas_complex_float *beta,
    reidblas_complex_float *y,
    const std::int64_t *incy
);

void cgbmv(
    const char *trans,
    const std::int64_t *m,
    const std::int64_t *n,
    const std::int64_t *kl,
    const std::int64_t *ku,
    const reidblas_complex_float *alpha,
    const reidblas_complex_float *a,
    const std::int64_t *lda,
    const reidblas_complex_float *x,
    const std::int64_t *incx,
    const reidblas_complex_float *beta,
    reidblas_complex_float *y,
    const std::int64_t *incy
);

void chemv(
    const char *uplo,
    const std::int64_t *n,
    const reidblas_complex_float *alpha,
    const reidblas_complex_float *a,
    const std::int64_t *lda,
    const reidblas_complex_float *x,
    const std::int64_t *incx,
    const reidblas_complex_float *beta,
    reidblas_complex_float *y,
    const std::int64_t *incy
);

void chbmv(
    const char *uplo,
    const std::int64_t *n,
    const std::int64_t *k,
    const reidblas_complex_float *alpha,
    const reidblas_complex_float *a,
    const std::int64_t *lda,
    const reidblas_complex_float *x,
    const std::int64_t *incx,
    const reidblas_complex_float *beta,
    reidblas_complex_float *y,
    const std::int64_t *incy
);

void chpmv(
    const char *uplo,
    const std::int64_t *n,
    const reidblas_complex_float *alpha,
    const reidblas_complex_float *ap,
    const reidblas_complex_float *x,
    const std::int64_t *incx,
    const reidblas_complex_float *beta,
    reidblas_complex_float *y,
    const std::int64_t *incy
);

void ctrmv(
    const char *uplo,
    const char *trans,
    const char *diag,
    const std::int64_t *n,
    const reidblas_complex_float *a,
    const std::int64_t *lda,
    reidblas_complex_float *x,
    const std::int64_t *incx
);

void ctbmv(
    const char *uplo,
    const char *trans,
    const char *diag,
    const std::int64_t *n,
    const std::int64_t *k,
    const reidblas_complex_float *a,
    const std::int64_t *lda,
    reidblas_complex_float *x,
    const std::int64_t *incx
);

void ctpmv(
    const char *uplo,
    const char *trans,
    const char *diag,
    const std::int64_t *n,
    const reidblas_complex_float *ap,
    reidblas_complex_float *x,
    const std::int64_t *incx
);

void ctrsv(
    const char *uplo,
    const char *trans,
    const char *diag,
    const std::int64_t *n,
    const reidblas_complex_float *a,
    const std::int64_t *lda,
    reidblas_complex_float *x,
    const std::int64_t *incx
);

void ctbsv(
    const char *uplo,
    const char *trans,
    const char *diag,
    const std::int64_t *n,
    const std::int64_t *k,
    const reidblas_complex_float *a,
    const std::int64_t *lda,
    reidblas_complex_float *x,
    const std::int64_t *incx
);

void ctpsv(
    const char *uplo,
    const char *trans,
    const char *diag,
    const std::int64_t *n,
    const reidblas_complex_float *ap,
    reidblas_complex_float *x,
    const std::int64_t *incx
);

void cgerc(
    const std::int64_t *m,
    const std::int64_t *n,
    const reidblas_complex_float *alpha,
    const reidblas_complex_float *x,
    const std::int64_t *incx,
    const reidblas_complex_float *y,
    const std::int64_t *incy,
    reidblas_complex_float *a,
    const std::int64_t *lda
);

void cgeru(
    const std::int64_t *m,
    const std::int64_t *n,
    const reidblas_complex_float *alpha,
    const reidblas_complex_float *x,
    const std::int64_t *incx,
    const reidblas_complex_float *y,
    const std::int64_t *incy,
    reidblas_complex_float *a,
    const std::int64_t *lda
);

void cher(
    const char *uplo,
    const std::int64_t *n,
    const float *alpha,
    const reidblas_complex_float *x,
    const std::int64_t *incx,
    reidblas_complex_float *a,
    const std::int64_t *lda
);

void chpr(
    const char *uplo,
    const std::int64_t *n,
    const float *alpha,
    const reidblas_complex_float *x,
    const std::int64_t *incx,
    reidblas_complex_float *ap
);

void cher2(
    const char *uplo,
    const std::int64_t *n,
    const reidblas_complex_float *alpha,
    const reidblas_complex_float *x,
    const std::int64_t *incx,
    const reidblas_complex_float *y,
    const std::int64_t *incy,
    reidblas_complex_float *a,
    const std::int64_t *lda
);

void chpr2(
    const char *uplo,
    const std::int64_t *n,
    const reidblas_complex_float *alpha,
    const reidblas_complex_float *x,
    const std::int64_t *incx,
    const reidblas_complex_float *y,
    const std::int64_t *incy,
    reidblas_complex_float *ap
);

void zgemv(
    const char *trans,
    const std::int64_t *m,
    const std::int64_t *n,
    const reidblas_complex_double *alpha,
    const reidblas_complex_double *a,
    const std::int64_t *lda,
    const reidblas_complex_double *x,
    const std::int64_t *incx,
    const reidblas_complex_double *beta,
    reidblas_complex_double *y,
    const std::int64_t *incy
);

void zgbmv(
    const char *trans,
    const std::int64_t *m,
    const std::int64_t *n,
    const std::int64_t *kl,
    const std::int64_t *ku,
    const reidblas_complex_double *alpha,
    const reidblas_complex_double *a,
    const std::int64_t *lda,
    const reidblas_complex_double *x,
    const std::int64_t *incx,
    const reidblas_complex_double *beta,
    reidblas_complex_double *y,
    const std::int64_t *incy
);

void zhemv(
    const char *uplo,
    const std::int64_t *n,
    const reidblas_complex_double *alpha,
    const reidblas_complex_double *a,
    const std::int64_t *lda,
    const reidblas_complex_double *x,
    const std::int64_t *incx,
    const reidblas_complex_double *beta,
    reidblas_complex_double *y,
    const std::int64_t *incy
);

void zhbmv(
    const char *uplo,
    const std::int64_t *n,
    const std::int64_t *k,
    const reidblas_complex_double *alpha,
    const reidblas_complex_double *a,
    const std::int64_t *lda,
    const reidblas_complex_double *x,
    const std::int64_t *incx,
    const reidblas_complex_double *beta,
    reidblas_complex_double *y,
    const std::int64_t *incy
);

void zhpmv(
    const char *uplo,
    const std::int64_t *n,
    const reidblas_complex_double *alpha,
    const reidblas_complex_double *ap,
    const reidblas_complex_double *x,
    const std::int64_t *incx,
    const reidblas_complex_double *beta,
    reidblas_complex_double *y,
    const std::int64_t *incy
);

void ztrmv(
    const char *uplo,
    const char *trans,
    const char *diag,
    const std::int64_t *n,
    const reidblas_complex_double *a,
    const std::int64_t *lda,
    reidblas_complex_double *x,
    const std::int64_t *incx
);

void ztbmv(
    const char *uplo,
    const char *trans,
    const char *diag,
    const std::int64_t *n,
    const std::int64_t *k,
    const reidblas_complex_double *a,
    const std::int64_t *lda,
    reidblas_complex_double *x,
    const std::int64_t *incx
);

void ztpmv(
    const char *uplo,
    const char *trans,
    const char *diag,
    const std::int64_t *n,
    const reidblas_complex_double *ap,
    reidblas_complex_double *x,
    const std::int64_t *incx
);

void ztrsv(
    const char *uplo,
    const char *trans,
    const char *diag,
    const std::int64_t *n,
    const reidblas_complex_double *a,
    const std::int64_t *lda,
    reidblas_complex_double *x,
    const std::int64_t *incx
);

void ztbsv(
    const char *uplo,
    const char *trans,
    const char *diag,
    const std::int64_t *n,
    const std::int64_t *k,
    const reidblas_complex_double *a,
    const std::int64_t *lda,
    reidblas_complex_double *x,
    const std::int64_t *incx
);

void ztpsv(
    const char *uplo,
    const char *trans,
    const char *diag,
    const std::int64_t *n,
    const reidblas_complex_double *ap,
    reidblas_complex_double *x,
    const std::int64_t *incx
);

void zgerc(
    const std::int64_t *m,
    const std::int64_t *n,
    const reidblas_complex_double *alpha,
    const reidblas_complex_double *x,
    const std::int64_t *incx,
    const reidblas_complex_double *y,
    const std::int64_t *incy,
    reidblas_complex_double *a,
    const std::int64_t *lda
);

void zgeru(
    const std::int64_t *m,
    const std::int64_t *n,
    const reidblas_complex_double *alpha,
    const reidblas_complex_double *x,
    const std::int64_t *incx,
    const reidblas_complex_double *y,
    const std::int64_t *incy,
    reidblas_complex_double *a,
    const std::int64_t *lda
);

void zher(
    const char *uplo,
    const std::int64_t *n,
    const double *alpha,
    const reidblas_complex_double *x,
    const std::int64_t *incx,
    reidblas_complex_double *a,
    const std::int64_t *lda
);

void zhpr(
    const char *uplo,
    const std::int64_t *n,
    const double *alpha,
    const reidblas_complex_double *x,
    const std::int64_t *incx,
    reidblas_complex_double *ap
);

void zher2(
    const char *uplo,
    const std::int64_t *n,
    const reidblas_complex_double *alpha,
    const reidblas_complex_double *x,
    const std::int64_t *incx,
    const reidblas_complex_double *y,
    const std::int64_t *incy,
    reidblas_complex_double *a,
    const std::int64_t *lda
);

void zhpr2(
    const char *uplo,
    const std::int64_t *n,
    const reidblas_complex_double *alpha,
    const reidblas_complex_double *x,
    const std::int64_t *incx,
    const reidblas_complex_double *y,
    const std::int64_t *incy,
    reidblas_complex_double *ap
);

void sgemm(
    const char *transa,
    const char *transb,
    const std::int64_t *m,
    const std::int64_t *n,
    const std::int64_t *k,
    const float *alpha,
    const float *a,
    const std::int64_t *lda,
    const float *b,
    const std::int64_t *ldb,
    const float *beta,
    float *c,
    const std::int64_t *ldc
);

void ssymm(
    const char *side,
    const char *uplo,
    const std::int64_t *m,
    const std::int64_t *n,
    const float *alpha,
    const float *a,
    const std::int64_t *lda,
    const float *b,
    const std::int64_t *ldb,
    const float *beta,
    float *c,
    const std::int64_t *ldc
);

void ssyrk(
    const char *uplo,
    const char *trans,
    const std::int64_t *n,
    const std::int64_t *k,
    const float *alpha,
    const float *a,
    const std::int64_t *lda,
    const float *beta,
    float *c,
    const std::int64_t *ldc
);

void ssyr2k(
    const char *uplo,
    const char *trans,
    const std::int64_t *n,
    const std::int64_t *k,
    const float *alpha,
    const float *a,
    const std::int64_t *lda,
    const float *b,
    const std::int64_t *ldb,
    const float *beta,
    float *c,
    const std::int64_t *ldc
);

void strsm(
    const char *side,
    const char *uplo,
    const char *transa,
    const char *diag,
    const std::int64_t *m,
    const std::int64_t *n,
    const float *alpha,
    const float *a,
    const std::int64_t *lda,
    float *b,
    const std::int64_t *ldb
);

void strmm(
    const char *side,
    const char *uplo,
    const char *transa,
    const char *diag,
    const std::int64_t *m,
    const std::int64_t *n,
    const float *alpha,
    const float *a,
    const std::int64_t *lda,
    float *b,
    const std::int64_t *ldb
);

void dgemm(
    const char *transa,
    const char *transb,
    const std::int64_t *m,
    const std::int64_t *n,
    const std::int64_t *k,
    const double *alpha,
    const double *a,
    const std::int64_t *lda,
    const double *b,
    const std::int64_t *ldb,
    const double *beta,
    double *c,
    const std::int64_t *ldc
);

void dsymm(
    const char *side,
    const char *uplo,
    const std::int64_t *m,
    const std::int64_t *n,
    const double *alpha,
    const double *a,
    const std::int64_t *lda,
    const double *b,
    const std::int64_t *ldb,
    const double *beta,
    double *c,
    const std::int64_t *ldc
);

void dsyrk(
    const char *uplo,
    const char *trans,
    const std::int64_t *n,
    const std::int64_t *k,
    const double *alpha,
    const double *a,
    const std::int64_t *lda,
    const double *beta,
    double *c,
    const std::int64_t *ldc
);

void dsyr2k(
    const char *uplo,
    const char *trans,
    const std::int64_t *n,
    const std::int64_t *k,
    const double *alpha,
    const double *a,
    const std::int64_t *lda,
    const double *b,
    const std::int64_t *ldb,
    const double *beta,
    double *c,
    const std::int64_t *ldc
);

void dtrsm(
    const char *side,
    const char *uplo,
    const char *transa,
    const char *diag,
    const std::int64_t *m,
    const std::int64_t *n,
    const double *alpha,
    const double *a,
    const std::int64_t *lda,
    double *b,
    const std::int64_t *ldb
);

void dtrmm(
    const char *side,
    const char *uplo,
    const char *transa,
    const char *diag,
    const std::int64_t *m,
    const std::int64_t *n,
    const double *alpha,
    const double *a,
    const std::int64_t *lda,
    double *b,
    const std::int64_t *ldb
);

void cgemm(
    const char *transa,
    const char *transb,
    const std::int64_t *m,
    const std::int64_t *n,
    const std::int64_t *k,
    const reidblas_complex_float *alpha,
    const reidblas_complex_float *a,
    const std::int64_t *lda,
    const reidblas_complex_float *b,
    const std::int64_t *ldb,
    const reidblas_complex_float *beta,
    reidblas_complex_float *c,
    const std::int64_t *ldc
);

void csymm(
    const char *side,
    const char *uplo,
    const std::int64_t *m,
    const std::int64_t *n,
    const reidblas_complex_float *alpha,
    const reidblas_complex_float *a,
    const std::int64_t *lda,
    const reidblas_complex_float *b,
    const std::int64_t *ldb,
    const reidblas_complex_float *beta,
    reidblas_complex_float *c,
    const std::int64_t *ldc
);

void chemm(
    const char *side,
    const char *uplo,
    const std::int64_t *m,
    const std::int64_t *n,
    const reidblas_complex_float *alpha,
    const reidblas_complex_float *a,
    const std::int64_t *lda,
    const reidblas_complex_float *b,
    const std::int64_t *ldb,
    const reidblas_complex_float *beta,
    reidblas_complex_float *c,
    const std::int64_t *ldc
);

void csyrk(
    const char *uplo,
    const char *trans,
    const std::int64_t *n,
    const std::int64_t *k,
    const reidblas_complex_float *alpha,
    const reidblas_complex_float *a,
    const std::int64_t *lda,
    const reidblas_complex_float *beta,
    reidblas_complex_float *c,
    const std::int64_t *ldc
);

void cherk(
    const char *uplo,
    const char *trans,
    const std::int64_t *n,
    const std::int64_t *k,
    const float *alpha,
    const reidblas_complex_float *a,
    const std::int64_t *lda,
    const float *beta,
    reidblas_complex_float *c,
    const std::int64_t *ldc
);

void csyr2k(
    const char *uplo,
    const char *trans,
    const std::int64_t *n,
    const std::int64_t *k,
    const reidblas_complex_float *alpha,
    const reidblas_complex_float *a,
    const std::int64_t *lda,
    const reidblas_complex_float *b,
    const std::int64_t *ldb,
    const reidblas_complex_float *beta,
    reidblas_complex_float *c,
    const std::int64_t *ldc
);

void cher2k(
    const char *uplo,
    const char *trans,
    const std::int64_t *n,
    const std::int64_t *k,
    const reidblas_complex_float *alpha,
    const reidblas_complex_float *a,
    const std::int64_t *lda,
    const reidblas_complex_float *b,
    const std::int64_t *ldb,
    const float *beta,
    reidblas_complex_float *c,
    const std::int64_t *ldc
);

void ctrsm(
    const char *side,
    const char *uplo,
    const char *transa,
    const char *diag,
    const std::int64_t *m,
    const std::int64_t *n,
    const reidblas_complex_float *alpha,
    const reidblas_complex_float *a,
    const std::int64_t *lda,
    reidblas_complex_float *b,
    const std::int64_t *ldb
);

void ctrmm(
    const char *side,
    const char *uplo,
    const char *transa,
    const char *diag,
    const std::int64_t *m,
    const std::int64_t *n,
    const reidblas_complex_float *alpha,
    const reidblas_complex_float *a,
    const std::int64_t *lda,
    reidblas_complex_float *b,
    const std::int64_t *ldb
);

void zgemm(
    const char *transa,
    const char *transb,
    const std::int64_t *m,
    const std::int64_t *n,
    const std::int64_t *k,
    const reidblas_complex_double *alpha,
    const reidblas_complex_double *a,
    const std::int64_t *lda,
    const reidblas_complex_double *b,
    const std::int64_t *ldb,
    const reidblas_complex_double *beta,
    reidblas_complex_double *c,
    const std::int64_t *ldc
);

void zsymm(
    const char *side,
    const char *uplo,
    const std::int64_t *m,
    const std::int64_t *n,
    const reidblas_complex_double *alpha,
    const reidblas_complex_double *a,
    const std::int64_t *lda,
    const reidblas_complex_double *b,
    const std::int64_t *ldb,
    const reidblas_complex_double *beta,
    reidblas_complex_double *c,
    const std::int64_t *ldc
);

void zhemm(
    const char *side,
    const char *uplo,
    const std::int64_t *m,
    const std::int64_t *n,
    const reidblas_complex_double *alpha,
    const reidblas_complex_double *a,
    const std::int64_t *lda,
    const reidblas_complex_double *b,
    const std::int64_t *ldb,
    const reidblas_complex_double *beta,
    reidblas_complex_double *c,
    const std::int64_t *ldc
);

void zsyrk(
    const char *uplo,
    const char *trans,
    const std::int64_t *n,
    const std::int64_t *k,
    const reidblas_complex_double *alpha,
    const reidblas_complex_double *a,
    const std::int64_t *lda,
    const reidblas_complex_double *beta,
    reidblas_complex_double *c,
    const std::int64_t *ldc
);

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
);

void zsyr2k(
    const char *uplo,
    const char *trans,
    const std::int64_t *n,
    const std::int64_t *k,
    const reidblas_complex_double *alpha,
    const reidblas_complex_double *a,
    const std::int64_t *lda,
    const reidblas_complex_double *b,
    const std::int64_t *ldb,
    const reidblas_complex_double *beta,
    reidblas_complex_double *c,
    const std::int64_t *ldc
);

void zher2k(
    const char *uplo,
    const char *trans,
    const std::int64_t *n,
    const std::int64_t *k,
    const reidblas_complex_double *alpha,
    const reidblas_complex_double *a,
    const std::int64_t *lda,
    const reidblas_complex_double *b,
    const std::int64_t *ldb,
    const double *beta,
    reidblas_complex_double *c,
    const std::int64_t *ldc
);

void ztrsm(
    const char *side,
    const char *uplo,
    const char *transa,
    const char *diag,
    const std::int64_t *m,
    const std::int64_t *n,
    const reidblas_complex_double *alpha,
    const reidblas_complex_double *a,
    const std::int64_t *lda,
    reidblas_complex_double *b,
    const std::int64_t *ldb
);

void ztrmm(
    const char *side,
    const char *uplo,
    const char *transa,
    const char *diag,
    const std::int64_t *m,
    const std::int64_t *n,
    const reidblas_complex_double *alpha,
    const reidblas_complex_double *a,
    const std::int64_t *lda,
    reidblas_complex_double *b,
    const std::int64_t *ldb
);

}  // namespace reidblas::ilp64
