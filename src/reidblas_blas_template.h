#pragma once
// Primary Fortran-style BLAS interface exported by libreidblas.

#ifdef __cplusplus
extern "C" {
#endif

#ifndef REIDBLAS_INDEX_TYPE
#error "REIDBLAS_INDEX_TYPE must be defined before including reidblas_blas_template.h"
#endif

#ifndef REIDBLAS_INT_ALIAS
#error "REIDBLAS_INT_ALIAS must be defined before including reidblas_blas_template.h"
#endif

typedef REIDBLAS_INDEX_TYPE REIDBLAS_INT_ALIAS;

typedef struct { float real; float imag; } reidblas_complex_float;
typedef struct { double real; double imag; } reidblas_complex_double;

// Error handling helper
// Reports an argument error detected inside BLAS.
void xerbla_(
    const char *srname,
    const reidblas_blas_int *info
);

// Level 1 BLAS - real single precision
// Computes plane rotation parameters for single-precision scalars.
void srotg_(
    float *a,
    float *b,
    float *c,
    float *s
);
// Constructs modified Givens transformation for single-precision scalars.
void srotmg_(
    float *d1,
    float *d2,
    float *x1,
    const float *y1,
    float *param
);
// Applies a plane rotation to two single-precision vectors.
void srot_(
    const reidblas_blas_int *n,
    float *x,
    const reidblas_blas_int *incx,
    float *y,
    const reidblas_blas_int *incy,
    const float *c,
    const float *s
);
// Applies a modified Givens rotation to two single-precision vectors.
void srotm_(
    const reidblas_blas_int *n,
    float *x,
    const reidblas_blas_int *incx,
    float *y,
    const reidblas_blas_int *incy,
    const float *param
);
// Interchanges two single-precision vectors.
void sswap_(
    const reidblas_blas_int *n,
    float *x,
    const reidblas_blas_int *incx,
    float *y,
    const reidblas_blas_int *incy
);
// Scales a single-precision vector by a scalar.
void sscal_(
    const reidblas_blas_int *n,
    const float *alpha,
    float *x,
    const reidblas_blas_int *incx
);
// Copies a single-precision vector into another.
void scopy_(
    const reidblas_blas_int *n,
    const float *x,
    const reidblas_blas_int *incx,
    float *y,
    const reidblas_blas_int *incy
);
// Computes a constant times a single-precision vector plus a vector.
void saxpy_(
    const reidblas_blas_int *n,
    const float *alpha,
    const float *x,
    const reidblas_blas_int *incx,
    float *y,
    const reidblas_blas_int *incy
);
// Computes the dot product of two single-precision vectors.
float sdot_(
    const reidblas_blas_int *n,
    const float *x,
    const reidblas_blas_int *incx,
    const float *y,
    const reidblas_blas_int *incy
);
// Computes the dot product in double precision returning single precision.
float sdsdot_(
    const reidblas_blas_int *n,
    const float *sb,
    const float *x,
    const reidblas_blas_int *incx,
    const float *y,
    const reidblas_blas_int *incy
);
// Computes the Euclidean norm of a single-precision vector.
float snrm2_(
    const reidblas_blas_int *n,
    const float *x,
    const reidblas_blas_int *incx
);
// Computes the sum of absolute values of a single-precision vector.
float sasum_(
    const reidblas_blas_int *n,
    const float *x,
    const reidblas_blas_int *incx
);
// Finds the index of the element with maximum absolute value in a single-precision vector.
reidblas_blas_int isamax_(
    const reidblas_blas_int *n,
    const float *x,
    const reidblas_blas_int *incx
);

// Level 1 BLAS - real double precision
// Computes plane rotation parameters for double-precision scalars.
void drotg_(
    double *a,
    double *b,
    double *c,
    double *s
);
// Constructs modified Givens transformation for double-precision scalars.
void drotmg_(
    double *d1,
    double *d2,
    double *x1,
    const double *y1,
    double *param
);
// Applies a plane rotation to two double-precision vectors.
void drot_(
    const reidblas_blas_int *n,
    double *x,
    const reidblas_blas_int *incx,
    double *y,
    const reidblas_blas_int *incy,
    const double *c,
    const double *s
);
// Applies a modified Givens rotation to two double-precision vectors.
void drotm_(
    const reidblas_blas_int *n,
    double *x,
    const reidblas_blas_int *incx,
    double *y,
    const reidblas_blas_int *incy,
    const double *param
);
// Interchanges two double-precision vectors.
void dswap_(
    const reidblas_blas_int *n,
    double *x,
    const reidblas_blas_int *incx,
    double *y,
    const reidblas_blas_int *incy
);
// Scales a double-precision vector by a scalar.
void dscal_(
    const reidblas_blas_int *n,
    const double *alpha,
    double *x,
    const reidblas_blas_int *incx
);
// Copies a double-precision vector into another.
void dcopy_(
    const reidblas_blas_int *n,
    const double *x,
    const reidblas_blas_int *incx,
    double *y,
    const reidblas_blas_int *incy
);
// Computes a constant times a double-precision vector plus a vector.
void daxpy_(
    const reidblas_blas_int *n,
    const double *alpha,
    const double *x,
    const reidblas_blas_int *incx,
    double *y,
    const reidblas_blas_int *incy
);
// Computes the dot product of two double-precision vectors.
double ddot_(
    const reidblas_blas_int *n,
    const double *x,
    const reidblas_blas_int *incx,
    const double *y,
    const reidblas_blas_int *incy
);
// Computes the double-precision dot product of single-precision vectors.
double dsdot_(
    const reidblas_blas_int *n,
    const float *x,
    const reidblas_blas_int *incx,
    const float *y,
    const reidblas_blas_int *incy
);
// Computes the Euclidean norm of a double-precision vector.
double dnrm2_(
    const reidblas_blas_int *n,
    const double *x,
    const reidblas_blas_int *incx
);
// Computes the sum of absolute values of a double-precision vector.
double dasum_(
    const reidblas_blas_int *n,
    const double *x,
    const reidblas_blas_int *incx
);
// Finds the index of the element with maximum absolute value in a double-precision vector.
reidblas_blas_int idamax_(
    const reidblas_blas_int *n,
    const double *x,
    const reidblas_blas_int *incx
);

// Level 1 BLAS - complex single precision
// Computes plane rotation parameters for complex single-precision scalars.
void crotg_(
    reidblas_complex_float *a,
    const reidblas_complex_float *b,
    float *c,
    reidblas_complex_float *s
);
// Applies a real plane rotation to complex single-precision vectors.
void csrot_(
    const reidblas_blas_int *n,
    reidblas_complex_float *x,
    const reidblas_blas_int *incx,
    reidblas_complex_float *y,
    const reidblas_blas_int *incy,
    const float *c,
    const float *s
);
// Scales a complex single-precision vector by a real scalar.
void csscal_(
    const reidblas_blas_int *n,
    const float *alpha,
    reidblas_complex_float *x,
    const reidblas_blas_int *incx
);
// Scales a complex single-precision vector by a complex scalar.
void cscal_(
    const reidblas_blas_int *n,
    const reidblas_complex_float *alpha,
    reidblas_complex_float *x,
    const reidblas_blas_int *incx
);
// Interchanges two complex single-precision vectors.
void cswap_(
    const reidblas_blas_int *n,
    reidblas_complex_float *x,
    const reidblas_blas_int *incx,
    reidblas_complex_float *y,
    const reidblas_blas_int *incy
);
// Copies a complex single-precision vector into another.
void ccopy_(
    const reidblas_blas_int *n,
    const reidblas_complex_float *x,
    const reidblas_blas_int *incx,
    reidblas_complex_float *y,
    const reidblas_blas_int *incy
);
// Computes a constant times a complex single-precision vector plus a vector.
void caxpy_(
    const reidblas_blas_int *n,
    const reidblas_complex_float *alpha,
    const reidblas_complex_float *x,
    const reidblas_blas_int *incx,
    reidblas_complex_float *y,
    const reidblas_blas_int *incy
);
// Computes the unconjugated dot product of two complex single-precision vectors.
reidblas_complex_float cdotu_(
    const reidblas_blas_int *n,
    const reidblas_complex_float *x,
    const reidblas_blas_int *incx,
    const reidblas_complex_float *y,
    const reidblas_blas_int *incy
);
// Computes the conjugated dot product of two complex single-precision vectors.
reidblas_complex_float cdotc_(
    const reidblas_blas_int *n,
    const reidblas_complex_float *x,
    const reidblas_blas_int *incx,
    const reidblas_complex_float *y,
    const reidblas_blas_int *incy
);
// Computes the Euclidean norm of a complex single-precision vector.
float scnrm2_(
    const reidblas_blas_int *n,
    const reidblas_complex_float *x,
    const reidblas_blas_int *incx
);
// Computes the sum of absolute values of a complex single-precision vector.
float scasum_(
    const reidblas_blas_int *n,
    const reidblas_complex_float *x,
    const reidblas_blas_int *incx
);
// Finds the index of the element with maximum absolute value in a complex single-precision vector.
reidblas_blas_int icamax_(
    const reidblas_blas_int *n,
    const reidblas_complex_float *x,
    const reidblas_blas_int *incx
);

// Level 1 BLAS - complex double precision
// Computes plane rotation parameters for complex double-precision scalars.
void zrotg_(
    reidblas_complex_double *a,
    const reidblas_complex_double *b,
    double *c,
    reidblas_complex_double *s
);
// Applies a real plane rotation to complex double-precision vectors.
void zdrot_(
    const reidblas_blas_int *n,
    reidblas_complex_double *x,
    const reidblas_blas_int *incx,
    reidblas_complex_double *y,
    const reidblas_blas_int *incy,
    const double *c,
    const double *s
);
// Scales a complex double-precision vector by a real scalar.
void zdscal_(
    const reidblas_blas_int *n,
    const double *alpha,
    reidblas_complex_double *x,
    const reidblas_blas_int *incx
);
// Scales a complex double-precision vector by a complex scalar.
void zscal_(
    const reidblas_blas_int *n,
    const reidblas_complex_double *alpha,
    reidblas_complex_double *x,
    const reidblas_blas_int *incx
);
// Interchanges two complex double-precision vectors.
void zswap_(
    const reidblas_blas_int *n,
    reidblas_complex_double *x,
    const reidblas_blas_int *incx,
    reidblas_complex_double *y,
    const reidblas_blas_int *incy
);
// Copies a complex double-precision vector into another.
void zcopy_(
    const reidblas_blas_int *n,
    const reidblas_complex_double *x,
    const reidblas_blas_int *incx,
    reidblas_complex_double *y,
    const reidblas_blas_int *incy
);
// Computes a constant times a complex double-precision vector plus a vector.
void zaxpy_(
    const reidblas_blas_int *n,
    const reidblas_complex_double *alpha,
    const reidblas_complex_double *x,
    const reidblas_blas_int *incx,
    reidblas_complex_double *y,
    const reidblas_blas_int *incy
);
// Computes the unconjugated dot product of two complex double-precision vectors.
reidblas_complex_double zdotu_(
    const reidblas_blas_int *n,
    const reidblas_complex_double *x,
    const reidblas_blas_int *incx,
    const reidblas_complex_double *y,
    const reidblas_blas_int *incy
);
// Computes the conjugated dot product of two complex double-precision vectors.
reidblas_complex_double zdotc_(
    const reidblas_blas_int *n,
    const reidblas_complex_double *x,
    const reidblas_blas_int *incx,
    const reidblas_complex_double *y,
    const reidblas_blas_int *incy
);
// Computes the Euclidean norm of a complex double-precision vector.
double dznrm2_(
    const reidblas_blas_int *n,
    const reidblas_complex_double *x,
    const reidblas_blas_int *incx
);
// Computes the sum of absolute values of a complex double-precision vector.
double dzasum_(
    const reidblas_blas_int *n,
    const reidblas_complex_double *x,
    const reidblas_blas_int *incx
);
// Finds the index of the element with maximum absolute value in a complex double-precision vector.
reidblas_blas_int izamax_(
    const reidblas_blas_int *n,
    const reidblas_complex_double *x,
    const reidblas_blas_int *incx
);

// Level 2 BLAS - real single precision
// Performs general matrix-vector multiply for single precision.
void sgemv_(
    const char *trans,
    const reidblas_blas_int *m,
    const reidblas_blas_int *n,
    const float *alpha,
    const float *a,
    const reidblas_blas_int *lda,
    const float *x,
    const reidblas_blas_int *incx,
    const float *beta,
    float *y,
    const reidblas_blas_int *incy
);
// Performs general band matrix-vector multiply for single precision.
void sgbmv_(
    const char *trans,
    const reidblas_blas_int *m,
    const reidblas_blas_int *n,
    const reidblas_blas_int *kl,
    const reidblas_blas_int *ku,
    const float *alpha,
    const float *a,
    const reidblas_blas_int *lda,
    const float *x,
    const reidblas_blas_int *incx,
    const float *beta,
    float *y,
    const reidblas_blas_int *incy
);
// Performs symmetric matrix-vector multiply for single precision.
void ssymv_(
    const char *uplo,
    const reidblas_blas_int *n,
    const float *alpha,
    const float *a,
    const reidblas_blas_int *lda,
    const float *x,
    const reidblas_blas_int *incx,
    const float *beta,
    float *y,
    const reidblas_blas_int *incy
);
// Performs symmetric band matrix-vector multiply for single precision.
void ssbmv_(
    const char *uplo,
    const reidblas_blas_int *n,
    const reidblas_blas_int *k,
    const float *alpha,
    const float *a,
    const reidblas_blas_int *lda,
    const float *x,
    const reidblas_blas_int *incx,
    const float *beta,
    float *y,
    const reidblas_blas_int *incy
);
// Performs symmetric packed matrix-vector multiply for single precision.
void sspmv_(
    const char *uplo,
    const reidblas_blas_int *n,
    const float *alpha,
    const float *ap,
    const float *x,
    const reidblas_blas_int *incx,
    const float *beta,
    float *y,
    const reidblas_blas_int *incy
);
// Performs triangular matrix-vector multiply for single precision.
void strmv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const reidblas_blas_int *n,
    const float *a,
    const reidblas_blas_int *lda,
    float *x,
    const reidblas_blas_int *incx
);
// Performs triangular band matrix-vector multiply for single precision.
void stbmv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const reidblas_blas_int *n,
    const reidblas_blas_int *k,
    const float *a,
    const reidblas_blas_int *lda,
    float *x,
    const reidblas_blas_int *incx
);
// Performs triangular packed matrix-vector multiply for single precision.
void stpmv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const reidblas_blas_int *n,
    const float *ap,
    float *x,
    const reidblas_blas_int *incx
);
// Solves a triangular system with a single-precision matrix.
void strsv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const reidblas_blas_int *n,
    const float *a,
    const reidblas_blas_int *lda,
    float *x,
    const reidblas_blas_int *incx
);
// Solves a triangular band system with a single-precision matrix.
void stbsv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const reidblas_blas_int *n,
    const reidblas_blas_int *k,
    const float *a,
    const reidblas_blas_int *lda,
    float *x,
    const reidblas_blas_int *incx
);
// Solves a triangular packed system with a single-precision matrix.
void stpsv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const reidblas_blas_int *n,
    const float *ap,
    float *x,
    const reidblas_blas_int *incx
);
// Performs single-precision rank-1 update A := alpha*x*y**T + A.
void sger_(
    const reidblas_blas_int *m,
    const reidblas_blas_int *n,
    const float *alpha,
    const float *x,
    const reidblas_blas_int *incx,
    const float *y,
    const reidblas_blas_int *incy,
    float *a,
    const reidblas_blas_int *lda
);
// Performs symmetric rank-1 update A := alpha*x*x**T + A (packed storage).
void sspr_(
    const char *uplo,
    const reidblas_blas_int *n,
    const float *alpha,
    const float *x,
    const reidblas_blas_int *incx,
    float *ap
);
// Performs symmetric rank-1 update A := alpha*x*x**T + A.
void ssyr_(
    const char *uplo,
    const reidblas_blas_int *n,
    const float *alpha,
    const float *x,
    const reidblas_blas_int *incx,
    float *a,
    const reidblas_blas_int *lda
);
// Performs symmetric rank-2 update A := alpha*x*y**T + alpha*y*x**T + A (packed storage).
void sspr2_(
    const char *uplo,
    const reidblas_blas_int *n,
    const float *alpha,
    const float *x,
    const reidblas_blas_int *incx,
    const float *y,
    const reidblas_blas_int *incy,
    float *ap
);
// Performs symmetric rank-2 update A := alpha*x*y**T + alpha*y*x**T + A.
void ssyr2_(
    const char *uplo,
    const reidblas_blas_int *n,
    const float *alpha,
    const float *x,
    const reidblas_blas_int *incx,
    const float *y,
    const reidblas_blas_int *incy,
    float *a,
    const reidblas_blas_int *lda
);

// Level 2 BLAS - real double precision
// Performs general matrix-vector multiply for double precision.
void dgemv_(
    const char *trans,
    const reidblas_blas_int *m,
    const reidblas_blas_int *n,
    const double *alpha,
    const double *a,
    const reidblas_blas_int *lda,
    const double *x,
    const reidblas_blas_int *incx,
    const double *beta,
    double *y,
    const reidblas_blas_int *incy
);
// Performs general band matrix-vector multiply for double precision.
void dgbmv_(
    const char *trans,
    const reidblas_blas_int *m,
    const reidblas_blas_int *n,
    const reidblas_blas_int *kl,
    const reidblas_blas_int *ku,
    const double *alpha,
    const double *a,
    const reidblas_blas_int *lda,
    const double *x,
    const reidblas_blas_int *incx,
    const double *beta,
    double *y,
    const reidblas_blas_int *incy
);
// Performs symmetric matrix-vector multiply for double precision.
void dsymv_(
    const char *uplo,
    const reidblas_blas_int *n,
    const double *alpha,
    const double *a,
    const reidblas_blas_int *lda,
    const double *x,
    const reidblas_blas_int *incx,
    const double *beta,
    double *y,
    const reidblas_blas_int *incy
);
// Performs symmetric band matrix-vector multiply for double precision.
void dsbmv_(
    const char *uplo,
    const reidblas_blas_int *n,
    const reidblas_blas_int *k,
    const double *alpha,
    const double *a,
    const reidblas_blas_int *lda,
    const double *x,
    const reidblas_blas_int *incx,
    const double *beta,
    double *y,
    const reidblas_blas_int *incy
);
// Performs symmetric packed matrix-vector multiply for double precision.
void dspmv_(
    const char *uplo,
    const reidblas_blas_int *n,
    const double *alpha,
    const double *ap,
    const double *x,
    const reidblas_blas_int *incx,
    const double *beta,
    double *y,
    const reidblas_blas_int *incy
);
// Performs triangular matrix-vector multiply for double precision.
void dtrmv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const reidblas_blas_int *n,
    const double *a,
    const reidblas_blas_int *lda,
    double *x,
    const reidblas_blas_int *incx
);
// Performs triangular band matrix-vector multiply for double precision.
void dtbmv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const reidblas_blas_int *n,
    const reidblas_blas_int *k,
    const double *a,
    const reidblas_blas_int *lda,
    double *x,
    const reidblas_blas_int *incx
);
// Performs triangular packed matrix-vector multiply for double precision.
void dtpmv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const reidblas_blas_int *n,
    const double *ap,
    double *x,
    const reidblas_blas_int *incx
);
// Solves a triangular system with a double-precision matrix.
void dtrsv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const reidblas_blas_int *n,
    const double *a,
    const reidblas_blas_int *lda,
    double *x,
    const reidblas_blas_int *incx
);
// Solves a triangular band system with a double-precision matrix.
void dtbsv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const reidblas_blas_int *n,
    const reidblas_blas_int *k,
    const double *a,
    const reidblas_blas_int *lda,
    double *x,
    const reidblas_blas_int *incx
);
// Solves a triangular packed system with a double-precision matrix.
void dtpsv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const reidblas_blas_int *n,
    const double *ap,
    double *x,
    const reidblas_blas_int *incx
);
// Performs double-precision rank-1 update A := alpha*x*y**T + A.
void dger_(
    const reidblas_blas_int *m,
    const reidblas_blas_int *n,
    const double *alpha,
    const double *x,
    const reidblas_blas_int *incx,
    const double *y,
    const reidblas_blas_int *incy,
    double *a,
    const reidblas_blas_int *lda
);
// Performs symmetric rank-1 update A := alpha*x*x**T + A (packed storage).
void dspr_(
    const char *uplo,
    const reidblas_blas_int *n,
    const double *alpha,
    const double *x,
    const reidblas_blas_int *incx,
    double *ap
);
// Performs symmetric rank-1 update A := alpha*x*x**T + A.
void dsyr_(
    const char *uplo,
    const reidblas_blas_int *n,
    const double *alpha,
    const double *x,
    const reidblas_blas_int *incx,
    double *a,
    const reidblas_blas_int *lda
);
// Performs symmetric rank-2 update A := alpha*x*y**T + alpha*y*x**T + A (packed storage).
void dspr2_(
    const char *uplo,
    const reidblas_blas_int *n,
    const double *alpha,
    const double *x,
    const reidblas_blas_int *incx,
    const double *y,
    const reidblas_blas_int *incy,
    double *ap
);
// Performs symmetric rank-2 update A := alpha*x*y**T + alpha*y*x**T + A.
void dsyr2_(
    const char *uplo,
    const reidblas_blas_int *n,
    const double *alpha,
    const double *x,
    const reidblas_blas_int *incx,
    const double *y,
    const reidblas_blas_int *incy,
    double *a,
    const reidblas_blas_int *lda
);

// Level 2 BLAS - complex single precision
// Performs general matrix-vector multiply for complex single precision.
void cgemv_(
    const char *trans,
    const reidblas_blas_int *m,
    const reidblas_blas_int *n,
    const reidblas_complex_float *alpha,
    const reidblas_complex_float *a,
    const reidblas_blas_int *lda,
    const reidblas_complex_float *x,
    const reidblas_blas_int *incx,
    const reidblas_complex_float *beta,
    reidblas_complex_float *y,
    const reidblas_blas_int *incy
);
// Performs general band matrix-vector multiply for complex single precision.
void cgbmv_(
    const char *trans,
    const reidblas_blas_int *m,
    const reidblas_blas_int *n,
    const reidblas_blas_int *kl,
    const reidblas_blas_int *ku,
    const reidblas_complex_float *alpha,
    const reidblas_complex_float *a,
    const reidblas_blas_int *lda,
    const reidblas_complex_float *x,
    const reidblas_blas_int *incx,
    const reidblas_complex_float *beta,
    reidblas_complex_float *y,
    const reidblas_blas_int *incy
);
// Performs Hermitian matrix-vector multiply for complex single precision.
void chemv_(
    const char *uplo,
    const reidblas_blas_int *n,
    const reidblas_complex_float *alpha,
    const reidblas_complex_float *a,
    const reidblas_blas_int *lda,
    const reidblas_complex_float *x,
    const reidblas_blas_int *incx,
    const reidblas_complex_float *beta,
    reidblas_complex_float *y,
    const reidblas_blas_int *incy
);
// Performs Hermitian band matrix-vector multiply for complex single precision.
void chbmv_(
    const char *uplo,
    const reidblas_blas_int *n,
    const reidblas_blas_int *k,
    const reidblas_complex_float *alpha,
    const reidblas_complex_float *a,
    const reidblas_blas_int *lda,
    const reidblas_complex_float *x,
    const reidblas_blas_int *incx,
    const reidblas_complex_float *beta,
    reidblas_complex_float *y,
    const reidblas_blas_int *incy
);
// Performs Hermitian packed matrix-vector multiply for complex single precision.
void chpmv_(
    const char *uplo,
    const reidblas_blas_int *n,
    const reidblas_complex_float *alpha,
    const reidblas_complex_float *ap,
    const reidblas_complex_float *x,
    const reidblas_blas_int *incx,
    const reidblas_complex_float *beta,
    reidblas_complex_float *y,
    const reidblas_blas_int *incy
);
// Performs triangular matrix-vector multiply for complex single precision.
void ctrmv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const reidblas_blas_int *n,
    const reidblas_complex_float *a,
    const reidblas_blas_int *lda,
    reidblas_complex_float *x,
    const reidblas_blas_int *incx
);
// Performs triangular band matrix-vector multiply for complex single precision.
void ctbmv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const reidblas_blas_int *n,
    const reidblas_blas_int *k,
    const reidblas_complex_float *a,
    const reidblas_blas_int *lda,
    reidblas_complex_float *x,
    const reidblas_blas_int *incx
);
// Performs triangular packed matrix-vector multiply for complex single precision.
void ctpmv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const reidblas_blas_int *n,
    const reidblas_complex_float *ap,
    reidblas_complex_float *x,
    const reidblas_blas_int *incx
);
// Solves a triangular system with a complex single-precision matrix.
void ctrsv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const reidblas_blas_int *n,
    const reidblas_complex_float *a,
    const reidblas_blas_int *lda,
    reidblas_complex_float *x,
    const reidblas_blas_int *incx
);
// Solves a triangular band system with a complex single-precision matrix.
void ctbsv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const reidblas_blas_int *n,
    const reidblas_blas_int *k,
    const reidblas_complex_float *a,
    const reidblas_blas_int *lda,
    reidblas_complex_float *x,
    const reidblas_blas_int *incx
);
// Solves a triangular packed system with a complex single-precision matrix.
void ctpsv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const reidblas_blas_int *n,
    const reidblas_complex_float *ap,
    reidblas_complex_float *x,
    const reidblas_blas_int *incx
);
// Performs complex rank-1 update A := alpha*x*y**H + A.
void cgerc_(
    const reidblas_blas_int *m,
    const reidblas_blas_int *n,
    const reidblas_complex_float *alpha,
    const reidblas_complex_float *x,
    const reidblas_blas_int *incx,
    const reidblas_complex_float *y,
    const reidblas_blas_int *incy,
    reidblas_complex_float *a,
    const reidblas_blas_int *lda
);
// Performs complex rank-1 update A := alpha*x*y**T + A.
void cgeru_(
    const reidblas_blas_int *m,
    const reidblas_blas_int *n,
    const reidblas_complex_float *alpha,
    const reidblas_complex_float *x,
    const reidblas_blas_int *incx,
    const reidblas_complex_float *y,
    const reidblas_blas_int *incy,
    reidblas_complex_float *a,
    const reidblas_blas_int *lda
);
// Performs Hermitian rank-1 update A := alpha*x*x**H + A.
void cher_(
    const char *uplo,
    const reidblas_blas_int *n,
    const float *alpha,
    const reidblas_complex_float *x,
    const reidblas_blas_int *incx,
    reidblas_complex_float *a,
    const reidblas_blas_int *lda
);
// Performs Hermitian packed rank-1 update A := alpha*x*x**H + A.
void chpr_(
    const char *uplo,
    const reidblas_blas_int *n,
    const float *alpha,
    const reidblas_complex_float *x,
    const reidblas_blas_int *incx,
    reidblas_complex_float *ap
);
// Performs Hermitian rank-2 update A := alpha*x*y**H + conj(alpha)*y*x**H + A.
void cher2_(
    const char *uplo,
    const reidblas_blas_int *n,
    const reidblas_complex_float *alpha,
    const reidblas_complex_float *x,
    const reidblas_blas_int *incx,
    const reidblas_complex_float *y,
    const reidblas_blas_int *incy,
    reidblas_complex_float *a,
    const reidblas_blas_int *lda
);
// Performs Hermitian packed rank-2 update A := alpha*x*y**H + conj(alpha)*y*x**H + A.
void chpr2_(
    const char *uplo,
    const reidblas_blas_int *n,
    const reidblas_complex_float *alpha,
    const reidblas_complex_float *x,
    const reidblas_blas_int *incx,
    const reidblas_complex_float *y,
    const reidblas_blas_int *incy,
    reidblas_complex_float *ap
);

// Level 2 BLAS - complex double precision
// Performs general matrix-vector multiply for complex double precision.
void zgemv_(
    const char *trans,
    const reidblas_blas_int *m,
    const reidblas_blas_int *n,
    const reidblas_complex_double *alpha,
    const reidblas_complex_double *a,
    const reidblas_blas_int *lda,
    const reidblas_complex_double *x,
    const reidblas_blas_int *incx,
    const reidblas_complex_double *beta,
    reidblas_complex_double *y,
    const reidblas_blas_int *incy
);
// Performs general band matrix-vector multiply for complex double precision.
void zgbmv_(
    const char *trans,
    const reidblas_blas_int *m,
    const reidblas_blas_int *n,
    const reidblas_blas_int *kl,
    const reidblas_blas_int *ku,
    const reidblas_complex_double *alpha,
    const reidblas_complex_double *a,
    const reidblas_blas_int *lda,
    const reidblas_complex_double *x,
    const reidblas_blas_int *incx,
    const reidblas_complex_double *beta,
    reidblas_complex_double *y,
    const reidblas_blas_int *incy
);
// Performs Hermitian matrix-vector multiply for complex double precision.
void zhemv_(
    const char *uplo,
    const reidblas_blas_int *n,
    const reidblas_complex_double *alpha,
    const reidblas_complex_double *a,
    const reidblas_blas_int *lda,
    const reidblas_complex_double *x,
    const reidblas_blas_int *incx,
    const reidblas_complex_double *beta,
    reidblas_complex_double *y,
    const reidblas_blas_int *incy
);
// Performs Hermitian band matrix-vector multiply for complex double precision.
void zhbmv_(
    const char *uplo,
    const reidblas_blas_int *n,
    const reidblas_blas_int *k,
    const reidblas_complex_double *alpha,
    const reidblas_complex_double *a,
    const reidblas_blas_int *lda,
    const reidblas_complex_double *x,
    const reidblas_blas_int *incx,
    const reidblas_complex_double *beta,
    reidblas_complex_double *y,
    const reidblas_blas_int *incy
);
// Performs Hermitian packed matrix-vector multiply for complex double precision.
void zhpmv_(
    const char *uplo,
    const reidblas_blas_int *n,
    const reidblas_complex_double *alpha,
    const reidblas_complex_double *ap,
    const reidblas_complex_double *x,
    const reidblas_blas_int *incx,
    const reidblas_complex_double *beta,
    reidblas_complex_double *y,
    const reidblas_blas_int *incy
);
// Performs triangular matrix-vector multiply for complex double precision.
void ztrmv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const reidblas_blas_int *n,
    const reidblas_complex_double *a,
    const reidblas_blas_int *lda,
    reidblas_complex_double *x,
    const reidblas_blas_int *incx
);
// Performs triangular band matrix-vector multiply for complex double precision.
void ztbmv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const reidblas_blas_int *n,
    const reidblas_blas_int *k,
    const reidblas_complex_double *a,
    const reidblas_blas_int *lda,
    reidblas_complex_double *x,
    const reidblas_blas_int *incx
);
// Performs triangular packed matrix-vector multiply for complex double precision.
void ztpmv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const reidblas_blas_int *n,
    const reidblas_complex_double *ap,
    reidblas_complex_double *x,
    const reidblas_blas_int *incx
);
// Solves a triangular system with a complex double-precision matrix.
void ztrsv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const reidblas_blas_int *n,
    const reidblas_complex_double *a,
    const reidblas_blas_int *lda,
    reidblas_complex_double *x,
    const reidblas_blas_int *incx
);
// Solves a triangular band system with a complex double-precision matrix.
void ztbsv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const reidblas_blas_int *n,
    const reidblas_blas_int *k,
    const reidblas_complex_double *a,
    const reidblas_blas_int *lda,
    reidblas_complex_double *x,
    const reidblas_blas_int *incx
);
// Solves a triangular packed system with a complex double-precision matrix.
void ztpsv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const reidblas_blas_int *n,
    const reidblas_complex_double *ap,
    reidblas_complex_double *x,
    const reidblas_blas_int *incx
);
// Performs complex rank-1 update A := alpha*x*y**H + A.
void zgerc_(
    const reidblas_blas_int *m,
    const reidblas_blas_int *n,
    const reidblas_complex_double *alpha,
    const reidblas_complex_double *x,
    const reidblas_blas_int *incx,
    const reidblas_complex_double *y,
    const reidblas_blas_int *incy,
    reidblas_complex_double *a,
    const reidblas_blas_int *lda
);
// Performs complex rank-1 update A := alpha*x*y**T + A.
void zgeru_(
    const reidblas_blas_int *m,
    const reidblas_blas_int *n,
    const reidblas_complex_double *alpha,
    const reidblas_complex_double *x,
    const reidblas_blas_int *incx,
    const reidblas_complex_double *y,
    const reidblas_blas_int *incy,
    reidblas_complex_double *a,
    const reidblas_blas_int *lda
);
// Performs Hermitian rank-1 update A := alpha*x*x**H + A.
void zher_(
    const char *uplo,
    const reidblas_blas_int *n,
    const double *alpha,
    const reidblas_complex_double *x,
    const reidblas_blas_int *incx,
    reidblas_complex_double *a,
    const reidblas_blas_int *lda
);
// Performs Hermitian packed rank-1 update A := alpha*x*x**H + A.
void zhpr_(
    const char *uplo,
    const reidblas_blas_int *n,
    const double *alpha,
    const reidblas_complex_double *x,
    const reidblas_blas_int *incx,
    reidblas_complex_double *ap
);
// Performs Hermitian rank-2 update A := alpha*x*y**H + conj(alpha)*y*x**H + A.
void zher2_(
    const char *uplo,
    const reidblas_blas_int *n,
    const reidblas_complex_double *alpha,
    const reidblas_complex_double *x,
    const reidblas_blas_int *incx,
    const reidblas_complex_double *y,
    const reidblas_blas_int *incy,
    reidblas_complex_double *a,
    const reidblas_blas_int *lda
);
// Performs Hermitian packed rank-2 update A := alpha*x*y**H + conj(alpha)*y*x**H + A.
void zhpr2_(
    const char *uplo,
    const reidblas_blas_int *n,
    const reidblas_complex_double *alpha,
    const reidblas_complex_double *x,
    const reidblas_blas_int *incx,
    const reidblas_complex_double *y,
    const reidblas_blas_int *incy,
    reidblas_complex_double *ap
);

// Level 3 BLAS - real single precision
// Performs general matrix-matrix multiply for single precision.
void sgemm_(
    const char *transa,
    const char *transb,
    const reidblas_blas_int *m,
    const reidblas_blas_int *n,
    const reidblas_blas_int *k,
    const float *alpha,
    const float *a,
    const reidblas_blas_int *lda,
    const float *b,
    const reidblas_blas_int *ldb,
    const float *beta,
    float *c,
    const reidblas_blas_int *ldc
);
// Performs symmetric matrix-matrix multiply for single precision.
void ssymm_(
    const char *side,
    const char *uplo,
    const reidblas_blas_int *m,
    const reidblas_blas_int *n,
    const float *alpha,
    const float *a,
    const reidblas_blas_int *lda,
    const float *b,
    const reidblas_blas_int *ldb,
    const float *beta,
    float *c,
    const reidblas_blas_int *ldc
);
// Performs symmetric rank-k update C := alpha*A*A**T + beta*C or C := alpha*A**T*A + beta*C.
void ssyrk_(
    const char *uplo,
    const char *trans,
    const reidblas_blas_int *n,
    const reidblas_blas_int *k,
    const float *alpha,
    const float *a,
    const reidblas_blas_int *lda,
    const float *beta,
    float *c,
    const reidblas_blas_int *ldc
);
// Performs symmetric rank-2k update C := alpha*A*B**T + alpha*B*A**T + beta*C.
void ssyr2k_(
    const char *uplo,
    const char *trans,
    const reidblas_blas_int *n,
    const reidblas_blas_int *k,
    const float *alpha,
    const float *a,
    const reidblas_blas_int *lda,
    const float *b,
    const reidblas_blas_int *ldb,
    const float *beta,
    float *c,
    const reidblas_blas_int *ldc
);
// Solves triangular systems with multiple right-hand sides in single precision.
void strsm_(
    const char *side,
    const char *uplo,
    const char *transa,
    const char *diag,
    const reidblas_blas_int *m,
    const reidblas_blas_int *n,
    const float *alpha,
    const float *a,
    const reidblas_blas_int *lda,
    float *b,
    const reidblas_blas_int *ldb
);
// Multiplies a triangular matrix by another matrix in single precision.
void strmm_(
    const char *side,
    const char *uplo,
    const char *transa,
    const char *diag,
    const reidblas_blas_int *m,
    const reidblas_blas_int *n,
    const float *alpha,
    const float *a,
    const reidblas_blas_int *lda,
    float *b,
    const reidblas_blas_int *ldb
);

// Level 3 BLAS - real double precision
// Performs general matrix-matrix multiply for double precision.
void dgemm_(
    const char *transa,
    const char *transb,
    const reidblas_blas_int *m,
    const reidblas_blas_int *n,
    const reidblas_blas_int *k,
    const double *alpha,
    const double *a,
    const reidblas_blas_int *lda,
    const double *b,
    const reidblas_blas_int *ldb,
    const double *beta,
    double *c,
    const reidblas_blas_int *ldc
);
// Performs symmetric matrix-matrix multiply for double precision.
void dsymm_(
    const char *side,
    const char *uplo,
    const reidblas_blas_int *m,
    const reidblas_blas_int *n,
    const double *alpha,
    const double *a,
    const reidblas_blas_int *lda,
    const double *b,
    const reidblas_blas_int *ldb,
    const double *beta,
    double *c,
    const reidblas_blas_int *ldc
);
// Performs symmetric rank-k update C := alpha*A*A**T + beta*C or C := alpha*A**T*A + beta*C.
void dsyrk_(
    const char *uplo,
    const char *trans,
    const reidblas_blas_int *n,
    const reidblas_blas_int *k,
    const double *alpha,
    const double *a,
    const reidblas_blas_int *lda,
    const double *beta,
    double *c,
    const reidblas_blas_int *ldc
);
// Performs symmetric rank-2k update C := alpha*A*B**T + alpha*B*A**T + beta*C.
void dsyr2k_(
    const char *uplo,
    const char *trans,
    const reidblas_blas_int *n,
    const reidblas_blas_int *k,
    const double *alpha,
    const double *a,
    const reidblas_blas_int *lda,
    const double *b,
    const reidblas_blas_int *ldb,
    const double *beta,
    double *c,
    const reidblas_blas_int *ldc
);
// Solves triangular systems with multiple right-hand sides in double precision.
void dtrsm_(
    const char *side,
    const char *uplo,
    const char *transa,
    const char *diag,
    const reidblas_blas_int *m,
    const reidblas_blas_int *n,
    const double *alpha,
    const double *a,
    const reidblas_blas_int *lda,
    double *b,
    const reidblas_blas_int *ldb
);
// Multiplies a triangular matrix by another matrix in double precision.
void dtrmm_(
    const char *side,
    const char *uplo,
    const char *transa,
    const char *diag,
    const reidblas_blas_int *m,
    const reidblas_blas_int *n,
    const double *alpha,
    const double *a,
    const reidblas_blas_int *lda,
    double *b,
    const reidblas_blas_int *ldb
);

// Level 3 BLAS - complex single precision
// Performs general matrix-matrix multiply for complex single precision.
void cgemm_(
    const char *transa,
    const char *transb,
    const reidblas_blas_int *m,
    const reidblas_blas_int *n,
    const reidblas_blas_int *k,
    const reidblas_complex_float *alpha,
    const reidblas_complex_float *a,
    const reidblas_blas_int *lda,
    const reidblas_complex_float *b,
    const reidblas_blas_int *ldb,
    const reidblas_complex_float *beta,
    reidblas_complex_float *c,
    const reidblas_blas_int *ldc
);
// Performs symmetric matrix-matrix multiply for complex single precision.
void csymm_(
    const char *side,
    const char *uplo,
    const reidblas_blas_int *m,
    const reidblas_blas_int *n,
    const reidblas_complex_float *alpha,
    const reidblas_complex_float *a,
    const reidblas_blas_int *lda,
    const reidblas_complex_float *b,
    const reidblas_blas_int *ldb,
    const reidblas_complex_float *beta,
    reidblas_complex_float *c,
    const reidblas_blas_int *ldc
);
// Performs Hermitian matrix-matrix multiply for complex single precision.
void chemm_(
    const char *side,
    const char *uplo,
    const reidblas_blas_int *m,
    const reidblas_blas_int *n,
    const reidblas_complex_float *alpha,
    const reidblas_complex_float *a,
    const reidblas_blas_int *lda,
    const reidblas_complex_float *b,
    const reidblas_blas_int *ldb,
    const reidblas_complex_float *beta,
    reidblas_complex_float *c,
    const reidblas_blas_int *ldc
);
// Performs symmetric rank-k update for complex single precision.
void csyrk_(
    const char *uplo,
    const char *trans,
    const reidblas_blas_int *n,
    const reidblas_blas_int *k,
    const reidblas_complex_float *alpha,
    const reidblas_complex_float *a,
    const reidblas_blas_int *lda,
    const reidblas_complex_float *beta,
    reidblas_complex_float *c,
    const reidblas_blas_int *ldc
);
// Performs Hermitian rank-k update for complex single precision.
void cherk_(
    const char *uplo,
    const char *trans,
    const reidblas_blas_int *n,
    const reidblas_blas_int *k,
    const float *alpha,
    const reidblas_complex_float *a,
    const reidblas_blas_int *lda,
    const float *beta,
    reidblas_complex_float *c,
    const reidblas_blas_int *ldc
);
// Performs symmetric rank-2k update for complex single precision.
void csyr2k_(
    const char *uplo,
    const char *trans,
    const reidblas_blas_int *n,
    const reidblas_blas_int *k,
    const reidblas_complex_float *alpha,
    const reidblas_complex_float *a,
    const reidblas_blas_int *lda,
    const reidblas_complex_float *b,
    const reidblas_blas_int *ldb,
    const reidblas_complex_float *beta,
    reidblas_complex_float *c,
    const reidblas_blas_int *ldc
);
// Performs Hermitian rank-2k update for complex single precision.
void cher2k_(
    const char *uplo,
    const char *trans,
    const reidblas_blas_int *n,
    const reidblas_blas_int *k,
    const reidblas_complex_float *alpha,
    const reidblas_complex_float *a,
    const reidblas_blas_int *lda,
    const reidblas_complex_float *b,
    const reidblas_blas_int *ldb,
    const float *beta,
    reidblas_complex_float *c,
    const reidblas_blas_int *ldc
);
// Solves triangular systems with multiple right-hand sides in complex single precision.
void ctrsm_(
    const char *side,
    const char *uplo,
    const char *transa,
    const char *diag,
    const reidblas_blas_int *m,
    const reidblas_blas_int *n,
    const reidblas_complex_float *alpha,
    const reidblas_complex_float *a,
    const reidblas_blas_int *lda,
    reidblas_complex_float *b,
    const reidblas_blas_int *ldb
);
// Multiplies a triangular matrix by another matrix in complex single precision.
void ctrmm_(
    const char *side,
    const char *uplo,
    const char *transa,
    const char *diag,
    const reidblas_blas_int *m,
    const reidblas_blas_int *n,
    const reidblas_complex_float *alpha,
    const reidblas_complex_float *a,
    const reidblas_blas_int *lda,
    reidblas_complex_float *b,
    const reidblas_blas_int *ldb
);

// Level 3 BLAS - complex double precision
// Performs general matrix-matrix multiply for complex double precision.
void zgemm_(
    const char *transa,
    const char *transb,
    const reidblas_blas_int *m,
    const reidblas_blas_int *n,
    const reidblas_blas_int *k,
    const reidblas_complex_double *alpha,
    const reidblas_complex_double *a,
    const reidblas_blas_int *lda,
    const reidblas_complex_double *b,
    const reidblas_blas_int *ldb,
    const reidblas_complex_double *beta,
    reidblas_complex_double *c,
    const reidblas_blas_int *ldc
);
// Performs symmetric matrix-matrix multiply for complex double precision.
void zsymm_(
    const char *side,
    const char *uplo,
    const reidblas_blas_int *m,
    const reidblas_blas_int *n,
    const reidblas_complex_double *alpha,
    const reidblas_complex_double *a,
    const reidblas_blas_int *lda,
    const reidblas_complex_double *b,
    const reidblas_blas_int *ldb,
    const reidblas_complex_double *beta,
    reidblas_complex_double *c,
    const reidblas_blas_int *ldc
);
// Performs Hermitian matrix-matrix multiply for complex double precision.
void zhemm_(
    const char *side,
    const char *uplo,
    const reidblas_blas_int *m,
    const reidblas_blas_int *n,
    const reidblas_complex_double *alpha,
    const reidblas_complex_double *a,
    const reidblas_blas_int *lda,
    const reidblas_complex_double *b,
    const reidblas_blas_int *ldb,
    const reidblas_complex_double *beta,
    reidblas_complex_double *c,
    const reidblas_blas_int *ldc
);
// Performs symmetric rank-k update for complex double precision.
void zsyrk_(
    const char *uplo,
    const char *trans,
    const reidblas_blas_int *n,
    const reidblas_blas_int *k,
    const reidblas_complex_double *alpha,
    const reidblas_complex_double *a,
    const reidblas_blas_int *lda,
    const reidblas_complex_double *beta,
    reidblas_complex_double *c,
    const reidblas_blas_int *ldc
);
// Performs Hermitian rank-k update for complex double precision.
void zherk_(
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
);
// Performs symmetric rank-2k update for complex double precision.
void zsyr2k_(
    const char *uplo,
    const char *trans,
    const reidblas_blas_int *n,
    const reidblas_blas_int *k,
    const reidblas_complex_double *alpha,
    const reidblas_complex_double *a,
    const reidblas_blas_int *lda,
    const reidblas_complex_double *b,
    const reidblas_blas_int *ldb,
    const reidblas_complex_double *beta,
    reidblas_complex_double *c,
    const reidblas_blas_int *ldc
);
// Performs Hermitian rank-2k update for complex double precision.
void zher2k_(
    const char *uplo,
    const char *trans,
    const reidblas_blas_int *n,
    const reidblas_blas_int *k,
    const reidblas_complex_double *alpha,
    const reidblas_complex_double *a,
    const reidblas_blas_int *lda,
    const reidblas_complex_double *b,
    const reidblas_blas_int *ldb,
    const double *beta,
    reidblas_complex_double *c,
    const reidblas_blas_int *ldc
);
// Solves triangular systems with multiple right-hand sides in complex double precision.
void ztrsm_(
    const char *side,
    const char *uplo,
    const char *transa,
    const char *diag,
    const reidblas_blas_int *m,
    const reidblas_blas_int *n,
    const reidblas_complex_double *alpha,
    const reidblas_complex_double *a,
    const reidblas_blas_int *lda,
    reidblas_complex_double *b,
    const reidblas_blas_int *ldb
);
// Multiplies a triangular matrix by another matrix in complex double precision.
void ztrmm_(
    const char *side,
    const char *uplo,
    const char *transa,
    const char *diag,
    const reidblas_blas_int *m,
    const reidblas_blas_int *n,
    const reidblas_complex_double *alpha,
    const reidblas_complex_double *a,
    const reidblas_blas_int *lda,
    reidblas_complex_double *b,
    const reidblas_blas_int *ldb
);

#ifdef __cplusplus
}
#endif
