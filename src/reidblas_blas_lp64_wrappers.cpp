#include "reidblas_blas_lp64.h"
#include "reidblas_blas_ilp64_internal.hpp"

#include <cstdint>

extern "C" void srotg_(
    float *a,
    float *b,
    float *c,
    float *s
) {
    reidblas::ilp64::srotg(a, b, c, s);
}

extern "C" void srotmg_(
    float *d1,
    float *d2,
    float *x1,
    const float *y1,
    float *param
) {
    reidblas::ilp64::srotmg(d1, d2, x1, y1, param);
}

extern "C" void srot_(
    const reidblas_blas_int *n,
    float *x,
    const reidblas_blas_int *incx,
    float *y,
    const reidblas_blas_int *incy,
    const float *c,
    const float *s
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }
    std::int64_t incy_value = 0;
    const std::int64_t *incy_ilp64 = nullptr;
    if (incy != nullptr) {
        incy_value = static_cast<std::int64_t>(*incy);
        incy_ilp64 = &incy_value;
    }

    reidblas::ilp64::srot(n_ilp64, x, incx_ilp64, y, incy_ilp64, c, s);
}

extern "C" void srotm_(
    const reidblas_blas_int *n,
    float *x,
    const reidblas_blas_int *incx,
    float *y,
    const reidblas_blas_int *incy,
    const float *param
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }
    std::int64_t incy_value = 0;
    const std::int64_t *incy_ilp64 = nullptr;
    if (incy != nullptr) {
        incy_value = static_cast<std::int64_t>(*incy);
        incy_ilp64 = &incy_value;
    }

    reidblas::ilp64::srotm(n_ilp64, x, incx_ilp64, y, incy_ilp64, param);
}

extern "C" void sswap_(
    const reidblas_blas_int *n,
    float *x,
    const reidblas_blas_int *incx,
    float *y,
    const reidblas_blas_int *incy
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }
    std::int64_t incy_value = 0;
    const std::int64_t *incy_ilp64 = nullptr;
    if (incy != nullptr) {
        incy_value = static_cast<std::int64_t>(*incy);
        incy_ilp64 = &incy_value;
    }

    reidblas::ilp64::sswap(n_ilp64, x, incx_ilp64, y, incy_ilp64);
}

extern "C" void sscal_(
    const reidblas_blas_int *n,
    const float *alpha,
    float *x,
    const reidblas_blas_int *incx
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }

    reidblas::ilp64::sscal(n_ilp64, alpha, x, incx_ilp64);
}

extern "C" void scopy_(
    const reidblas_blas_int *n,
    const float *x,
    const reidblas_blas_int *incx,
    float *y,
    const reidblas_blas_int *incy
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }
    std::int64_t incy_value = 0;
    const std::int64_t *incy_ilp64 = nullptr;
    if (incy != nullptr) {
        incy_value = static_cast<std::int64_t>(*incy);
        incy_ilp64 = &incy_value;
    }

    reidblas::ilp64::scopy(n_ilp64, x, incx_ilp64, y, incy_ilp64);
}

extern "C" void saxpy_(
    const reidblas_blas_int *n,
    const float *alpha,
    const float *x,
    const reidblas_blas_int *incx,
    float *y,
    const reidblas_blas_int *incy
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }
    std::int64_t incy_value = 0;
    const std::int64_t *incy_ilp64 = nullptr;
    if (incy != nullptr) {
        incy_value = static_cast<std::int64_t>(*incy);
        incy_ilp64 = &incy_value;
    }

    reidblas::ilp64::saxpy(n_ilp64, alpha, x, incx_ilp64, y, incy_ilp64);
}

extern "C" float sdot_(
    const reidblas_blas_int *n,
    const float *x,
    const reidblas_blas_int *incx,
    const float *y,
    const reidblas_blas_int *incy
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }
    std::int64_t incy_value = 0;
    const std::int64_t *incy_ilp64 = nullptr;
    if (incy != nullptr) {
        incy_value = static_cast<std::int64_t>(*incy);
        incy_ilp64 = &incy_value;
    }

    return reidblas::ilp64::sdot(n_ilp64, x, incx_ilp64, y, incy_ilp64);
}

extern "C" float sdsdot_(
    const reidblas_blas_int *n,
    const float *sb,
    const float *x,
    const reidblas_blas_int *incx,
    const float *y,
    const reidblas_blas_int *incy
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }
    std::int64_t incy_value = 0;
    const std::int64_t *incy_ilp64 = nullptr;
    if (incy != nullptr) {
        incy_value = static_cast<std::int64_t>(*incy);
        incy_ilp64 = &incy_value;
    }

    return reidblas::ilp64::sdsdot(n_ilp64, sb, x, incx_ilp64, y, incy_ilp64);
}

extern "C" float snrm2_(
    const reidblas_blas_int *n,
    const float *x,
    const reidblas_blas_int *incx
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }

    return reidblas::ilp64::snrm2(n_ilp64, x, incx_ilp64);
}

extern "C" float sasum_(
    const reidblas_blas_int *n,
    const float *x,
    const reidblas_blas_int *incx
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }

    return reidblas::ilp64::sasum(n_ilp64, x, incx_ilp64);
}

extern "C" reidblas_blas_int isamax_(
    const reidblas_blas_int *n,
    const float *x,
    const reidblas_blas_int *incx
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }

    auto result = reidblas::ilp64::isamax(n_ilp64, x, incx_ilp64);
    return static_cast<reidblas_blas_int>(result);
}

extern "C" void drotg_(
    double *a,
    double *b,
    double *c,
    double *s
) {
    reidblas::ilp64::drotg(a, b, c, s);
}

extern "C" void drotmg_(
    double *d1,
    double *d2,
    double *x1,
    const double *y1,
    double *param
) {
    reidblas::ilp64::drotmg(d1, d2, x1, y1, param);
}

extern "C" void drot_(
    const reidblas_blas_int *n,
    double *x,
    const reidblas_blas_int *incx,
    double *y,
    const reidblas_blas_int *incy,
    const double *c,
    const double *s
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }
    std::int64_t incy_value = 0;
    const std::int64_t *incy_ilp64 = nullptr;
    if (incy != nullptr) {
        incy_value = static_cast<std::int64_t>(*incy);
        incy_ilp64 = &incy_value;
    }

    reidblas::ilp64::drot(n_ilp64, x, incx_ilp64, y, incy_ilp64, c, s);
}

extern "C" void drotm_(
    const reidblas_blas_int *n,
    double *x,
    const reidblas_blas_int *incx,
    double *y,
    const reidblas_blas_int *incy,
    const double *param
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }
    std::int64_t incy_value = 0;
    const std::int64_t *incy_ilp64 = nullptr;
    if (incy != nullptr) {
        incy_value = static_cast<std::int64_t>(*incy);
        incy_ilp64 = &incy_value;
    }

    reidblas::ilp64::drotm(n_ilp64, x, incx_ilp64, y, incy_ilp64, param);
}

extern "C" void dswap_(
    const reidblas_blas_int *n,
    double *x,
    const reidblas_blas_int *incx,
    double *y,
    const reidblas_blas_int *incy
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }
    std::int64_t incy_value = 0;
    const std::int64_t *incy_ilp64 = nullptr;
    if (incy != nullptr) {
        incy_value = static_cast<std::int64_t>(*incy);
        incy_ilp64 = &incy_value;
    }

    reidblas::ilp64::dswap(n_ilp64, x, incx_ilp64, y, incy_ilp64);
}

extern "C" void dscal_(
    const reidblas_blas_int *n,
    const double *alpha,
    double *x,
    const reidblas_blas_int *incx
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }

    reidblas::ilp64::dscal(n_ilp64, alpha, x, incx_ilp64);
}

extern "C" void dcopy_(
    const reidblas_blas_int *n,
    const double *x,
    const reidblas_blas_int *incx,
    double *y,
    const reidblas_blas_int *incy
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }
    std::int64_t incy_value = 0;
    const std::int64_t *incy_ilp64 = nullptr;
    if (incy != nullptr) {
        incy_value = static_cast<std::int64_t>(*incy);
        incy_ilp64 = &incy_value;
    }

    reidblas::ilp64::dcopy(n_ilp64, x, incx_ilp64, y, incy_ilp64);
}

extern "C" void daxpy_(
    const reidblas_blas_int *n,
    const double *alpha,
    const double *x,
    const reidblas_blas_int *incx,
    double *y,
    const reidblas_blas_int *incy
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }
    std::int64_t incy_value = 0;
    const std::int64_t *incy_ilp64 = nullptr;
    if (incy != nullptr) {
        incy_value = static_cast<std::int64_t>(*incy);
        incy_ilp64 = &incy_value;
    }

    reidblas::ilp64::daxpy(n_ilp64, alpha, x, incx_ilp64, y, incy_ilp64);
}

extern "C" double ddot_(
    const reidblas_blas_int *n,
    const double *x,
    const reidblas_blas_int *incx,
    const double *y,
    const reidblas_blas_int *incy
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }
    std::int64_t incy_value = 0;
    const std::int64_t *incy_ilp64 = nullptr;
    if (incy != nullptr) {
        incy_value = static_cast<std::int64_t>(*incy);
        incy_ilp64 = &incy_value;
    }

    return reidblas::ilp64::ddot(n_ilp64, x, incx_ilp64, y, incy_ilp64);
}

extern "C" double dsdot_(
    const reidblas_blas_int *n,
    const float *x,
    const reidblas_blas_int *incx,
    const float *y,
    const reidblas_blas_int *incy
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }
    std::int64_t incy_value = 0;
    const std::int64_t *incy_ilp64 = nullptr;
    if (incy != nullptr) {
        incy_value = static_cast<std::int64_t>(*incy);
        incy_ilp64 = &incy_value;
    }

    return reidblas::ilp64::dsdot(n_ilp64, x, incx_ilp64, y, incy_ilp64);
}

extern "C" double dnrm2_(
    const reidblas_blas_int *n,
    const double *x,
    const reidblas_blas_int *incx
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }

    return reidblas::ilp64::dnrm2(n_ilp64, x, incx_ilp64);
}

extern "C" double dasum_(
    const reidblas_blas_int *n,
    const double *x,
    const reidblas_blas_int *incx
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }

    return reidblas::ilp64::dasum(n_ilp64, x, incx_ilp64);
}

extern "C" reidblas_blas_int idamax_(
    const reidblas_blas_int *n,
    const double *x,
    const reidblas_blas_int *incx
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }

    auto result = reidblas::ilp64::idamax(n_ilp64, x, incx_ilp64);
    return static_cast<reidblas_blas_int>(result);
}

extern "C" void crotg_(
    reidblas_complex_float *a,
    const reidblas_complex_float *b,
    float *c,
    reidblas_complex_float *s
) {
    reidblas::ilp64::crotg(a, b, c, s);
}

extern "C" void csrot_(
    const reidblas_blas_int *n,
    reidblas_complex_float *x,
    const reidblas_blas_int *incx,
    reidblas_complex_float *y,
    const reidblas_blas_int *incy,
    const float *c,
    const float *s
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }
    std::int64_t incy_value = 0;
    const std::int64_t *incy_ilp64 = nullptr;
    if (incy != nullptr) {
        incy_value = static_cast<std::int64_t>(*incy);
        incy_ilp64 = &incy_value;
    }

    reidblas::ilp64::csrot(n_ilp64, x, incx_ilp64, y, incy_ilp64, c, s);
}

extern "C" void csscal_(
    const reidblas_blas_int *n,
    const float *alpha,
    reidblas_complex_float *x,
    const reidblas_blas_int *incx
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }

    reidblas::ilp64::csscal(n_ilp64, alpha, x, incx_ilp64);
}

extern "C" void cscal_(
    const reidblas_blas_int *n,
    const reidblas_complex_float *alpha,
    reidblas_complex_float *x,
    const reidblas_blas_int *incx
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }

    reidblas::ilp64::cscal(n_ilp64, alpha, x, incx_ilp64);
}

extern "C" void cswap_(
    const reidblas_blas_int *n,
    reidblas_complex_float *x,
    const reidblas_blas_int *incx,
    reidblas_complex_float *y,
    const reidblas_blas_int *incy
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }
    std::int64_t incy_value = 0;
    const std::int64_t *incy_ilp64 = nullptr;
    if (incy != nullptr) {
        incy_value = static_cast<std::int64_t>(*incy);
        incy_ilp64 = &incy_value;
    }

    reidblas::ilp64::cswap(n_ilp64, x, incx_ilp64, y, incy_ilp64);
}

extern "C" void ccopy_(
    const reidblas_blas_int *n,
    const reidblas_complex_float *x,
    const reidblas_blas_int *incx,
    reidblas_complex_float *y,
    const reidblas_blas_int *incy
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }
    std::int64_t incy_value = 0;
    const std::int64_t *incy_ilp64 = nullptr;
    if (incy != nullptr) {
        incy_value = static_cast<std::int64_t>(*incy);
        incy_ilp64 = &incy_value;
    }

    reidblas::ilp64::ccopy(n_ilp64, x, incx_ilp64, y, incy_ilp64);
}

extern "C" void caxpy_(
    const reidblas_blas_int *n,
    const reidblas_complex_float *alpha,
    const reidblas_complex_float *x,
    const reidblas_blas_int *incx,
    reidblas_complex_float *y,
    const reidblas_blas_int *incy
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }
    std::int64_t incy_value = 0;
    const std::int64_t *incy_ilp64 = nullptr;
    if (incy != nullptr) {
        incy_value = static_cast<std::int64_t>(*incy);
        incy_ilp64 = &incy_value;
    }

    reidblas::ilp64::caxpy(n_ilp64, alpha, x, incx_ilp64, y, incy_ilp64);
}

extern "C" reidblas_complex_float cdotu_(
    const reidblas_blas_int *n,
    const reidblas_complex_float *x,
    const reidblas_blas_int *incx,
    const reidblas_complex_float *y,
    const reidblas_blas_int *incy
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }
    std::int64_t incy_value = 0;
    const std::int64_t *incy_ilp64 = nullptr;
    if (incy != nullptr) {
        incy_value = static_cast<std::int64_t>(*incy);
        incy_ilp64 = &incy_value;
    }

    return reidblas::ilp64::cdotu(n_ilp64, x, incx_ilp64, y, incy_ilp64);
}

extern "C" reidblas_complex_float cdotc_(
    const reidblas_blas_int *n,
    const reidblas_complex_float *x,
    const reidblas_blas_int *incx,
    const reidblas_complex_float *y,
    const reidblas_blas_int *incy
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }
    std::int64_t incy_value = 0;
    const std::int64_t *incy_ilp64 = nullptr;
    if (incy != nullptr) {
        incy_value = static_cast<std::int64_t>(*incy);
        incy_ilp64 = &incy_value;
    }

    return reidblas::ilp64::cdotc(n_ilp64, x, incx_ilp64, y, incy_ilp64);
}

extern "C" float scnrm2_(
    const reidblas_blas_int *n,
    const reidblas_complex_float *x,
    const reidblas_blas_int *incx
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }

    return reidblas::ilp64::scnrm2(n_ilp64, x, incx_ilp64);
}

extern "C" float scasum_(
    const reidblas_blas_int *n,
    const reidblas_complex_float *x,
    const reidblas_blas_int *incx
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }

    return reidblas::ilp64::scasum(n_ilp64, x, incx_ilp64);
}

extern "C" reidblas_blas_int icamax_(
    const reidblas_blas_int *n,
    const reidblas_complex_float *x,
    const reidblas_blas_int *incx
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }

    auto result = reidblas::ilp64::icamax(n_ilp64, x, incx_ilp64);
    return static_cast<reidblas_blas_int>(result);
}

extern "C" void zrotg_(
    reidblas_complex_double *a,
    const reidblas_complex_double *b,
    double *c,
    reidblas_complex_double *s
) {
    reidblas::ilp64::zrotg(a, b, c, s);
}

extern "C" void zdrot_(
    const reidblas_blas_int *n,
    reidblas_complex_double *x,
    const reidblas_blas_int *incx,
    reidblas_complex_double *y,
    const reidblas_blas_int *incy,
    const double *c,
    const double *s
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }
    std::int64_t incy_value = 0;
    const std::int64_t *incy_ilp64 = nullptr;
    if (incy != nullptr) {
        incy_value = static_cast<std::int64_t>(*incy);
        incy_ilp64 = &incy_value;
    }

    reidblas::ilp64::zdrot(n_ilp64, x, incx_ilp64, y, incy_ilp64, c, s);
}

extern "C" void zdscal_(
    const reidblas_blas_int *n,
    const double *alpha,
    reidblas_complex_double *x,
    const reidblas_blas_int *incx
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }

    reidblas::ilp64::zdscal(n_ilp64, alpha, x, incx_ilp64);
}

extern "C" void zscal_(
    const reidblas_blas_int *n,
    const reidblas_complex_double *alpha,
    reidblas_complex_double *x,
    const reidblas_blas_int *incx
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }

    reidblas::ilp64::zscal(n_ilp64, alpha, x, incx_ilp64);
}

extern "C" void zswap_(
    const reidblas_blas_int *n,
    reidblas_complex_double *x,
    const reidblas_blas_int *incx,
    reidblas_complex_double *y,
    const reidblas_blas_int *incy
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }
    std::int64_t incy_value = 0;
    const std::int64_t *incy_ilp64 = nullptr;
    if (incy != nullptr) {
        incy_value = static_cast<std::int64_t>(*incy);
        incy_ilp64 = &incy_value;
    }

    reidblas::ilp64::zswap(n_ilp64, x, incx_ilp64, y, incy_ilp64);
}

extern "C" void zcopy_(
    const reidblas_blas_int *n,
    const reidblas_complex_double *x,
    const reidblas_blas_int *incx,
    reidblas_complex_double *y,
    const reidblas_blas_int *incy
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }
    std::int64_t incy_value = 0;
    const std::int64_t *incy_ilp64 = nullptr;
    if (incy != nullptr) {
        incy_value = static_cast<std::int64_t>(*incy);
        incy_ilp64 = &incy_value;
    }

    reidblas::ilp64::zcopy(n_ilp64, x, incx_ilp64, y, incy_ilp64);
}

extern "C" void zaxpy_(
    const reidblas_blas_int *n,
    const reidblas_complex_double *alpha,
    const reidblas_complex_double *x,
    const reidblas_blas_int *incx,
    reidblas_complex_double *y,
    const reidblas_blas_int *incy
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }
    std::int64_t incy_value = 0;
    const std::int64_t *incy_ilp64 = nullptr;
    if (incy != nullptr) {
        incy_value = static_cast<std::int64_t>(*incy);
        incy_ilp64 = &incy_value;
    }

    reidblas::ilp64::zaxpy(n_ilp64, alpha, x, incx_ilp64, y, incy_ilp64);
}

extern "C" reidblas_complex_double zdotu_(
    const reidblas_blas_int *n,
    const reidblas_complex_double *x,
    const reidblas_blas_int *incx,
    const reidblas_complex_double *y,
    const reidblas_blas_int *incy
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }
    std::int64_t incy_value = 0;
    const std::int64_t *incy_ilp64 = nullptr;
    if (incy != nullptr) {
        incy_value = static_cast<std::int64_t>(*incy);
        incy_ilp64 = &incy_value;
    }

    return reidblas::ilp64::zdotu(n_ilp64, x, incx_ilp64, y, incy_ilp64);
}

extern "C" reidblas_complex_double zdotc_(
    const reidblas_blas_int *n,
    const reidblas_complex_double *x,
    const reidblas_blas_int *incx,
    const reidblas_complex_double *y,
    const reidblas_blas_int *incy
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }
    std::int64_t incy_value = 0;
    const std::int64_t *incy_ilp64 = nullptr;
    if (incy != nullptr) {
        incy_value = static_cast<std::int64_t>(*incy);
        incy_ilp64 = &incy_value;
    }

    return reidblas::ilp64::zdotc(n_ilp64, x, incx_ilp64, y, incy_ilp64);
}

extern "C" double dznrm2_(
    const reidblas_blas_int *n,
    const reidblas_complex_double *x,
    const reidblas_blas_int *incx
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }

    return reidblas::ilp64::dznrm2(n_ilp64, x, incx_ilp64);
}

extern "C" double dzasum_(
    const reidblas_blas_int *n,
    const reidblas_complex_double *x,
    const reidblas_blas_int *incx
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }

    return reidblas::ilp64::dzasum(n_ilp64, x, incx_ilp64);
}

extern "C" reidblas_blas_int izamax_(
    const reidblas_blas_int *n,
    const reidblas_complex_double *x,
    const reidblas_blas_int *incx
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }

    auto result = reidblas::ilp64::izamax(n_ilp64, x, incx_ilp64);
    return static_cast<reidblas_blas_int>(result);
}

extern "C" void sgemv_(
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
) {
    std::int64_t m_value = 0;
    const std::int64_t *m_ilp64 = nullptr;
    if (m != nullptr) {
        m_value = static_cast<std::int64_t>(*m);
        m_ilp64 = &m_value;
    }
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t lda_value = 0;
    const std::int64_t *lda_ilp64 = nullptr;
    if (lda != nullptr) {
        lda_value = static_cast<std::int64_t>(*lda);
        lda_ilp64 = &lda_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }
    std::int64_t incy_value = 0;
    const std::int64_t *incy_ilp64 = nullptr;
    if (incy != nullptr) {
        incy_value = static_cast<std::int64_t>(*incy);
        incy_ilp64 = &incy_value;
    }

    reidblas::ilp64::sgemv(trans, m_ilp64, n_ilp64, alpha, a, lda_ilp64, x, incx_ilp64, beta, y, incy_ilp64);
}

extern "C" void sgbmv_(
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
) {
    std::int64_t m_value = 0;
    const std::int64_t *m_ilp64 = nullptr;
    if (m != nullptr) {
        m_value = static_cast<std::int64_t>(*m);
        m_ilp64 = &m_value;
    }
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t kl_value = 0;
    const std::int64_t *kl_ilp64 = nullptr;
    if (kl != nullptr) {
        kl_value = static_cast<std::int64_t>(*kl);
        kl_ilp64 = &kl_value;
    }
    std::int64_t ku_value = 0;
    const std::int64_t *ku_ilp64 = nullptr;
    if (ku != nullptr) {
        ku_value = static_cast<std::int64_t>(*ku);
        ku_ilp64 = &ku_value;
    }
    std::int64_t lda_value = 0;
    const std::int64_t *lda_ilp64 = nullptr;
    if (lda != nullptr) {
        lda_value = static_cast<std::int64_t>(*lda);
        lda_ilp64 = &lda_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }
    std::int64_t incy_value = 0;
    const std::int64_t *incy_ilp64 = nullptr;
    if (incy != nullptr) {
        incy_value = static_cast<std::int64_t>(*incy);
        incy_ilp64 = &incy_value;
    }

    reidblas::ilp64::sgbmv(trans, m_ilp64, n_ilp64, kl_ilp64, ku_ilp64, alpha, a, lda_ilp64, x, incx_ilp64, beta, y, incy_ilp64);
}

extern "C" void ssymv_(
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
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t lda_value = 0;
    const std::int64_t *lda_ilp64 = nullptr;
    if (lda != nullptr) {
        lda_value = static_cast<std::int64_t>(*lda);
        lda_ilp64 = &lda_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }
    std::int64_t incy_value = 0;
    const std::int64_t *incy_ilp64 = nullptr;
    if (incy != nullptr) {
        incy_value = static_cast<std::int64_t>(*incy);
        incy_ilp64 = &incy_value;
    }

    reidblas::ilp64::ssymv(uplo, n_ilp64, alpha, a, lda_ilp64, x, incx_ilp64, beta, y, incy_ilp64);
}

extern "C" void ssbmv_(
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
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t k_value = 0;
    const std::int64_t *k_ilp64 = nullptr;
    if (k != nullptr) {
        k_value = static_cast<std::int64_t>(*k);
        k_ilp64 = &k_value;
    }
    std::int64_t lda_value = 0;
    const std::int64_t *lda_ilp64 = nullptr;
    if (lda != nullptr) {
        lda_value = static_cast<std::int64_t>(*lda);
        lda_ilp64 = &lda_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }
    std::int64_t incy_value = 0;
    const std::int64_t *incy_ilp64 = nullptr;
    if (incy != nullptr) {
        incy_value = static_cast<std::int64_t>(*incy);
        incy_ilp64 = &incy_value;
    }

    reidblas::ilp64::ssbmv(uplo, n_ilp64, k_ilp64, alpha, a, lda_ilp64, x, incx_ilp64, beta, y, incy_ilp64);
}

extern "C" void sspmv_(
    const char *uplo,
    const reidblas_blas_int *n,
    const float *alpha,
    const float *ap,
    const float *x,
    const reidblas_blas_int *incx,
    const float *beta,
    float *y,
    const reidblas_blas_int *incy
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }
    std::int64_t incy_value = 0;
    const std::int64_t *incy_ilp64 = nullptr;
    if (incy != nullptr) {
        incy_value = static_cast<std::int64_t>(*incy);
        incy_ilp64 = &incy_value;
    }

    reidblas::ilp64::sspmv(uplo, n_ilp64, alpha, ap, x, incx_ilp64, beta, y, incy_ilp64);
}

extern "C" void strmv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const reidblas_blas_int *n,
    const float *a,
    const reidblas_blas_int *lda,
    float *x,
    const reidblas_blas_int *incx
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t lda_value = 0;
    const std::int64_t *lda_ilp64 = nullptr;
    if (lda != nullptr) {
        lda_value = static_cast<std::int64_t>(*lda);
        lda_ilp64 = &lda_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }

    reidblas::ilp64::strmv(uplo, trans, diag, n_ilp64, a, lda_ilp64, x, incx_ilp64);
}

extern "C" void stbmv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const reidblas_blas_int *n,
    const reidblas_blas_int *k,
    const float *a,
    const reidblas_blas_int *lda,
    float *x,
    const reidblas_blas_int *incx
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t k_value = 0;
    const std::int64_t *k_ilp64 = nullptr;
    if (k != nullptr) {
        k_value = static_cast<std::int64_t>(*k);
        k_ilp64 = &k_value;
    }
    std::int64_t lda_value = 0;
    const std::int64_t *lda_ilp64 = nullptr;
    if (lda != nullptr) {
        lda_value = static_cast<std::int64_t>(*lda);
        lda_ilp64 = &lda_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }

    reidblas::ilp64::stbmv(uplo, trans, diag, n_ilp64, k_ilp64, a, lda_ilp64, x, incx_ilp64);
}

extern "C" void stpmv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const reidblas_blas_int *n,
    const float *ap,
    float *x,
    const reidblas_blas_int *incx
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }

    reidblas::ilp64::stpmv(uplo, trans, diag, n_ilp64, ap, x, incx_ilp64);
}

extern "C" void strsv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const reidblas_blas_int *n,
    const float *a,
    const reidblas_blas_int *lda,
    float *x,
    const reidblas_blas_int *incx
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t lda_value = 0;
    const std::int64_t *lda_ilp64 = nullptr;
    if (lda != nullptr) {
        lda_value = static_cast<std::int64_t>(*lda);
        lda_ilp64 = &lda_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }

    reidblas::ilp64::strsv(uplo, trans, diag, n_ilp64, a, lda_ilp64, x, incx_ilp64);
}

extern "C" void stbsv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const reidblas_blas_int *n,
    const reidblas_blas_int *k,
    const float *a,
    const reidblas_blas_int *lda,
    float *x,
    const reidblas_blas_int *incx
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t k_value = 0;
    const std::int64_t *k_ilp64 = nullptr;
    if (k != nullptr) {
        k_value = static_cast<std::int64_t>(*k);
        k_ilp64 = &k_value;
    }
    std::int64_t lda_value = 0;
    const std::int64_t *lda_ilp64 = nullptr;
    if (lda != nullptr) {
        lda_value = static_cast<std::int64_t>(*lda);
        lda_ilp64 = &lda_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }

    reidblas::ilp64::stbsv(uplo, trans, diag, n_ilp64, k_ilp64, a, lda_ilp64, x, incx_ilp64);
}

extern "C" void stpsv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const reidblas_blas_int *n,
    const float *ap,
    float *x,
    const reidblas_blas_int *incx
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }

    reidblas::ilp64::stpsv(uplo, trans, diag, n_ilp64, ap, x, incx_ilp64);
}

extern "C" void sger_(
    const reidblas_blas_int *m,
    const reidblas_blas_int *n,
    const float *alpha,
    const float *x,
    const reidblas_blas_int *incx,
    const float *y,
    const reidblas_blas_int *incy,
    float *a,
    const reidblas_blas_int *lda
) {
    std::int64_t m_value = 0;
    const std::int64_t *m_ilp64 = nullptr;
    if (m != nullptr) {
        m_value = static_cast<std::int64_t>(*m);
        m_ilp64 = &m_value;
    }
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }
    std::int64_t incy_value = 0;
    const std::int64_t *incy_ilp64 = nullptr;
    if (incy != nullptr) {
        incy_value = static_cast<std::int64_t>(*incy);
        incy_ilp64 = &incy_value;
    }
    std::int64_t lda_value = 0;
    const std::int64_t *lda_ilp64 = nullptr;
    if (lda != nullptr) {
        lda_value = static_cast<std::int64_t>(*lda);
        lda_ilp64 = &lda_value;
    }

    reidblas::ilp64::sger(m_ilp64, n_ilp64, alpha, x, incx_ilp64, y, incy_ilp64, a, lda_ilp64);
}

extern "C" void sspr_(
    const char *uplo,
    const reidblas_blas_int *n,
    const float *alpha,
    const float *x,
    const reidblas_blas_int *incx,
    float *ap
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }

    reidblas::ilp64::sspr(uplo, n_ilp64, alpha, x, incx_ilp64, ap);
}

extern "C" void ssyr_(
    const char *uplo,
    const reidblas_blas_int *n,
    const float *alpha,
    const float *x,
    const reidblas_blas_int *incx,
    float *a,
    const reidblas_blas_int *lda
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }
    std::int64_t lda_value = 0;
    const std::int64_t *lda_ilp64 = nullptr;
    if (lda != nullptr) {
        lda_value = static_cast<std::int64_t>(*lda);
        lda_ilp64 = &lda_value;
    }

    reidblas::ilp64::ssyr(uplo, n_ilp64, alpha, x, incx_ilp64, a, lda_ilp64);
}

extern "C" void sspr2_(
    const char *uplo,
    const reidblas_blas_int *n,
    const float *alpha,
    const float *x,
    const reidblas_blas_int *incx,
    const float *y,
    const reidblas_blas_int *incy,
    float *ap
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }
    std::int64_t incy_value = 0;
    const std::int64_t *incy_ilp64 = nullptr;
    if (incy != nullptr) {
        incy_value = static_cast<std::int64_t>(*incy);
        incy_ilp64 = &incy_value;
    }

    reidblas::ilp64::sspr2(uplo, n_ilp64, alpha, x, incx_ilp64, y, incy_ilp64, ap);
}

extern "C" void ssyr2_(
    const char *uplo,
    const reidblas_blas_int *n,
    const float *alpha,
    const float *x,
    const reidblas_blas_int *incx,
    const float *y,
    const reidblas_blas_int *incy,
    float *a,
    const reidblas_blas_int *lda
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }
    std::int64_t incy_value = 0;
    const std::int64_t *incy_ilp64 = nullptr;
    if (incy != nullptr) {
        incy_value = static_cast<std::int64_t>(*incy);
        incy_ilp64 = &incy_value;
    }
    std::int64_t lda_value = 0;
    const std::int64_t *lda_ilp64 = nullptr;
    if (lda != nullptr) {
        lda_value = static_cast<std::int64_t>(*lda);
        lda_ilp64 = &lda_value;
    }

    reidblas::ilp64::ssyr2(uplo, n_ilp64, alpha, x, incx_ilp64, y, incy_ilp64, a, lda_ilp64);
}

extern "C" void dgemv_(
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
) {
    std::int64_t m_value = 0;
    const std::int64_t *m_ilp64 = nullptr;
    if (m != nullptr) {
        m_value = static_cast<std::int64_t>(*m);
        m_ilp64 = &m_value;
    }
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t lda_value = 0;
    const std::int64_t *lda_ilp64 = nullptr;
    if (lda != nullptr) {
        lda_value = static_cast<std::int64_t>(*lda);
        lda_ilp64 = &lda_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }
    std::int64_t incy_value = 0;
    const std::int64_t *incy_ilp64 = nullptr;
    if (incy != nullptr) {
        incy_value = static_cast<std::int64_t>(*incy);
        incy_ilp64 = &incy_value;
    }

    reidblas::ilp64::dgemv(trans, m_ilp64, n_ilp64, alpha, a, lda_ilp64, x, incx_ilp64, beta, y, incy_ilp64);
}

extern "C" void dgbmv_(
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
) {
    std::int64_t m_value = 0;
    const std::int64_t *m_ilp64 = nullptr;
    if (m != nullptr) {
        m_value = static_cast<std::int64_t>(*m);
        m_ilp64 = &m_value;
    }
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t kl_value = 0;
    const std::int64_t *kl_ilp64 = nullptr;
    if (kl != nullptr) {
        kl_value = static_cast<std::int64_t>(*kl);
        kl_ilp64 = &kl_value;
    }
    std::int64_t ku_value = 0;
    const std::int64_t *ku_ilp64 = nullptr;
    if (ku != nullptr) {
        ku_value = static_cast<std::int64_t>(*ku);
        ku_ilp64 = &ku_value;
    }
    std::int64_t lda_value = 0;
    const std::int64_t *lda_ilp64 = nullptr;
    if (lda != nullptr) {
        lda_value = static_cast<std::int64_t>(*lda);
        lda_ilp64 = &lda_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }
    std::int64_t incy_value = 0;
    const std::int64_t *incy_ilp64 = nullptr;
    if (incy != nullptr) {
        incy_value = static_cast<std::int64_t>(*incy);
        incy_ilp64 = &incy_value;
    }

    reidblas::ilp64::dgbmv(trans, m_ilp64, n_ilp64, kl_ilp64, ku_ilp64, alpha, a, lda_ilp64, x, incx_ilp64, beta, y, incy_ilp64);
}

extern "C" void dsymv_(
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
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t lda_value = 0;
    const std::int64_t *lda_ilp64 = nullptr;
    if (lda != nullptr) {
        lda_value = static_cast<std::int64_t>(*lda);
        lda_ilp64 = &lda_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }
    std::int64_t incy_value = 0;
    const std::int64_t *incy_ilp64 = nullptr;
    if (incy != nullptr) {
        incy_value = static_cast<std::int64_t>(*incy);
        incy_ilp64 = &incy_value;
    }

    reidblas::ilp64::dsymv(uplo, n_ilp64, alpha, a, lda_ilp64, x, incx_ilp64, beta, y, incy_ilp64);
}

extern "C" void dsbmv_(
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
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t k_value = 0;
    const std::int64_t *k_ilp64 = nullptr;
    if (k != nullptr) {
        k_value = static_cast<std::int64_t>(*k);
        k_ilp64 = &k_value;
    }
    std::int64_t lda_value = 0;
    const std::int64_t *lda_ilp64 = nullptr;
    if (lda != nullptr) {
        lda_value = static_cast<std::int64_t>(*lda);
        lda_ilp64 = &lda_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }
    std::int64_t incy_value = 0;
    const std::int64_t *incy_ilp64 = nullptr;
    if (incy != nullptr) {
        incy_value = static_cast<std::int64_t>(*incy);
        incy_ilp64 = &incy_value;
    }

    reidblas::ilp64::dsbmv(uplo, n_ilp64, k_ilp64, alpha, a, lda_ilp64, x, incx_ilp64, beta, y, incy_ilp64);
}

extern "C" void dspmv_(
    const char *uplo,
    const reidblas_blas_int *n,
    const double *alpha,
    const double *ap,
    const double *x,
    const reidblas_blas_int *incx,
    const double *beta,
    double *y,
    const reidblas_blas_int *incy
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }
    std::int64_t incy_value = 0;
    const std::int64_t *incy_ilp64 = nullptr;
    if (incy != nullptr) {
        incy_value = static_cast<std::int64_t>(*incy);
        incy_ilp64 = &incy_value;
    }

    reidblas::ilp64::dspmv(uplo, n_ilp64, alpha, ap, x, incx_ilp64, beta, y, incy_ilp64);
}

extern "C" void dtrmv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const reidblas_blas_int *n,
    const double *a,
    const reidblas_blas_int *lda,
    double *x,
    const reidblas_blas_int *incx
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t lda_value = 0;
    const std::int64_t *lda_ilp64 = nullptr;
    if (lda != nullptr) {
        lda_value = static_cast<std::int64_t>(*lda);
        lda_ilp64 = &lda_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }

    reidblas::ilp64::dtrmv(uplo, trans, diag, n_ilp64, a, lda_ilp64, x, incx_ilp64);
}

extern "C" void dtbmv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const reidblas_blas_int *n,
    const reidblas_blas_int *k,
    const double *a,
    const reidblas_blas_int *lda,
    double *x,
    const reidblas_blas_int *incx
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t k_value = 0;
    const std::int64_t *k_ilp64 = nullptr;
    if (k != nullptr) {
        k_value = static_cast<std::int64_t>(*k);
        k_ilp64 = &k_value;
    }
    std::int64_t lda_value = 0;
    const std::int64_t *lda_ilp64 = nullptr;
    if (lda != nullptr) {
        lda_value = static_cast<std::int64_t>(*lda);
        lda_ilp64 = &lda_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }

    reidblas::ilp64::dtbmv(uplo, trans, diag, n_ilp64, k_ilp64, a, lda_ilp64, x, incx_ilp64);
}

extern "C" void dtpmv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const reidblas_blas_int *n,
    const double *ap,
    double *x,
    const reidblas_blas_int *incx
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }

    reidblas::ilp64::dtpmv(uplo, trans, diag, n_ilp64, ap, x, incx_ilp64);
}

extern "C" void dtrsv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const reidblas_blas_int *n,
    const double *a,
    const reidblas_blas_int *lda,
    double *x,
    const reidblas_blas_int *incx
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t lda_value = 0;
    const std::int64_t *lda_ilp64 = nullptr;
    if (lda != nullptr) {
        lda_value = static_cast<std::int64_t>(*lda);
        lda_ilp64 = &lda_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }

    reidblas::ilp64::dtrsv(uplo, trans, diag, n_ilp64, a, lda_ilp64, x, incx_ilp64);
}

extern "C" void dtbsv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const reidblas_blas_int *n,
    const reidblas_blas_int *k,
    const double *a,
    const reidblas_blas_int *lda,
    double *x,
    const reidblas_blas_int *incx
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t k_value = 0;
    const std::int64_t *k_ilp64 = nullptr;
    if (k != nullptr) {
        k_value = static_cast<std::int64_t>(*k);
        k_ilp64 = &k_value;
    }
    std::int64_t lda_value = 0;
    const std::int64_t *lda_ilp64 = nullptr;
    if (lda != nullptr) {
        lda_value = static_cast<std::int64_t>(*lda);
        lda_ilp64 = &lda_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }

    reidblas::ilp64::dtbsv(uplo, trans, diag, n_ilp64, k_ilp64, a, lda_ilp64, x, incx_ilp64);
}

extern "C" void dtpsv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const reidblas_blas_int *n,
    const double *ap,
    double *x,
    const reidblas_blas_int *incx
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }

    reidblas::ilp64::dtpsv(uplo, trans, diag, n_ilp64, ap, x, incx_ilp64);
}

extern "C" void dger_(
    const reidblas_blas_int *m,
    const reidblas_blas_int *n,
    const double *alpha,
    const double *x,
    const reidblas_blas_int *incx,
    const double *y,
    const reidblas_blas_int *incy,
    double *a,
    const reidblas_blas_int *lda
) {
    std::int64_t m_value = 0;
    const std::int64_t *m_ilp64 = nullptr;
    if (m != nullptr) {
        m_value = static_cast<std::int64_t>(*m);
        m_ilp64 = &m_value;
    }
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }
    std::int64_t incy_value = 0;
    const std::int64_t *incy_ilp64 = nullptr;
    if (incy != nullptr) {
        incy_value = static_cast<std::int64_t>(*incy);
        incy_ilp64 = &incy_value;
    }
    std::int64_t lda_value = 0;
    const std::int64_t *lda_ilp64 = nullptr;
    if (lda != nullptr) {
        lda_value = static_cast<std::int64_t>(*lda);
        lda_ilp64 = &lda_value;
    }

    reidblas::ilp64::dger(m_ilp64, n_ilp64, alpha, x, incx_ilp64, y, incy_ilp64, a, lda_ilp64);
}

extern "C" void dspr_(
    const char *uplo,
    const reidblas_blas_int *n,
    const double *alpha,
    const double *x,
    const reidblas_blas_int *incx,
    double *ap
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }

    reidblas::ilp64::dspr(uplo, n_ilp64, alpha, x, incx_ilp64, ap);
}

extern "C" void dsyr_(
    const char *uplo,
    const reidblas_blas_int *n,
    const double *alpha,
    const double *x,
    const reidblas_blas_int *incx,
    double *a,
    const reidblas_blas_int *lda
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }
    std::int64_t lda_value = 0;
    const std::int64_t *lda_ilp64 = nullptr;
    if (lda != nullptr) {
        lda_value = static_cast<std::int64_t>(*lda);
        lda_ilp64 = &lda_value;
    }

    reidblas::ilp64::dsyr(uplo, n_ilp64, alpha, x, incx_ilp64, a, lda_ilp64);
}

extern "C" void dspr2_(
    const char *uplo,
    const reidblas_blas_int *n,
    const double *alpha,
    const double *x,
    const reidblas_blas_int *incx,
    const double *y,
    const reidblas_blas_int *incy,
    double *ap
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }
    std::int64_t incy_value = 0;
    const std::int64_t *incy_ilp64 = nullptr;
    if (incy != nullptr) {
        incy_value = static_cast<std::int64_t>(*incy);
        incy_ilp64 = &incy_value;
    }

    reidblas::ilp64::dspr2(uplo, n_ilp64, alpha, x, incx_ilp64, y, incy_ilp64, ap);
}

extern "C" void dsyr2_(
    const char *uplo,
    const reidblas_blas_int *n,
    const double *alpha,
    const double *x,
    const reidblas_blas_int *incx,
    const double *y,
    const reidblas_blas_int *incy,
    double *a,
    const reidblas_blas_int *lda
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }
    std::int64_t incy_value = 0;
    const std::int64_t *incy_ilp64 = nullptr;
    if (incy != nullptr) {
        incy_value = static_cast<std::int64_t>(*incy);
        incy_ilp64 = &incy_value;
    }
    std::int64_t lda_value = 0;
    const std::int64_t *lda_ilp64 = nullptr;
    if (lda != nullptr) {
        lda_value = static_cast<std::int64_t>(*lda);
        lda_ilp64 = &lda_value;
    }

    reidblas::ilp64::dsyr2(uplo, n_ilp64, alpha, x, incx_ilp64, y, incy_ilp64, a, lda_ilp64);
}

extern "C" void cgemv_(
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
) {
    std::int64_t m_value = 0;
    const std::int64_t *m_ilp64 = nullptr;
    if (m != nullptr) {
        m_value = static_cast<std::int64_t>(*m);
        m_ilp64 = &m_value;
    }
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t lda_value = 0;
    const std::int64_t *lda_ilp64 = nullptr;
    if (lda != nullptr) {
        lda_value = static_cast<std::int64_t>(*lda);
        lda_ilp64 = &lda_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }
    std::int64_t incy_value = 0;
    const std::int64_t *incy_ilp64 = nullptr;
    if (incy != nullptr) {
        incy_value = static_cast<std::int64_t>(*incy);
        incy_ilp64 = &incy_value;
    }

    reidblas::ilp64::cgemv(trans, m_ilp64, n_ilp64, alpha, a, lda_ilp64, x, incx_ilp64, beta, y, incy_ilp64);
}

extern "C" void cgbmv_(
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
) {
    std::int64_t m_value = 0;
    const std::int64_t *m_ilp64 = nullptr;
    if (m != nullptr) {
        m_value = static_cast<std::int64_t>(*m);
        m_ilp64 = &m_value;
    }
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t kl_value = 0;
    const std::int64_t *kl_ilp64 = nullptr;
    if (kl != nullptr) {
        kl_value = static_cast<std::int64_t>(*kl);
        kl_ilp64 = &kl_value;
    }
    std::int64_t ku_value = 0;
    const std::int64_t *ku_ilp64 = nullptr;
    if (ku != nullptr) {
        ku_value = static_cast<std::int64_t>(*ku);
        ku_ilp64 = &ku_value;
    }
    std::int64_t lda_value = 0;
    const std::int64_t *lda_ilp64 = nullptr;
    if (lda != nullptr) {
        lda_value = static_cast<std::int64_t>(*lda);
        lda_ilp64 = &lda_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }
    std::int64_t incy_value = 0;
    const std::int64_t *incy_ilp64 = nullptr;
    if (incy != nullptr) {
        incy_value = static_cast<std::int64_t>(*incy);
        incy_ilp64 = &incy_value;
    }

    reidblas::ilp64::cgbmv(trans, m_ilp64, n_ilp64, kl_ilp64, ku_ilp64, alpha, a, lda_ilp64, x, incx_ilp64, beta, y, incy_ilp64);
}

extern "C" void chemv_(
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
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t lda_value = 0;
    const std::int64_t *lda_ilp64 = nullptr;
    if (lda != nullptr) {
        lda_value = static_cast<std::int64_t>(*lda);
        lda_ilp64 = &lda_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }
    std::int64_t incy_value = 0;
    const std::int64_t *incy_ilp64 = nullptr;
    if (incy != nullptr) {
        incy_value = static_cast<std::int64_t>(*incy);
        incy_ilp64 = &incy_value;
    }

    reidblas::ilp64::chemv(uplo, n_ilp64, alpha, a, lda_ilp64, x, incx_ilp64, beta, y, incy_ilp64);
}

extern "C" void chbmv_(
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
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t k_value = 0;
    const std::int64_t *k_ilp64 = nullptr;
    if (k != nullptr) {
        k_value = static_cast<std::int64_t>(*k);
        k_ilp64 = &k_value;
    }
    std::int64_t lda_value = 0;
    const std::int64_t *lda_ilp64 = nullptr;
    if (lda != nullptr) {
        lda_value = static_cast<std::int64_t>(*lda);
        lda_ilp64 = &lda_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }
    std::int64_t incy_value = 0;
    const std::int64_t *incy_ilp64 = nullptr;
    if (incy != nullptr) {
        incy_value = static_cast<std::int64_t>(*incy);
        incy_ilp64 = &incy_value;
    }

    reidblas::ilp64::chbmv(uplo, n_ilp64, k_ilp64, alpha, a, lda_ilp64, x, incx_ilp64, beta, y, incy_ilp64);
}

extern "C" void chpmv_(
    const char *uplo,
    const reidblas_blas_int *n,
    const reidblas_complex_float *alpha,
    const reidblas_complex_float *ap,
    const reidblas_complex_float *x,
    const reidblas_blas_int *incx,
    const reidblas_complex_float *beta,
    reidblas_complex_float *y,
    const reidblas_blas_int *incy
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }
    std::int64_t incy_value = 0;
    const std::int64_t *incy_ilp64 = nullptr;
    if (incy != nullptr) {
        incy_value = static_cast<std::int64_t>(*incy);
        incy_ilp64 = &incy_value;
    }

    reidblas::ilp64::chpmv(uplo, n_ilp64, alpha, ap, x, incx_ilp64, beta, y, incy_ilp64);
}

extern "C" void ctrmv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const reidblas_blas_int *n,
    const reidblas_complex_float *a,
    const reidblas_blas_int *lda,
    reidblas_complex_float *x,
    const reidblas_blas_int *incx
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t lda_value = 0;
    const std::int64_t *lda_ilp64 = nullptr;
    if (lda != nullptr) {
        lda_value = static_cast<std::int64_t>(*lda);
        lda_ilp64 = &lda_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }

    reidblas::ilp64::ctrmv(uplo, trans, diag, n_ilp64, a, lda_ilp64, x, incx_ilp64);
}

extern "C" void ctbmv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const reidblas_blas_int *n,
    const reidblas_blas_int *k,
    const reidblas_complex_float *a,
    const reidblas_blas_int *lda,
    reidblas_complex_float *x,
    const reidblas_blas_int *incx
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t k_value = 0;
    const std::int64_t *k_ilp64 = nullptr;
    if (k != nullptr) {
        k_value = static_cast<std::int64_t>(*k);
        k_ilp64 = &k_value;
    }
    std::int64_t lda_value = 0;
    const std::int64_t *lda_ilp64 = nullptr;
    if (lda != nullptr) {
        lda_value = static_cast<std::int64_t>(*lda);
        lda_ilp64 = &lda_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }

    reidblas::ilp64::ctbmv(uplo, trans, diag, n_ilp64, k_ilp64, a, lda_ilp64, x, incx_ilp64);
}

extern "C" void ctpmv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const reidblas_blas_int *n,
    const reidblas_complex_float *ap,
    reidblas_complex_float *x,
    const reidblas_blas_int *incx
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }

    reidblas::ilp64::ctpmv(uplo, trans, diag, n_ilp64, ap, x, incx_ilp64);
}

extern "C" void ctrsv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const reidblas_blas_int *n,
    const reidblas_complex_float *a,
    const reidblas_blas_int *lda,
    reidblas_complex_float *x,
    const reidblas_blas_int *incx
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t lda_value = 0;
    const std::int64_t *lda_ilp64 = nullptr;
    if (lda != nullptr) {
        lda_value = static_cast<std::int64_t>(*lda);
        lda_ilp64 = &lda_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }

    reidblas::ilp64::ctrsv(uplo, trans, diag, n_ilp64, a, lda_ilp64, x, incx_ilp64);
}

extern "C" void ctbsv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const reidblas_blas_int *n,
    const reidblas_blas_int *k,
    const reidblas_complex_float *a,
    const reidblas_blas_int *lda,
    reidblas_complex_float *x,
    const reidblas_blas_int *incx
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t k_value = 0;
    const std::int64_t *k_ilp64 = nullptr;
    if (k != nullptr) {
        k_value = static_cast<std::int64_t>(*k);
        k_ilp64 = &k_value;
    }
    std::int64_t lda_value = 0;
    const std::int64_t *lda_ilp64 = nullptr;
    if (lda != nullptr) {
        lda_value = static_cast<std::int64_t>(*lda);
        lda_ilp64 = &lda_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }

    reidblas::ilp64::ctbsv(uplo, trans, diag, n_ilp64, k_ilp64, a, lda_ilp64, x, incx_ilp64);
}

extern "C" void ctpsv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const reidblas_blas_int *n,
    const reidblas_complex_float *ap,
    reidblas_complex_float *x,
    const reidblas_blas_int *incx
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }

    reidblas::ilp64::ctpsv(uplo, trans, diag, n_ilp64, ap, x, incx_ilp64);
}

extern "C" void cgerc_(
    const reidblas_blas_int *m,
    const reidblas_blas_int *n,
    const reidblas_complex_float *alpha,
    const reidblas_complex_float *x,
    const reidblas_blas_int *incx,
    const reidblas_complex_float *y,
    const reidblas_blas_int *incy,
    reidblas_complex_float *a,
    const reidblas_blas_int *lda
) {
    std::int64_t m_value = 0;
    const std::int64_t *m_ilp64 = nullptr;
    if (m != nullptr) {
        m_value = static_cast<std::int64_t>(*m);
        m_ilp64 = &m_value;
    }
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }
    std::int64_t incy_value = 0;
    const std::int64_t *incy_ilp64 = nullptr;
    if (incy != nullptr) {
        incy_value = static_cast<std::int64_t>(*incy);
        incy_ilp64 = &incy_value;
    }
    std::int64_t lda_value = 0;
    const std::int64_t *lda_ilp64 = nullptr;
    if (lda != nullptr) {
        lda_value = static_cast<std::int64_t>(*lda);
        lda_ilp64 = &lda_value;
    }

    reidblas::ilp64::cgerc(m_ilp64, n_ilp64, alpha, x, incx_ilp64, y, incy_ilp64, a, lda_ilp64);
}

extern "C" void cgeru_(
    const reidblas_blas_int *m,
    const reidblas_blas_int *n,
    const reidblas_complex_float *alpha,
    const reidblas_complex_float *x,
    const reidblas_blas_int *incx,
    const reidblas_complex_float *y,
    const reidblas_blas_int *incy,
    reidblas_complex_float *a,
    const reidblas_blas_int *lda
) {
    std::int64_t m_value = 0;
    const std::int64_t *m_ilp64 = nullptr;
    if (m != nullptr) {
        m_value = static_cast<std::int64_t>(*m);
        m_ilp64 = &m_value;
    }
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }
    std::int64_t incy_value = 0;
    const std::int64_t *incy_ilp64 = nullptr;
    if (incy != nullptr) {
        incy_value = static_cast<std::int64_t>(*incy);
        incy_ilp64 = &incy_value;
    }
    std::int64_t lda_value = 0;
    const std::int64_t *lda_ilp64 = nullptr;
    if (lda != nullptr) {
        lda_value = static_cast<std::int64_t>(*lda);
        lda_ilp64 = &lda_value;
    }

    reidblas::ilp64::cgeru(m_ilp64, n_ilp64, alpha, x, incx_ilp64, y, incy_ilp64, a, lda_ilp64);
}

extern "C" void cher_(
    const char *uplo,
    const reidblas_blas_int *n,
    const float *alpha,
    const reidblas_complex_float *x,
    const reidblas_blas_int *incx,
    reidblas_complex_float *a,
    const reidblas_blas_int *lda
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }
    std::int64_t lda_value = 0;
    const std::int64_t *lda_ilp64 = nullptr;
    if (lda != nullptr) {
        lda_value = static_cast<std::int64_t>(*lda);
        lda_ilp64 = &lda_value;
    }

    reidblas::ilp64::cher(uplo, n_ilp64, alpha, x, incx_ilp64, a, lda_ilp64);
}

extern "C" void chpr_(
    const char *uplo,
    const reidblas_blas_int *n,
    const float *alpha,
    const reidblas_complex_float *x,
    const reidblas_blas_int *incx,
    reidblas_complex_float *ap
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }

    reidblas::ilp64::chpr(uplo, n_ilp64, alpha, x, incx_ilp64, ap);
}

extern "C" void cher2_(
    const char *uplo,
    const reidblas_blas_int *n,
    const reidblas_complex_float *alpha,
    const reidblas_complex_float *x,
    const reidblas_blas_int *incx,
    const reidblas_complex_float *y,
    const reidblas_blas_int *incy,
    reidblas_complex_float *a,
    const reidblas_blas_int *lda
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }
    std::int64_t incy_value = 0;
    const std::int64_t *incy_ilp64 = nullptr;
    if (incy != nullptr) {
        incy_value = static_cast<std::int64_t>(*incy);
        incy_ilp64 = &incy_value;
    }
    std::int64_t lda_value = 0;
    const std::int64_t *lda_ilp64 = nullptr;
    if (lda != nullptr) {
        lda_value = static_cast<std::int64_t>(*lda);
        lda_ilp64 = &lda_value;
    }

    reidblas::ilp64::cher2(uplo, n_ilp64, alpha, x, incx_ilp64, y, incy_ilp64, a, lda_ilp64);
}

extern "C" void chpr2_(
    const char *uplo,
    const reidblas_blas_int *n,
    const reidblas_complex_float *alpha,
    const reidblas_complex_float *x,
    const reidblas_blas_int *incx,
    const reidblas_complex_float *y,
    const reidblas_blas_int *incy,
    reidblas_complex_float *ap
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }
    std::int64_t incy_value = 0;
    const std::int64_t *incy_ilp64 = nullptr;
    if (incy != nullptr) {
        incy_value = static_cast<std::int64_t>(*incy);
        incy_ilp64 = &incy_value;
    }

    reidblas::ilp64::chpr2(uplo, n_ilp64, alpha, x, incx_ilp64, y, incy_ilp64, ap);
}

extern "C" void zgemv_(
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
) {
    std::int64_t m_value = 0;
    const std::int64_t *m_ilp64 = nullptr;
    if (m != nullptr) {
        m_value = static_cast<std::int64_t>(*m);
        m_ilp64 = &m_value;
    }
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t lda_value = 0;
    const std::int64_t *lda_ilp64 = nullptr;
    if (lda != nullptr) {
        lda_value = static_cast<std::int64_t>(*lda);
        lda_ilp64 = &lda_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }
    std::int64_t incy_value = 0;
    const std::int64_t *incy_ilp64 = nullptr;
    if (incy != nullptr) {
        incy_value = static_cast<std::int64_t>(*incy);
        incy_ilp64 = &incy_value;
    }

    reidblas::ilp64::zgemv(trans, m_ilp64, n_ilp64, alpha, a, lda_ilp64, x, incx_ilp64, beta, y, incy_ilp64);
}

extern "C" void zgbmv_(
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
) {
    std::int64_t m_value = 0;
    const std::int64_t *m_ilp64 = nullptr;
    if (m != nullptr) {
        m_value = static_cast<std::int64_t>(*m);
        m_ilp64 = &m_value;
    }
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t kl_value = 0;
    const std::int64_t *kl_ilp64 = nullptr;
    if (kl != nullptr) {
        kl_value = static_cast<std::int64_t>(*kl);
        kl_ilp64 = &kl_value;
    }
    std::int64_t ku_value = 0;
    const std::int64_t *ku_ilp64 = nullptr;
    if (ku != nullptr) {
        ku_value = static_cast<std::int64_t>(*ku);
        ku_ilp64 = &ku_value;
    }
    std::int64_t lda_value = 0;
    const std::int64_t *lda_ilp64 = nullptr;
    if (lda != nullptr) {
        lda_value = static_cast<std::int64_t>(*lda);
        lda_ilp64 = &lda_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }
    std::int64_t incy_value = 0;
    const std::int64_t *incy_ilp64 = nullptr;
    if (incy != nullptr) {
        incy_value = static_cast<std::int64_t>(*incy);
        incy_ilp64 = &incy_value;
    }

    reidblas::ilp64::zgbmv(trans, m_ilp64, n_ilp64, kl_ilp64, ku_ilp64, alpha, a, lda_ilp64, x, incx_ilp64, beta, y, incy_ilp64);
}

extern "C" void zhemv_(
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
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t lda_value = 0;
    const std::int64_t *lda_ilp64 = nullptr;
    if (lda != nullptr) {
        lda_value = static_cast<std::int64_t>(*lda);
        lda_ilp64 = &lda_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }
    std::int64_t incy_value = 0;
    const std::int64_t *incy_ilp64 = nullptr;
    if (incy != nullptr) {
        incy_value = static_cast<std::int64_t>(*incy);
        incy_ilp64 = &incy_value;
    }

    reidblas::ilp64::zhemv(uplo, n_ilp64, alpha, a, lda_ilp64, x, incx_ilp64, beta, y, incy_ilp64);
}

extern "C" void zhbmv_(
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
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t k_value = 0;
    const std::int64_t *k_ilp64 = nullptr;
    if (k != nullptr) {
        k_value = static_cast<std::int64_t>(*k);
        k_ilp64 = &k_value;
    }
    std::int64_t lda_value = 0;
    const std::int64_t *lda_ilp64 = nullptr;
    if (lda != nullptr) {
        lda_value = static_cast<std::int64_t>(*lda);
        lda_ilp64 = &lda_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }
    std::int64_t incy_value = 0;
    const std::int64_t *incy_ilp64 = nullptr;
    if (incy != nullptr) {
        incy_value = static_cast<std::int64_t>(*incy);
        incy_ilp64 = &incy_value;
    }

    reidblas::ilp64::zhbmv(uplo, n_ilp64, k_ilp64, alpha, a, lda_ilp64, x, incx_ilp64, beta, y, incy_ilp64);
}

extern "C" void zhpmv_(
    const char *uplo,
    const reidblas_blas_int *n,
    const reidblas_complex_double *alpha,
    const reidblas_complex_double *ap,
    const reidblas_complex_double *x,
    const reidblas_blas_int *incx,
    const reidblas_complex_double *beta,
    reidblas_complex_double *y,
    const reidblas_blas_int *incy
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }
    std::int64_t incy_value = 0;
    const std::int64_t *incy_ilp64 = nullptr;
    if (incy != nullptr) {
        incy_value = static_cast<std::int64_t>(*incy);
        incy_ilp64 = &incy_value;
    }

    reidblas::ilp64::zhpmv(uplo, n_ilp64, alpha, ap, x, incx_ilp64, beta, y, incy_ilp64);
}

extern "C" void ztrmv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const reidblas_blas_int *n,
    const reidblas_complex_double *a,
    const reidblas_blas_int *lda,
    reidblas_complex_double *x,
    const reidblas_blas_int *incx
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t lda_value = 0;
    const std::int64_t *lda_ilp64 = nullptr;
    if (lda != nullptr) {
        lda_value = static_cast<std::int64_t>(*lda);
        lda_ilp64 = &lda_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }

    reidblas::ilp64::ztrmv(uplo, trans, diag, n_ilp64, a, lda_ilp64, x, incx_ilp64);
}

extern "C" void ztbmv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const reidblas_blas_int *n,
    const reidblas_blas_int *k,
    const reidblas_complex_double *a,
    const reidblas_blas_int *lda,
    reidblas_complex_double *x,
    const reidblas_blas_int *incx
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t k_value = 0;
    const std::int64_t *k_ilp64 = nullptr;
    if (k != nullptr) {
        k_value = static_cast<std::int64_t>(*k);
        k_ilp64 = &k_value;
    }
    std::int64_t lda_value = 0;
    const std::int64_t *lda_ilp64 = nullptr;
    if (lda != nullptr) {
        lda_value = static_cast<std::int64_t>(*lda);
        lda_ilp64 = &lda_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }

    reidblas::ilp64::ztbmv(uplo, trans, diag, n_ilp64, k_ilp64, a, lda_ilp64, x, incx_ilp64);
}

extern "C" void ztpmv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const reidblas_blas_int *n,
    const reidblas_complex_double *ap,
    reidblas_complex_double *x,
    const reidblas_blas_int *incx
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }

    reidblas::ilp64::ztpmv(uplo, trans, diag, n_ilp64, ap, x, incx_ilp64);
}

extern "C" void ztrsv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const reidblas_blas_int *n,
    const reidblas_complex_double *a,
    const reidblas_blas_int *lda,
    reidblas_complex_double *x,
    const reidblas_blas_int *incx
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t lda_value = 0;
    const std::int64_t *lda_ilp64 = nullptr;
    if (lda != nullptr) {
        lda_value = static_cast<std::int64_t>(*lda);
        lda_ilp64 = &lda_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }

    reidblas::ilp64::ztrsv(uplo, trans, diag, n_ilp64, a, lda_ilp64, x, incx_ilp64);
}

extern "C" void ztbsv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const reidblas_blas_int *n,
    const reidblas_blas_int *k,
    const reidblas_complex_double *a,
    const reidblas_blas_int *lda,
    reidblas_complex_double *x,
    const reidblas_blas_int *incx
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t k_value = 0;
    const std::int64_t *k_ilp64 = nullptr;
    if (k != nullptr) {
        k_value = static_cast<std::int64_t>(*k);
        k_ilp64 = &k_value;
    }
    std::int64_t lda_value = 0;
    const std::int64_t *lda_ilp64 = nullptr;
    if (lda != nullptr) {
        lda_value = static_cast<std::int64_t>(*lda);
        lda_ilp64 = &lda_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }

    reidblas::ilp64::ztbsv(uplo, trans, diag, n_ilp64, k_ilp64, a, lda_ilp64, x, incx_ilp64);
}

extern "C" void ztpsv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const reidblas_blas_int *n,
    const reidblas_complex_double *ap,
    reidblas_complex_double *x,
    const reidblas_blas_int *incx
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }

    reidblas::ilp64::ztpsv(uplo, trans, diag, n_ilp64, ap, x, incx_ilp64);
}

extern "C" void zgerc_(
    const reidblas_blas_int *m,
    const reidblas_blas_int *n,
    const reidblas_complex_double *alpha,
    const reidblas_complex_double *x,
    const reidblas_blas_int *incx,
    const reidblas_complex_double *y,
    const reidblas_blas_int *incy,
    reidblas_complex_double *a,
    const reidblas_blas_int *lda
) {
    std::int64_t m_value = 0;
    const std::int64_t *m_ilp64 = nullptr;
    if (m != nullptr) {
        m_value = static_cast<std::int64_t>(*m);
        m_ilp64 = &m_value;
    }
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }
    std::int64_t incy_value = 0;
    const std::int64_t *incy_ilp64 = nullptr;
    if (incy != nullptr) {
        incy_value = static_cast<std::int64_t>(*incy);
        incy_ilp64 = &incy_value;
    }
    std::int64_t lda_value = 0;
    const std::int64_t *lda_ilp64 = nullptr;
    if (lda != nullptr) {
        lda_value = static_cast<std::int64_t>(*lda);
        lda_ilp64 = &lda_value;
    }

    reidblas::ilp64::zgerc(m_ilp64, n_ilp64, alpha, x, incx_ilp64, y, incy_ilp64, a, lda_ilp64);
}

extern "C" void zgeru_(
    const reidblas_blas_int *m,
    const reidblas_blas_int *n,
    const reidblas_complex_double *alpha,
    const reidblas_complex_double *x,
    const reidblas_blas_int *incx,
    const reidblas_complex_double *y,
    const reidblas_blas_int *incy,
    reidblas_complex_double *a,
    const reidblas_blas_int *lda
) {
    std::int64_t m_value = 0;
    const std::int64_t *m_ilp64 = nullptr;
    if (m != nullptr) {
        m_value = static_cast<std::int64_t>(*m);
        m_ilp64 = &m_value;
    }
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }
    std::int64_t incy_value = 0;
    const std::int64_t *incy_ilp64 = nullptr;
    if (incy != nullptr) {
        incy_value = static_cast<std::int64_t>(*incy);
        incy_ilp64 = &incy_value;
    }
    std::int64_t lda_value = 0;
    const std::int64_t *lda_ilp64 = nullptr;
    if (lda != nullptr) {
        lda_value = static_cast<std::int64_t>(*lda);
        lda_ilp64 = &lda_value;
    }

    reidblas::ilp64::zgeru(m_ilp64, n_ilp64, alpha, x, incx_ilp64, y, incy_ilp64, a, lda_ilp64);
}

extern "C" void zher_(
    const char *uplo,
    const reidblas_blas_int *n,
    const double *alpha,
    const reidblas_complex_double *x,
    const reidblas_blas_int *incx,
    reidblas_complex_double *a,
    const reidblas_blas_int *lda
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }
    std::int64_t lda_value = 0;
    const std::int64_t *lda_ilp64 = nullptr;
    if (lda != nullptr) {
        lda_value = static_cast<std::int64_t>(*lda);
        lda_ilp64 = &lda_value;
    }

    reidblas::ilp64::zher(uplo, n_ilp64, alpha, x, incx_ilp64, a, lda_ilp64);
}

extern "C" void zhpr_(
    const char *uplo,
    const reidblas_blas_int *n,
    const double *alpha,
    const reidblas_complex_double *x,
    const reidblas_blas_int *incx,
    reidblas_complex_double *ap
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }

    reidblas::ilp64::zhpr(uplo, n_ilp64, alpha, x, incx_ilp64, ap);
}

extern "C" void zher2_(
    const char *uplo,
    const reidblas_blas_int *n,
    const reidblas_complex_double *alpha,
    const reidblas_complex_double *x,
    const reidblas_blas_int *incx,
    const reidblas_complex_double *y,
    const reidblas_blas_int *incy,
    reidblas_complex_double *a,
    const reidblas_blas_int *lda
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }
    std::int64_t incy_value = 0;
    const std::int64_t *incy_ilp64 = nullptr;
    if (incy != nullptr) {
        incy_value = static_cast<std::int64_t>(*incy);
        incy_ilp64 = &incy_value;
    }
    std::int64_t lda_value = 0;
    const std::int64_t *lda_ilp64 = nullptr;
    if (lda != nullptr) {
        lda_value = static_cast<std::int64_t>(*lda);
        lda_ilp64 = &lda_value;
    }

    reidblas::ilp64::zher2(uplo, n_ilp64, alpha, x, incx_ilp64, y, incy_ilp64, a, lda_ilp64);
}

extern "C" void zhpr2_(
    const char *uplo,
    const reidblas_blas_int *n,
    const reidblas_complex_double *alpha,
    const reidblas_complex_double *x,
    const reidblas_blas_int *incx,
    const reidblas_complex_double *y,
    const reidblas_blas_int *incy,
    reidblas_complex_double *ap
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t incx_value = 0;
    const std::int64_t *incx_ilp64 = nullptr;
    if (incx != nullptr) {
        incx_value = static_cast<std::int64_t>(*incx);
        incx_ilp64 = &incx_value;
    }
    std::int64_t incy_value = 0;
    const std::int64_t *incy_ilp64 = nullptr;
    if (incy != nullptr) {
        incy_value = static_cast<std::int64_t>(*incy);
        incy_ilp64 = &incy_value;
    }

    reidblas::ilp64::zhpr2(uplo, n_ilp64, alpha, x, incx_ilp64, y, incy_ilp64, ap);
}

extern "C" void sgemm_(
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
) {
    std::int64_t m_value = 0;
    const std::int64_t *m_ilp64 = nullptr;
    if (m != nullptr) {
        m_value = static_cast<std::int64_t>(*m);
        m_ilp64 = &m_value;
    }
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t k_value = 0;
    const std::int64_t *k_ilp64 = nullptr;
    if (k != nullptr) {
        k_value = static_cast<std::int64_t>(*k);
        k_ilp64 = &k_value;
    }
    std::int64_t lda_value = 0;
    const std::int64_t *lda_ilp64 = nullptr;
    if (lda != nullptr) {
        lda_value = static_cast<std::int64_t>(*lda);
        lda_ilp64 = &lda_value;
    }
    std::int64_t ldb_value = 0;
    const std::int64_t *ldb_ilp64 = nullptr;
    if (ldb != nullptr) {
        ldb_value = static_cast<std::int64_t>(*ldb);
        ldb_ilp64 = &ldb_value;
    }
    std::int64_t ldc_value = 0;
    const std::int64_t *ldc_ilp64 = nullptr;
    if (ldc != nullptr) {
        ldc_value = static_cast<std::int64_t>(*ldc);
        ldc_ilp64 = &ldc_value;
    }

    reidblas::ilp64::sgemm(transa, transb, m_ilp64, n_ilp64, k_ilp64, alpha, a, lda_ilp64, b, ldb_ilp64, beta, c, ldc_ilp64);
}

extern "C" void ssymm_(
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
) {
    std::int64_t m_value = 0;
    const std::int64_t *m_ilp64 = nullptr;
    if (m != nullptr) {
        m_value = static_cast<std::int64_t>(*m);
        m_ilp64 = &m_value;
    }
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t lda_value = 0;
    const std::int64_t *lda_ilp64 = nullptr;
    if (lda != nullptr) {
        lda_value = static_cast<std::int64_t>(*lda);
        lda_ilp64 = &lda_value;
    }
    std::int64_t ldb_value = 0;
    const std::int64_t *ldb_ilp64 = nullptr;
    if (ldb != nullptr) {
        ldb_value = static_cast<std::int64_t>(*ldb);
        ldb_ilp64 = &ldb_value;
    }
    std::int64_t ldc_value = 0;
    const std::int64_t *ldc_ilp64 = nullptr;
    if (ldc != nullptr) {
        ldc_value = static_cast<std::int64_t>(*ldc);
        ldc_ilp64 = &ldc_value;
    }

    reidblas::ilp64::ssymm(side, uplo, m_ilp64, n_ilp64, alpha, a, lda_ilp64, b, ldb_ilp64, beta, c, ldc_ilp64);
}

extern "C" void ssyrk_(
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
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t k_value = 0;
    const std::int64_t *k_ilp64 = nullptr;
    if (k != nullptr) {
        k_value = static_cast<std::int64_t>(*k);
        k_ilp64 = &k_value;
    }
    std::int64_t lda_value = 0;
    const std::int64_t *lda_ilp64 = nullptr;
    if (lda != nullptr) {
        lda_value = static_cast<std::int64_t>(*lda);
        lda_ilp64 = &lda_value;
    }
    std::int64_t ldc_value = 0;
    const std::int64_t *ldc_ilp64 = nullptr;
    if (ldc != nullptr) {
        ldc_value = static_cast<std::int64_t>(*ldc);
        ldc_ilp64 = &ldc_value;
    }

    reidblas::ilp64::ssyrk(uplo, trans, n_ilp64, k_ilp64, alpha, a, lda_ilp64, beta, c, ldc_ilp64);
}

extern "C" void ssyr2k_(
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
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t k_value = 0;
    const std::int64_t *k_ilp64 = nullptr;
    if (k != nullptr) {
        k_value = static_cast<std::int64_t>(*k);
        k_ilp64 = &k_value;
    }
    std::int64_t lda_value = 0;
    const std::int64_t *lda_ilp64 = nullptr;
    if (lda != nullptr) {
        lda_value = static_cast<std::int64_t>(*lda);
        lda_ilp64 = &lda_value;
    }
    std::int64_t ldb_value = 0;
    const std::int64_t *ldb_ilp64 = nullptr;
    if (ldb != nullptr) {
        ldb_value = static_cast<std::int64_t>(*ldb);
        ldb_ilp64 = &ldb_value;
    }
    std::int64_t ldc_value = 0;
    const std::int64_t *ldc_ilp64 = nullptr;
    if (ldc != nullptr) {
        ldc_value = static_cast<std::int64_t>(*ldc);
        ldc_ilp64 = &ldc_value;
    }

    reidblas::ilp64::ssyr2k(uplo, trans, n_ilp64, k_ilp64, alpha, a, lda_ilp64, b, ldb_ilp64, beta, c, ldc_ilp64);
}

extern "C" void strsm_(
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
) {
    std::int64_t m_value = 0;
    const std::int64_t *m_ilp64 = nullptr;
    if (m != nullptr) {
        m_value = static_cast<std::int64_t>(*m);
        m_ilp64 = &m_value;
    }
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t lda_value = 0;
    const std::int64_t *lda_ilp64 = nullptr;
    if (lda != nullptr) {
        lda_value = static_cast<std::int64_t>(*lda);
        lda_ilp64 = &lda_value;
    }
    std::int64_t ldb_value = 0;
    const std::int64_t *ldb_ilp64 = nullptr;
    if (ldb != nullptr) {
        ldb_value = static_cast<std::int64_t>(*ldb);
        ldb_ilp64 = &ldb_value;
    }

    reidblas::ilp64::strsm(side, uplo, transa, diag, m_ilp64, n_ilp64, alpha, a, lda_ilp64, b, ldb_ilp64);
}

extern "C" void strmm_(
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
) {
    std::int64_t m_value = 0;
    const std::int64_t *m_ilp64 = nullptr;
    if (m != nullptr) {
        m_value = static_cast<std::int64_t>(*m);
        m_ilp64 = &m_value;
    }
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t lda_value = 0;
    const std::int64_t *lda_ilp64 = nullptr;
    if (lda != nullptr) {
        lda_value = static_cast<std::int64_t>(*lda);
        lda_ilp64 = &lda_value;
    }
    std::int64_t ldb_value = 0;
    const std::int64_t *ldb_ilp64 = nullptr;
    if (ldb != nullptr) {
        ldb_value = static_cast<std::int64_t>(*ldb);
        ldb_ilp64 = &ldb_value;
    }

    reidblas::ilp64::strmm(side, uplo, transa, diag, m_ilp64, n_ilp64, alpha, a, lda_ilp64, b, ldb_ilp64);
}

extern "C" void dgemm_(
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
) {
    std::int64_t m_value = 0;
    const std::int64_t *m_ilp64 = nullptr;
    if (m != nullptr) {
        m_value = static_cast<std::int64_t>(*m);
        m_ilp64 = &m_value;
    }
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t k_value = 0;
    const std::int64_t *k_ilp64 = nullptr;
    if (k != nullptr) {
        k_value = static_cast<std::int64_t>(*k);
        k_ilp64 = &k_value;
    }
    std::int64_t lda_value = 0;
    const std::int64_t *lda_ilp64 = nullptr;
    if (lda != nullptr) {
        lda_value = static_cast<std::int64_t>(*lda);
        lda_ilp64 = &lda_value;
    }
    std::int64_t ldb_value = 0;
    const std::int64_t *ldb_ilp64 = nullptr;
    if (ldb != nullptr) {
        ldb_value = static_cast<std::int64_t>(*ldb);
        ldb_ilp64 = &ldb_value;
    }
    std::int64_t ldc_value = 0;
    const std::int64_t *ldc_ilp64 = nullptr;
    if (ldc != nullptr) {
        ldc_value = static_cast<std::int64_t>(*ldc);
        ldc_ilp64 = &ldc_value;
    }

    reidblas::ilp64::dgemm(transa, transb, m_ilp64, n_ilp64, k_ilp64, alpha, a, lda_ilp64, b, ldb_ilp64, beta, c, ldc_ilp64);
}

extern "C" void dsymm_(
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
) {
    std::int64_t m_value = 0;
    const std::int64_t *m_ilp64 = nullptr;
    if (m != nullptr) {
        m_value = static_cast<std::int64_t>(*m);
        m_ilp64 = &m_value;
    }
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t lda_value = 0;
    const std::int64_t *lda_ilp64 = nullptr;
    if (lda != nullptr) {
        lda_value = static_cast<std::int64_t>(*lda);
        lda_ilp64 = &lda_value;
    }
    std::int64_t ldb_value = 0;
    const std::int64_t *ldb_ilp64 = nullptr;
    if (ldb != nullptr) {
        ldb_value = static_cast<std::int64_t>(*ldb);
        ldb_ilp64 = &ldb_value;
    }
    std::int64_t ldc_value = 0;
    const std::int64_t *ldc_ilp64 = nullptr;
    if (ldc != nullptr) {
        ldc_value = static_cast<std::int64_t>(*ldc);
        ldc_ilp64 = &ldc_value;
    }

    reidblas::ilp64::dsymm(side, uplo, m_ilp64, n_ilp64, alpha, a, lda_ilp64, b, ldb_ilp64, beta, c, ldc_ilp64);
}

extern "C" void dsyrk_(
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
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t k_value = 0;
    const std::int64_t *k_ilp64 = nullptr;
    if (k != nullptr) {
        k_value = static_cast<std::int64_t>(*k);
        k_ilp64 = &k_value;
    }
    std::int64_t lda_value = 0;
    const std::int64_t *lda_ilp64 = nullptr;
    if (lda != nullptr) {
        lda_value = static_cast<std::int64_t>(*lda);
        lda_ilp64 = &lda_value;
    }
    std::int64_t ldc_value = 0;
    const std::int64_t *ldc_ilp64 = nullptr;
    if (ldc != nullptr) {
        ldc_value = static_cast<std::int64_t>(*ldc);
        ldc_ilp64 = &ldc_value;
    }

    reidblas::ilp64::dsyrk(uplo, trans, n_ilp64, k_ilp64, alpha, a, lda_ilp64, beta, c, ldc_ilp64);
}

extern "C" void dsyr2k_(
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
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t k_value = 0;
    const std::int64_t *k_ilp64 = nullptr;
    if (k != nullptr) {
        k_value = static_cast<std::int64_t>(*k);
        k_ilp64 = &k_value;
    }
    std::int64_t lda_value = 0;
    const std::int64_t *lda_ilp64 = nullptr;
    if (lda != nullptr) {
        lda_value = static_cast<std::int64_t>(*lda);
        lda_ilp64 = &lda_value;
    }
    std::int64_t ldb_value = 0;
    const std::int64_t *ldb_ilp64 = nullptr;
    if (ldb != nullptr) {
        ldb_value = static_cast<std::int64_t>(*ldb);
        ldb_ilp64 = &ldb_value;
    }
    std::int64_t ldc_value = 0;
    const std::int64_t *ldc_ilp64 = nullptr;
    if (ldc != nullptr) {
        ldc_value = static_cast<std::int64_t>(*ldc);
        ldc_ilp64 = &ldc_value;
    }

    reidblas::ilp64::dsyr2k(uplo, trans, n_ilp64, k_ilp64, alpha, a, lda_ilp64, b, ldb_ilp64, beta, c, ldc_ilp64);
}

extern "C" void dtrsm_(
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
) {
    std::int64_t m_value = 0;
    const std::int64_t *m_ilp64 = nullptr;
    if (m != nullptr) {
        m_value = static_cast<std::int64_t>(*m);
        m_ilp64 = &m_value;
    }
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t lda_value = 0;
    const std::int64_t *lda_ilp64 = nullptr;
    if (lda != nullptr) {
        lda_value = static_cast<std::int64_t>(*lda);
        lda_ilp64 = &lda_value;
    }
    std::int64_t ldb_value = 0;
    const std::int64_t *ldb_ilp64 = nullptr;
    if (ldb != nullptr) {
        ldb_value = static_cast<std::int64_t>(*ldb);
        ldb_ilp64 = &ldb_value;
    }

    reidblas::ilp64::dtrsm(side, uplo, transa, diag, m_ilp64, n_ilp64, alpha, a, lda_ilp64, b, ldb_ilp64);
}

extern "C" void dtrmm_(
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
) {
    std::int64_t m_value = 0;
    const std::int64_t *m_ilp64 = nullptr;
    if (m != nullptr) {
        m_value = static_cast<std::int64_t>(*m);
        m_ilp64 = &m_value;
    }
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t lda_value = 0;
    const std::int64_t *lda_ilp64 = nullptr;
    if (lda != nullptr) {
        lda_value = static_cast<std::int64_t>(*lda);
        lda_ilp64 = &lda_value;
    }
    std::int64_t ldb_value = 0;
    const std::int64_t *ldb_ilp64 = nullptr;
    if (ldb != nullptr) {
        ldb_value = static_cast<std::int64_t>(*ldb);
        ldb_ilp64 = &ldb_value;
    }

    reidblas::ilp64::dtrmm(side, uplo, transa, diag, m_ilp64, n_ilp64, alpha, a, lda_ilp64, b, ldb_ilp64);
}

extern "C" void cgemm_(
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
) {
    std::int64_t m_value = 0;
    const std::int64_t *m_ilp64 = nullptr;
    if (m != nullptr) {
        m_value = static_cast<std::int64_t>(*m);
        m_ilp64 = &m_value;
    }
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t k_value = 0;
    const std::int64_t *k_ilp64 = nullptr;
    if (k != nullptr) {
        k_value = static_cast<std::int64_t>(*k);
        k_ilp64 = &k_value;
    }
    std::int64_t lda_value = 0;
    const std::int64_t *lda_ilp64 = nullptr;
    if (lda != nullptr) {
        lda_value = static_cast<std::int64_t>(*lda);
        lda_ilp64 = &lda_value;
    }
    std::int64_t ldb_value = 0;
    const std::int64_t *ldb_ilp64 = nullptr;
    if (ldb != nullptr) {
        ldb_value = static_cast<std::int64_t>(*ldb);
        ldb_ilp64 = &ldb_value;
    }
    std::int64_t ldc_value = 0;
    const std::int64_t *ldc_ilp64 = nullptr;
    if (ldc != nullptr) {
        ldc_value = static_cast<std::int64_t>(*ldc);
        ldc_ilp64 = &ldc_value;
    }

    reidblas::ilp64::cgemm(transa, transb, m_ilp64, n_ilp64, k_ilp64, alpha, a, lda_ilp64, b, ldb_ilp64, beta, c, ldc_ilp64);
}

extern "C" void csymm_(
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
) {
    std::int64_t m_value = 0;
    const std::int64_t *m_ilp64 = nullptr;
    if (m != nullptr) {
        m_value = static_cast<std::int64_t>(*m);
        m_ilp64 = &m_value;
    }
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t lda_value = 0;
    const std::int64_t *lda_ilp64 = nullptr;
    if (lda != nullptr) {
        lda_value = static_cast<std::int64_t>(*lda);
        lda_ilp64 = &lda_value;
    }
    std::int64_t ldb_value = 0;
    const std::int64_t *ldb_ilp64 = nullptr;
    if (ldb != nullptr) {
        ldb_value = static_cast<std::int64_t>(*ldb);
        ldb_ilp64 = &ldb_value;
    }
    std::int64_t ldc_value = 0;
    const std::int64_t *ldc_ilp64 = nullptr;
    if (ldc != nullptr) {
        ldc_value = static_cast<std::int64_t>(*ldc);
        ldc_ilp64 = &ldc_value;
    }

    reidblas::ilp64::csymm(side, uplo, m_ilp64, n_ilp64, alpha, a, lda_ilp64, b, ldb_ilp64, beta, c, ldc_ilp64);
}

extern "C" void chemm_(
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
) {
    std::int64_t m_value = 0;
    const std::int64_t *m_ilp64 = nullptr;
    if (m != nullptr) {
        m_value = static_cast<std::int64_t>(*m);
        m_ilp64 = &m_value;
    }
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t lda_value = 0;
    const std::int64_t *lda_ilp64 = nullptr;
    if (lda != nullptr) {
        lda_value = static_cast<std::int64_t>(*lda);
        lda_ilp64 = &lda_value;
    }
    std::int64_t ldb_value = 0;
    const std::int64_t *ldb_ilp64 = nullptr;
    if (ldb != nullptr) {
        ldb_value = static_cast<std::int64_t>(*ldb);
        ldb_ilp64 = &ldb_value;
    }
    std::int64_t ldc_value = 0;
    const std::int64_t *ldc_ilp64 = nullptr;
    if (ldc != nullptr) {
        ldc_value = static_cast<std::int64_t>(*ldc);
        ldc_ilp64 = &ldc_value;
    }

    reidblas::ilp64::chemm(side, uplo, m_ilp64, n_ilp64, alpha, a, lda_ilp64, b, ldb_ilp64, beta, c, ldc_ilp64);
}

extern "C" void csyrk_(
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
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t k_value = 0;
    const std::int64_t *k_ilp64 = nullptr;
    if (k != nullptr) {
        k_value = static_cast<std::int64_t>(*k);
        k_ilp64 = &k_value;
    }
    std::int64_t lda_value = 0;
    const std::int64_t *lda_ilp64 = nullptr;
    if (lda != nullptr) {
        lda_value = static_cast<std::int64_t>(*lda);
        lda_ilp64 = &lda_value;
    }
    std::int64_t ldc_value = 0;
    const std::int64_t *ldc_ilp64 = nullptr;
    if (ldc != nullptr) {
        ldc_value = static_cast<std::int64_t>(*ldc);
        ldc_ilp64 = &ldc_value;
    }

    reidblas::ilp64::csyrk(uplo, trans, n_ilp64, k_ilp64, alpha, a, lda_ilp64, beta, c, ldc_ilp64);
}

extern "C" void cherk_(
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
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t k_value = 0;
    const std::int64_t *k_ilp64 = nullptr;
    if (k != nullptr) {
        k_value = static_cast<std::int64_t>(*k);
        k_ilp64 = &k_value;
    }
    std::int64_t lda_value = 0;
    const std::int64_t *lda_ilp64 = nullptr;
    if (lda != nullptr) {
        lda_value = static_cast<std::int64_t>(*lda);
        lda_ilp64 = &lda_value;
    }
    std::int64_t ldc_value = 0;
    const std::int64_t *ldc_ilp64 = nullptr;
    if (ldc != nullptr) {
        ldc_value = static_cast<std::int64_t>(*ldc);
        ldc_ilp64 = &ldc_value;
    }

    reidblas::ilp64::cherk(uplo, trans, n_ilp64, k_ilp64, alpha, a, lda_ilp64, beta, c, ldc_ilp64);
}

extern "C" void csyr2k_(
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
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t k_value = 0;
    const std::int64_t *k_ilp64 = nullptr;
    if (k != nullptr) {
        k_value = static_cast<std::int64_t>(*k);
        k_ilp64 = &k_value;
    }
    std::int64_t lda_value = 0;
    const std::int64_t *lda_ilp64 = nullptr;
    if (lda != nullptr) {
        lda_value = static_cast<std::int64_t>(*lda);
        lda_ilp64 = &lda_value;
    }
    std::int64_t ldb_value = 0;
    const std::int64_t *ldb_ilp64 = nullptr;
    if (ldb != nullptr) {
        ldb_value = static_cast<std::int64_t>(*ldb);
        ldb_ilp64 = &ldb_value;
    }
    std::int64_t ldc_value = 0;
    const std::int64_t *ldc_ilp64 = nullptr;
    if (ldc != nullptr) {
        ldc_value = static_cast<std::int64_t>(*ldc);
        ldc_ilp64 = &ldc_value;
    }

    reidblas::ilp64::csyr2k(uplo, trans, n_ilp64, k_ilp64, alpha, a, lda_ilp64, b, ldb_ilp64, beta, c, ldc_ilp64);
}

extern "C" void cher2k_(
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
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t k_value = 0;
    const std::int64_t *k_ilp64 = nullptr;
    if (k != nullptr) {
        k_value = static_cast<std::int64_t>(*k);
        k_ilp64 = &k_value;
    }
    std::int64_t lda_value = 0;
    const std::int64_t *lda_ilp64 = nullptr;
    if (lda != nullptr) {
        lda_value = static_cast<std::int64_t>(*lda);
        lda_ilp64 = &lda_value;
    }
    std::int64_t ldb_value = 0;
    const std::int64_t *ldb_ilp64 = nullptr;
    if (ldb != nullptr) {
        ldb_value = static_cast<std::int64_t>(*ldb);
        ldb_ilp64 = &ldb_value;
    }
    std::int64_t ldc_value = 0;
    const std::int64_t *ldc_ilp64 = nullptr;
    if (ldc != nullptr) {
        ldc_value = static_cast<std::int64_t>(*ldc);
        ldc_ilp64 = &ldc_value;
    }

    reidblas::ilp64::cher2k(uplo, trans, n_ilp64, k_ilp64, alpha, a, lda_ilp64, b, ldb_ilp64, beta, c, ldc_ilp64);
}

extern "C" void ctrsm_(
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
) {
    std::int64_t m_value = 0;
    const std::int64_t *m_ilp64 = nullptr;
    if (m != nullptr) {
        m_value = static_cast<std::int64_t>(*m);
        m_ilp64 = &m_value;
    }
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t lda_value = 0;
    const std::int64_t *lda_ilp64 = nullptr;
    if (lda != nullptr) {
        lda_value = static_cast<std::int64_t>(*lda);
        lda_ilp64 = &lda_value;
    }
    std::int64_t ldb_value = 0;
    const std::int64_t *ldb_ilp64 = nullptr;
    if (ldb != nullptr) {
        ldb_value = static_cast<std::int64_t>(*ldb);
        ldb_ilp64 = &ldb_value;
    }

    reidblas::ilp64::ctrsm(side, uplo, transa, diag, m_ilp64, n_ilp64, alpha, a, lda_ilp64, b, ldb_ilp64);
}

extern "C" void ctrmm_(
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
) {
    std::int64_t m_value = 0;
    const std::int64_t *m_ilp64 = nullptr;
    if (m != nullptr) {
        m_value = static_cast<std::int64_t>(*m);
        m_ilp64 = &m_value;
    }
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t lda_value = 0;
    const std::int64_t *lda_ilp64 = nullptr;
    if (lda != nullptr) {
        lda_value = static_cast<std::int64_t>(*lda);
        lda_ilp64 = &lda_value;
    }
    std::int64_t ldb_value = 0;
    const std::int64_t *ldb_ilp64 = nullptr;
    if (ldb != nullptr) {
        ldb_value = static_cast<std::int64_t>(*ldb);
        ldb_ilp64 = &ldb_value;
    }

    reidblas::ilp64::ctrmm(side, uplo, transa, diag, m_ilp64, n_ilp64, alpha, a, lda_ilp64, b, ldb_ilp64);
}

extern "C" void zgemm_(
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
) {
    std::int64_t m_value = 0;
    const std::int64_t *m_ilp64 = nullptr;
    if (m != nullptr) {
        m_value = static_cast<std::int64_t>(*m);
        m_ilp64 = &m_value;
    }
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t k_value = 0;
    const std::int64_t *k_ilp64 = nullptr;
    if (k != nullptr) {
        k_value = static_cast<std::int64_t>(*k);
        k_ilp64 = &k_value;
    }
    std::int64_t lda_value = 0;
    const std::int64_t *lda_ilp64 = nullptr;
    if (lda != nullptr) {
        lda_value = static_cast<std::int64_t>(*lda);
        lda_ilp64 = &lda_value;
    }
    std::int64_t ldb_value = 0;
    const std::int64_t *ldb_ilp64 = nullptr;
    if (ldb != nullptr) {
        ldb_value = static_cast<std::int64_t>(*ldb);
        ldb_ilp64 = &ldb_value;
    }
    std::int64_t ldc_value = 0;
    const std::int64_t *ldc_ilp64 = nullptr;
    if (ldc != nullptr) {
        ldc_value = static_cast<std::int64_t>(*ldc);
        ldc_ilp64 = &ldc_value;
    }

    reidblas::ilp64::zgemm(transa, transb, m_ilp64, n_ilp64, k_ilp64, alpha, a, lda_ilp64, b, ldb_ilp64, beta, c, ldc_ilp64);
}

extern "C" void zsymm_(
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
) {
    std::int64_t m_value = 0;
    const std::int64_t *m_ilp64 = nullptr;
    if (m != nullptr) {
        m_value = static_cast<std::int64_t>(*m);
        m_ilp64 = &m_value;
    }
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t lda_value = 0;
    const std::int64_t *lda_ilp64 = nullptr;
    if (lda != nullptr) {
        lda_value = static_cast<std::int64_t>(*lda);
        lda_ilp64 = &lda_value;
    }
    std::int64_t ldb_value = 0;
    const std::int64_t *ldb_ilp64 = nullptr;
    if (ldb != nullptr) {
        ldb_value = static_cast<std::int64_t>(*ldb);
        ldb_ilp64 = &ldb_value;
    }
    std::int64_t ldc_value = 0;
    const std::int64_t *ldc_ilp64 = nullptr;
    if (ldc != nullptr) {
        ldc_value = static_cast<std::int64_t>(*ldc);
        ldc_ilp64 = &ldc_value;
    }

    reidblas::ilp64::zsymm(side, uplo, m_ilp64, n_ilp64, alpha, a, lda_ilp64, b, ldb_ilp64, beta, c, ldc_ilp64);
}

extern "C" void zhemm_(
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
) {
    std::int64_t m_value = 0;
    const std::int64_t *m_ilp64 = nullptr;
    if (m != nullptr) {
        m_value = static_cast<std::int64_t>(*m);
        m_ilp64 = &m_value;
    }
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t lda_value = 0;
    const std::int64_t *lda_ilp64 = nullptr;
    if (lda != nullptr) {
        lda_value = static_cast<std::int64_t>(*lda);
        lda_ilp64 = &lda_value;
    }
    std::int64_t ldb_value = 0;
    const std::int64_t *ldb_ilp64 = nullptr;
    if (ldb != nullptr) {
        ldb_value = static_cast<std::int64_t>(*ldb);
        ldb_ilp64 = &ldb_value;
    }
    std::int64_t ldc_value = 0;
    const std::int64_t *ldc_ilp64 = nullptr;
    if (ldc != nullptr) {
        ldc_value = static_cast<std::int64_t>(*ldc);
        ldc_ilp64 = &ldc_value;
    }

    reidblas::ilp64::zhemm(side, uplo, m_ilp64, n_ilp64, alpha, a, lda_ilp64, b, ldb_ilp64, beta, c, ldc_ilp64);
}

extern "C" void zsyrk_(
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
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t k_value = 0;
    const std::int64_t *k_ilp64 = nullptr;
    if (k != nullptr) {
        k_value = static_cast<std::int64_t>(*k);
        k_ilp64 = &k_value;
    }
    std::int64_t lda_value = 0;
    const std::int64_t *lda_ilp64 = nullptr;
    if (lda != nullptr) {
        lda_value = static_cast<std::int64_t>(*lda);
        lda_ilp64 = &lda_value;
    }
    std::int64_t ldc_value = 0;
    const std::int64_t *ldc_ilp64 = nullptr;
    if (ldc != nullptr) {
        ldc_value = static_cast<std::int64_t>(*ldc);
        ldc_ilp64 = &ldc_value;
    }

    reidblas::ilp64::zsyrk(uplo, trans, n_ilp64, k_ilp64, alpha, a, lda_ilp64, beta, c, ldc_ilp64);
}

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
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t k_value = 0;
    const std::int64_t *k_ilp64 = nullptr;
    if (k != nullptr) {
        k_value = static_cast<std::int64_t>(*k);
        k_ilp64 = &k_value;
    }
    std::int64_t lda_value = 0;
    const std::int64_t *lda_ilp64 = nullptr;
    if (lda != nullptr) {
        lda_value = static_cast<std::int64_t>(*lda);
        lda_ilp64 = &lda_value;
    }
    std::int64_t ldc_value = 0;
    const std::int64_t *ldc_ilp64 = nullptr;
    if (ldc != nullptr) {
        ldc_value = static_cast<std::int64_t>(*ldc);
        ldc_ilp64 = &ldc_value;
    }

    reidblas::ilp64::zherk(uplo, trans, n_ilp64, k_ilp64, alpha, a, lda_ilp64, beta, c, ldc_ilp64);
}

extern "C" void zsyr2k_(
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
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t k_value = 0;
    const std::int64_t *k_ilp64 = nullptr;
    if (k != nullptr) {
        k_value = static_cast<std::int64_t>(*k);
        k_ilp64 = &k_value;
    }
    std::int64_t lda_value = 0;
    const std::int64_t *lda_ilp64 = nullptr;
    if (lda != nullptr) {
        lda_value = static_cast<std::int64_t>(*lda);
        lda_ilp64 = &lda_value;
    }
    std::int64_t ldb_value = 0;
    const std::int64_t *ldb_ilp64 = nullptr;
    if (ldb != nullptr) {
        ldb_value = static_cast<std::int64_t>(*ldb);
        ldb_ilp64 = &ldb_value;
    }
    std::int64_t ldc_value = 0;
    const std::int64_t *ldc_ilp64 = nullptr;
    if (ldc != nullptr) {
        ldc_value = static_cast<std::int64_t>(*ldc);
        ldc_ilp64 = &ldc_value;
    }

    reidblas::ilp64::zsyr2k(uplo, trans, n_ilp64, k_ilp64, alpha, a, lda_ilp64, b, ldb_ilp64, beta, c, ldc_ilp64);
}

extern "C" void zher2k_(
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
) {
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t k_value = 0;
    const std::int64_t *k_ilp64 = nullptr;
    if (k != nullptr) {
        k_value = static_cast<std::int64_t>(*k);
        k_ilp64 = &k_value;
    }
    std::int64_t lda_value = 0;
    const std::int64_t *lda_ilp64 = nullptr;
    if (lda != nullptr) {
        lda_value = static_cast<std::int64_t>(*lda);
        lda_ilp64 = &lda_value;
    }
    std::int64_t ldb_value = 0;
    const std::int64_t *ldb_ilp64 = nullptr;
    if (ldb != nullptr) {
        ldb_value = static_cast<std::int64_t>(*ldb);
        ldb_ilp64 = &ldb_value;
    }
    std::int64_t ldc_value = 0;
    const std::int64_t *ldc_ilp64 = nullptr;
    if (ldc != nullptr) {
        ldc_value = static_cast<std::int64_t>(*ldc);
        ldc_ilp64 = &ldc_value;
    }

    reidblas::ilp64::zher2k(uplo, trans, n_ilp64, k_ilp64, alpha, a, lda_ilp64, b, ldb_ilp64, beta, c, ldc_ilp64);
}

extern "C" void ztrsm_(
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
) {
    std::int64_t m_value = 0;
    const std::int64_t *m_ilp64 = nullptr;
    if (m != nullptr) {
        m_value = static_cast<std::int64_t>(*m);
        m_ilp64 = &m_value;
    }
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t lda_value = 0;
    const std::int64_t *lda_ilp64 = nullptr;
    if (lda != nullptr) {
        lda_value = static_cast<std::int64_t>(*lda);
        lda_ilp64 = &lda_value;
    }
    std::int64_t ldb_value = 0;
    const std::int64_t *ldb_ilp64 = nullptr;
    if (ldb != nullptr) {
        ldb_value = static_cast<std::int64_t>(*ldb);
        ldb_ilp64 = &ldb_value;
    }

    reidblas::ilp64::ztrsm(side, uplo, transa, diag, m_ilp64, n_ilp64, alpha, a, lda_ilp64, b, ldb_ilp64);
}

extern "C" void ztrmm_(
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
) {
    std::int64_t m_value = 0;
    const std::int64_t *m_ilp64 = nullptr;
    if (m != nullptr) {
        m_value = static_cast<std::int64_t>(*m);
        m_ilp64 = &m_value;
    }
    std::int64_t n_value = 0;
    const std::int64_t *n_ilp64 = nullptr;
    if (n != nullptr) {
        n_value = static_cast<std::int64_t>(*n);
        n_ilp64 = &n_value;
    }
    std::int64_t lda_value = 0;
    const std::int64_t *lda_ilp64 = nullptr;
    if (lda != nullptr) {
        lda_value = static_cast<std::int64_t>(*lda);
        lda_ilp64 = &lda_value;
    }
    std::int64_t ldb_value = 0;
    const std::int64_t *ldb_ilp64 = nullptr;
    if (ldb != nullptr) {
        ldb_value = static_cast<std::int64_t>(*ldb);
        ldb_ilp64 = &ldb_value;
    }

    reidblas::ilp64::ztrmm(side, uplo, transa, diag, m_ilp64, n_ilp64, alpha, a, lda_ilp64, b, ldb_ilp64);
}
