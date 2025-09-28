#include "compensated_blas_lp64.h"
#include "impl/compensated_blas_backend_ilp64.hpp"

#include <cstdint>

namespace {

inline compensated_blas::impl::BlasBackend &backend() {
    return compensated_blas::impl::get_active_backend();
}

}  // namespace

extern "C" void srotg_(
    float *a,
    float *b,
    float *c,
    float *s
) {
    backend().srotg(a, b, c, s);
}

extern "C" void srotmg_(
    float *d1,
    float *d2,
    float *x1,
    const float *y1,
    float *param
) {
    backend().srotmg(d1, d2, x1, y1, param);
}

extern "C" void srot_(
    const compensated_blas_blas_int *n,
    float *x,
    const compensated_blas_blas_int *incx,
    float *y,
    const compensated_blas_blas_int *incy,
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

    backend().srot(n_ilp64, x, incx_ilp64, y, incy_ilp64, c, s);
}

extern "C" void srotm_(
    const compensated_blas_blas_int *n,
    float *x,
    const compensated_blas_blas_int *incx,
    float *y,
    const compensated_blas_blas_int *incy,
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

    backend().srotm(n_ilp64, x, incx_ilp64, y, incy_ilp64, param);
}

extern "C" void sswap_(
    const compensated_blas_blas_int *n,
    float *x,
    const compensated_blas_blas_int *incx,
    float *y,
    const compensated_blas_blas_int *incy
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

    backend().sswap(n_ilp64, x, incx_ilp64, y, incy_ilp64);
}

extern "C" void sscal_(
    const compensated_blas_blas_int *n,
    const float *alpha,
    float *x,
    const compensated_blas_blas_int *incx
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

    backend().sscal(n_ilp64, alpha, x, incx_ilp64);
}

extern "C" void scopy_(
    const compensated_blas_blas_int *n,
    const float *x,
    const compensated_blas_blas_int *incx,
    float *y,
    const compensated_blas_blas_int *incy
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

    backend().scopy(n_ilp64, x, incx_ilp64, y, incy_ilp64);
}

extern "C" void saxpy_(
    const compensated_blas_blas_int *n,
    const float *alpha,
    const float *x,
    const compensated_blas_blas_int *incx,
    float *y,
    const compensated_blas_blas_int *incy
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

    backend().saxpy(n_ilp64, alpha, x, incx_ilp64, y, incy_ilp64);
}

extern "C" float sdot_(
    const compensated_blas_blas_int *n,
    const float *x,
    const compensated_blas_blas_int *incx,
    const float *y,
    const compensated_blas_blas_int *incy
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

    return backend().sdot(n_ilp64, x, incx_ilp64, y, incy_ilp64);
}

extern "C" float sdsdot_(
    const compensated_blas_blas_int *n,
    const float *sb,
    const float *x,
    const compensated_blas_blas_int *incx,
    const float *y,
    const compensated_blas_blas_int *incy
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

    return backend().sdsdot(n_ilp64, sb, x, incx_ilp64, y, incy_ilp64);
}

extern "C" float snrm2_(
    const compensated_blas_blas_int *n,
    const float *x,
    const compensated_blas_blas_int *incx
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

    return backend().snrm2(n_ilp64, x, incx_ilp64);
}

extern "C" float sasum_(
    const compensated_blas_blas_int *n,
    const float *x,
    const compensated_blas_blas_int *incx
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

    return backend().sasum(n_ilp64, x, incx_ilp64);
}

extern "C" compensated_blas_blas_int isamax_(
    const compensated_blas_blas_int *n,
    const float *x,
    const compensated_blas_blas_int *incx
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

    auto result = backend().isamax(n_ilp64, x, incx_ilp64);
    return static_cast<compensated_blas_blas_int>(result);
}

extern "C" void drotg_(
    double *a,
    double *b,
    double *c,
    double *s
) {
    backend().drotg(a, b, c, s);
}

extern "C" void drotmg_(
    double *d1,
    double *d2,
    double *x1,
    const double *y1,
    double *param
) {
    backend().drotmg(d1, d2, x1, y1, param);
}

extern "C" void drot_(
    const compensated_blas_blas_int *n,
    double *x,
    const compensated_blas_blas_int *incx,
    double *y,
    const compensated_blas_blas_int *incy,
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

    backend().drot(n_ilp64, x, incx_ilp64, y, incy_ilp64, c, s);
}

extern "C" void drotm_(
    const compensated_blas_blas_int *n,
    double *x,
    const compensated_blas_blas_int *incx,
    double *y,
    const compensated_blas_blas_int *incy,
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

    backend().drotm(n_ilp64, x, incx_ilp64, y, incy_ilp64, param);
}

extern "C" void dswap_(
    const compensated_blas_blas_int *n,
    double *x,
    const compensated_blas_blas_int *incx,
    double *y,
    const compensated_blas_blas_int *incy
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

    backend().dswap(n_ilp64, x, incx_ilp64, y, incy_ilp64);
}

extern "C" void dscal_(
    const compensated_blas_blas_int *n,
    const double *alpha,
    double *x,
    const compensated_blas_blas_int *incx
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

    backend().dscal(n_ilp64, alpha, x, incx_ilp64);
}

extern "C" void dcopy_(
    const compensated_blas_blas_int *n,
    const double *x,
    const compensated_blas_blas_int *incx,
    double *y,
    const compensated_blas_blas_int *incy
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

    backend().dcopy(n_ilp64, x, incx_ilp64, y, incy_ilp64);
}

extern "C" void daxpy_(
    const compensated_blas_blas_int *n,
    const double *alpha,
    const double *x,
    const compensated_blas_blas_int *incx,
    double *y,
    const compensated_blas_blas_int *incy
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

    backend().daxpy(n_ilp64, alpha, x, incx_ilp64, y, incy_ilp64);
}

extern "C" double ddot_(
    const compensated_blas_blas_int *n,
    const double *x,
    const compensated_blas_blas_int *incx,
    const double *y,
    const compensated_blas_blas_int *incy
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

    return backend().ddot(n_ilp64, x, incx_ilp64, y, incy_ilp64);
}

extern "C" double dsdot_(
    const compensated_blas_blas_int *n,
    const float *x,
    const compensated_blas_blas_int *incx,
    const float *y,
    const compensated_blas_blas_int *incy
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

    return backend().dsdot(n_ilp64, x, incx_ilp64, y, incy_ilp64);
}

extern "C" double dnrm2_(
    const compensated_blas_blas_int *n,
    const double *x,
    const compensated_blas_blas_int *incx
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

    return backend().dnrm2(n_ilp64, x, incx_ilp64);
}

extern "C" double dasum_(
    const compensated_blas_blas_int *n,
    const double *x,
    const compensated_blas_blas_int *incx
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

    return backend().dasum(n_ilp64, x, incx_ilp64);
}

extern "C" compensated_blas_blas_int idamax_(
    const compensated_blas_blas_int *n,
    const double *x,
    const compensated_blas_blas_int *incx
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

    auto result = backend().idamax(n_ilp64, x, incx_ilp64);
    return static_cast<compensated_blas_blas_int>(result);
}

extern "C" void crotg_(
    compensated_blas_complex_float *a,
    const compensated_blas_complex_float *b,
    float *c,
    compensated_blas_complex_float *s
) {
    backend().crotg(a, b, c, s);
}

extern "C" void csrot_(
    const compensated_blas_blas_int *n,
    compensated_blas_complex_float *x,
    const compensated_blas_blas_int *incx,
    compensated_blas_complex_float *y,
    const compensated_blas_blas_int *incy,
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

    backend().csrot(n_ilp64, x, incx_ilp64, y, incy_ilp64, c, s);
}

extern "C" void csscal_(
    const compensated_blas_blas_int *n,
    const float *alpha,
    compensated_blas_complex_float *x,
    const compensated_blas_blas_int *incx
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

    backend().csscal(n_ilp64, alpha, x, incx_ilp64);
}

extern "C" void cscal_(
    const compensated_blas_blas_int *n,
    const compensated_blas_complex_float *alpha,
    compensated_blas_complex_float *x,
    const compensated_blas_blas_int *incx
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

    backend().cscal(n_ilp64, alpha, x, incx_ilp64);
}

extern "C" void cswap_(
    const compensated_blas_blas_int *n,
    compensated_blas_complex_float *x,
    const compensated_blas_blas_int *incx,
    compensated_blas_complex_float *y,
    const compensated_blas_blas_int *incy
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

    backend().cswap(n_ilp64, x, incx_ilp64, y, incy_ilp64);
}

extern "C" void ccopy_(
    const compensated_blas_blas_int *n,
    const compensated_blas_complex_float *x,
    const compensated_blas_blas_int *incx,
    compensated_blas_complex_float *y,
    const compensated_blas_blas_int *incy
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

    backend().ccopy(n_ilp64, x, incx_ilp64, y, incy_ilp64);
}

extern "C" void caxpy_(
    const compensated_blas_blas_int *n,
    const compensated_blas_complex_float *alpha,
    const compensated_blas_complex_float *x,
    const compensated_blas_blas_int *incx,
    compensated_blas_complex_float *y,
    const compensated_blas_blas_int *incy
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

    backend().caxpy(n_ilp64, alpha, x, incx_ilp64, y, incy_ilp64);
}

extern "C" compensated_blas_complex_float cdotu_(
    const compensated_blas_blas_int *n,
    const compensated_blas_complex_float *x,
    const compensated_blas_blas_int *incx,
    const compensated_blas_complex_float *y,
    const compensated_blas_blas_int *incy
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

    return backend().cdotu(n_ilp64, x, incx_ilp64, y, incy_ilp64);
}

extern "C" compensated_blas_complex_float cdotc_(
    const compensated_blas_blas_int *n,
    const compensated_blas_complex_float *x,
    const compensated_blas_blas_int *incx,
    const compensated_blas_complex_float *y,
    const compensated_blas_blas_int *incy
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

    return backend().cdotc(n_ilp64, x, incx_ilp64, y, incy_ilp64);
}

extern "C" float scnrm2_(
    const compensated_blas_blas_int *n,
    const compensated_blas_complex_float *x,
    const compensated_blas_blas_int *incx
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

    return backend().scnrm2(n_ilp64, x, incx_ilp64);
}

extern "C" float scasum_(
    const compensated_blas_blas_int *n,
    const compensated_blas_complex_float *x,
    const compensated_blas_blas_int *incx
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

    return backend().scasum(n_ilp64, x, incx_ilp64);
}

extern "C" compensated_blas_blas_int icamax_(
    const compensated_blas_blas_int *n,
    const compensated_blas_complex_float *x,
    const compensated_blas_blas_int *incx
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

    auto result = backend().icamax(n_ilp64, x, incx_ilp64);
    return static_cast<compensated_blas_blas_int>(result);
}

extern "C" void zrotg_(
    compensated_blas_complex_double *a,
    const compensated_blas_complex_double *b,
    double *c,
    compensated_blas_complex_double *s
) {
    backend().zrotg(a, b, c, s);
}

extern "C" void zdrot_(
    const compensated_blas_blas_int *n,
    compensated_blas_complex_double *x,
    const compensated_blas_blas_int *incx,
    compensated_blas_complex_double *y,
    const compensated_blas_blas_int *incy,
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

    backend().zdrot(n_ilp64, x, incx_ilp64, y, incy_ilp64, c, s);
}

extern "C" void zdscal_(
    const compensated_blas_blas_int *n,
    const double *alpha,
    compensated_blas_complex_double *x,
    const compensated_blas_blas_int *incx
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

    backend().zdscal(n_ilp64, alpha, x, incx_ilp64);
}

extern "C" void zscal_(
    const compensated_blas_blas_int *n,
    const compensated_blas_complex_double *alpha,
    compensated_blas_complex_double *x,
    const compensated_blas_blas_int *incx
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

    backend().zscal(n_ilp64, alpha, x, incx_ilp64);
}

extern "C" void zswap_(
    const compensated_blas_blas_int *n,
    compensated_blas_complex_double *x,
    const compensated_blas_blas_int *incx,
    compensated_blas_complex_double *y,
    const compensated_blas_blas_int *incy
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

    backend().zswap(n_ilp64, x, incx_ilp64, y, incy_ilp64);
}

extern "C" void zcopy_(
    const compensated_blas_blas_int *n,
    const compensated_blas_complex_double *x,
    const compensated_blas_blas_int *incx,
    compensated_blas_complex_double *y,
    const compensated_blas_blas_int *incy
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

    backend().zcopy(n_ilp64, x, incx_ilp64, y, incy_ilp64);
}

extern "C" void zaxpy_(
    const compensated_blas_blas_int *n,
    const compensated_blas_complex_double *alpha,
    const compensated_blas_complex_double *x,
    const compensated_blas_blas_int *incx,
    compensated_blas_complex_double *y,
    const compensated_blas_blas_int *incy
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

    backend().zaxpy(n_ilp64, alpha, x, incx_ilp64, y, incy_ilp64);
}

extern "C" compensated_blas_complex_double zdotu_(
    const compensated_blas_blas_int *n,
    const compensated_blas_complex_double *x,
    const compensated_blas_blas_int *incx,
    const compensated_blas_complex_double *y,
    const compensated_blas_blas_int *incy
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

    return backend().zdotu(n_ilp64, x, incx_ilp64, y, incy_ilp64);
}

extern "C" compensated_blas_complex_double zdotc_(
    const compensated_blas_blas_int *n,
    const compensated_blas_complex_double *x,
    const compensated_blas_blas_int *incx,
    const compensated_blas_complex_double *y,
    const compensated_blas_blas_int *incy
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

    return backend().zdotc(n_ilp64, x, incx_ilp64, y, incy_ilp64);
}

extern "C" double dznrm2_(
    const compensated_blas_blas_int *n,
    const compensated_blas_complex_double *x,
    const compensated_blas_blas_int *incx
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

    return backend().dznrm2(n_ilp64, x, incx_ilp64);
}

extern "C" double dzasum_(
    const compensated_blas_blas_int *n,
    const compensated_blas_complex_double *x,
    const compensated_blas_blas_int *incx
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

    return backend().dzasum(n_ilp64, x, incx_ilp64);
}

extern "C" compensated_blas_blas_int izamax_(
    const compensated_blas_blas_int *n,
    const compensated_blas_complex_double *x,
    const compensated_blas_blas_int *incx
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

    auto result = backend().izamax(n_ilp64, x, incx_ilp64);
    return static_cast<compensated_blas_blas_int>(result);
}

extern "C" void sgemv_(
    const char *trans,
    const compensated_blas_blas_int *m,
    const compensated_blas_blas_int *n,
    const float *alpha,
    const float *a,
    const compensated_blas_blas_int *lda,
    const float *x,
    const compensated_blas_blas_int *incx,
    const float *beta,
    float *y,
    const compensated_blas_blas_int *incy
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

    backend().sgemv(trans, m_ilp64, n_ilp64, alpha, a, lda_ilp64, x, incx_ilp64, beta, y, incy_ilp64);
}

extern "C" void sgbmv_(
    const char *trans,
    const compensated_blas_blas_int *m,
    const compensated_blas_blas_int *n,
    const compensated_blas_blas_int *kl,
    const compensated_blas_blas_int *ku,
    const float *alpha,
    const float *a,
    const compensated_blas_blas_int *lda,
    const float *x,
    const compensated_blas_blas_int *incx,
    const float *beta,
    float *y,
    const compensated_blas_blas_int *incy
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

    backend().sgbmv(trans, m_ilp64, n_ilp64, kl_ilp64, ku_ilp64, alpha, a, lda_ilp64, x, incx_ilp64, beta, y, incy_ilp64);
}

extern "C" void ssymv_(
    const char *uplo,
    const compensated_blas_blas_int *n,
    const float *alpha,
    const float *a,
    const compensated_blas_blas_int *lda,
    const float *x,
    const compensated_blas_blas_int *incx,
    const float *beta,
    float *y,
    const compensated_blas_blas_int *incy
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

    backend().ssymv(uplo, n_ilp64, alpha, a, lda_ilp64, x, incx_ilp64, beta, y, incy_ilp64);
}

extern "C" void ssbmv_(
    const char *uplo,
    const compensated_blas_blas_int *n,
    const compensated_blas_blas_int *k,
    const float *alpha,
    const float *a,
    const compensated_blas_blas_int *lda,
    const float *x,
    const compensated_blas_blas_int *incx,
    const float *beta,
    float *y,
    const compensated_blas_blas_int *incy
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

    backend().ssbmv(uplo, n_ilp64, k_ilp64, alpha, a, lda_ilp64, x, incx_ilp64, beta, y, incy_ilp64);
}

extern "C" void sspmv_(
    const char *uplo,
    const compensated_blas_blas_int *n,
    const float *alpha,
    const float *ap,
    const float *x,
    const compensated_blas_blas_int *incx,
    const float *beta,
    float *y,
    const compensated_blas_blas_int *incy
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

    backend().sspmv(uplo, n_ilp64, alpha, ap, x, incx_ilp64, beta, y, incy_ilp64);
}

extern "C" void strmv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const compensated_blas_blas_int *n,
    const float *a,
    const compensated_blas_blas_int *lda,
    float *x,
    const compensated_blas_blas_int *incx
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

    backend().strmv(uplo, trans, diag, n_ilp64, a, lda_ilp64, x, incx_ilp64);
}

extern "C" void stbmv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const compensated_blas_blas_int *n,
    const compensated_blas_blas_int *k,
    const float *a,
    const compensated_blas_blas_int *lda,
    float *x,
    const compensated_blas_blas_int *incx
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

    backend().stbmv(uplo, trans, diag, n_ilp64, k_ilp64, a, lda_ilp64, x, incx_ilp64);
}

extern "C" void stpmv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const compensated_blas_blas_int *n,
    const float *ap,
    float *x,
    const compensated_blas_blas_int *incx
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

    backend().stpmv(uplo, trans, diag, n_ilp64, ap, x, incx_ilp64);
}

extern "C" void strsv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const compensated_blas_blas_int *n,
    const float *a,
    const compensated_blas_blas_int *lda,
    float *x,
    const compensated_blas_blas_int *incx
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

    backend().strsv(uplo, trans, diag, n_ilp64, a, lda_ilp64, x, incx_ilp64);
}

extern "C" void stbsv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const compensated_blas_blas_int *n,
    const compensated_blas_blas_int *k,
    const float *a,
    const compensated_blas_blas_int *lda,
    float *x,
    const compensated_blas_blas_int *incx
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

    backend().stbsv(uplo, trans, diag, n_ilp64, k_ilp64, a, lda_ilp64, x, incx_ilp64);
}

extern "C" void stpsv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const compensated_blas_blas_int *n,
    const float *ap,
    float *x,
    const compensated_blas_blas_int *incx
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

    backend().stpsv(uplo, trans, diag, n_ilp64, ap, x, incx_ilp64);
}

extern "C" void sger_(
    const compensated_blas_blas_int *m,
    const compensated_blas_blas_int *n,
    const float *alpha,
    const float *x,
    const compensated_blas_blas_int *incx,
    const float *y,
    const compensated_blas_blas_int *incy,
    float *a,
    const compensated_blas_blas_int *lda
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

    backend().sger(m_ilp64, n_ilp64, alpha, x, incx_ilp64, y, incy_ilp64, a, lda_ilp64);
}

extern "C" void sspr_(
    const char *uplo,
    const compensated_blas_blas_int *n,
    const float *alpha,
    const float *x,
    const compensated_blas_blas_int *incx,
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

    backend().sspr(uplo, n_ilp64, alpha, x, incx_ilp64, ap);
}

extern "C" void ssyr_(
    const char *uplo,
    const compensated_blas_blas_int *n,
    const float *alpha,
    const float *x,
    const compensated_blas_blas_int *incx,
    float *a,
    const compensated_blas_blas_int *lda
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

    backend().ssyr(uplo, n_ilp64, alpha, x, incx_ilp64, a, lda_ilp64);
}

extern "C" void sspr2_(
    const char *uplo,
    const compensated_blas_blas_int *n,
    const float *alpha,
    const float *x,
    const compensated_blas_blas_int *incx,
    const float *y,
    const compensated_blas_blas_int *incy,
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

    backend().sspr2(uplo, n_ilp64, alpha, x, incx_ilp64, y, incy_ilp64, ap);
}

extern "C" void ssyr2_(
    const char *uplo,
    const compensated_blas_blas_int *n,
    const float *alpha,
    const float *x,
    const compensated_blas_blas_int *incx,
    const float *y,
    const compensated_blas_blas_int *incy,
    float *a,
    const compensated_blas_blas_int *lda
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

    backend().ssyr2(uplo, n_ilp64, alpha, x, incx_ilp64, y, incy_ilp64, a, lda_ilp64);
}

extern "C" void dgemv_(
    const char *trans,
    const compensated_blas_blas_int *m,
    const compensated_blas_blas_int *n,
    const double *alpha,
    const double *a,
    const compensated_blas_blas_int *lda,
    const double *x,
    const compensated_blas_blas_int *incx,
    const double *beta,
    double *y,
    const compensated_blas_blas_int *incy
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

    backend().dgemv(trans, m_ilp64, n_ilp64, alpha, a, lda_ilp64, x, incx_ilp64, beta, y, incy_ilp64);
}

extern "C" void dgbmv_(
    const char *trans,
    const compensated_blas_blas_int *m,
    const compensated_blas_blas_int *n,
    const compensated_blas_blas_int *kl,
    const compensated_blas_blas_int *ku,
    const double *alpha,
    const double *a,
    const compensated_blas_blas_int *lda,
    const double *x,
    const compensated_blas_blas_int *incx,
    const double *beta,
    double *y,
    const compensated_blas_blas_int *incy
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

    backend().dgbmv(trans, m_ilp64, n_ilp64, kl_ilp64, ku_ilp64, alpha, a, lda_ilp64, x, incx_ilp64, beta, y, incy_ilp64);
}

extern "C" void dsymv_(
    const char *uplo,
    const compensated_blas_blas_int *n,
    const double *alpha,
    const double *a,
    const compensated_blas_blas_int *lda,
    const double *x,
    const compensated_blas_blas_int *incx,
    const double *beta,
    double *y,
    const compensated_blas_blas_int *incy
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

    backend().dsymv(uplo, n_ilp64, alpha, a, lda_ilp64, x, incx_ilp64, beta, y, incy_ilp64);
}

extern "C" void dsbmv_(
    const char *uplo,
    const compensated_blas_blas_int *n,
    const compensated_blas_blas_int *k,
    const double *alpha,
    const double *a,
    const compensated_blas_blas_int *lda,
    const double *x,
    const compensated_blas_blas_int *incx,
    const double *beta,
    double *y,
    const compensated_blas_blas_int *incy
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

    backend().dsbmv(uplo, n_ilp64, k_ilp64, alpha, a, lda_ilp64, x, incx_ilp64, beta, y, incy_ilp64);
}

extern "C" void dspmv_(
    const char *uplo,
    const compensated_blas_blas_int *n,
    const double *alpha,
    const double *ap,
    const double *x,
    const compensated_blas_blas_int *incx,
    const double *beta,
    double *y,
    const compensated_blas_blas_int *incy
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

    backend().dspmv(uplo, n_ilp64, alpha, ap, x, incx_ilp64, beta, y, incy_ilp64);
}

extern "C" void dtrmv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const compensated_blas_blas_int *n,
    const double *a,
    const compensated_blas_blas_int *lda,
    double *x,
    const compensated_blas_blas_int *incx
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

    backend().dtrmv(uplo, trans, diag, n_ilp64, a, lda_ilp64, x, incx_ilp64);
}

extern "C" void dtbmv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const compensated_blas_blas_int *n,
    const compensated_blas_blas_int *k,
    const double *a,
    const compensated_blas_blas_int *lda,
    double *x,
    const compensated_blas_blas_int *incx
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

    backend().dtbmv(uplo, trans, diag, n_ilp64, k_ilp64, a, lda_ilp64, x, incx_ilp64);
}

extern "C" void dtpmv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const compensated_blas_blas_int *n,
    const double *ap,
    double *x,
    const compensated_blas_blas_int *incx
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

    backend().dtpmv(uplo, trans, diag, n_ilp64, ap, x, incx_ilp64);
}

extern "C" void dtrsv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const compensated_blas_blas_int *n,
    const double *a,
    const compensated_blas_blas_int *lda,
    double *x,
    const compensated_blas_blas_int *incx
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

    backend().dtrsv(uplo, trans, diag, n_ilp64, a, lda_ilp64, x, incx_ilp64);
}

extern "C" void dtbsv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const compensated_blas_blas_int *n,
    const compensated_blas_blas_int *k,
    const double *a,
    const compensated_blas_blas_int *lda,
    double *x,
    const compensated_blas_blas_int *incx
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

    backend().dtbsv(uplo, trans, diag, n_ilp64, k_ilp64, a, lda_ilp64, x, incx_ilp64);
}

extern "C" void dtpsv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const compensated_blas_blas_int *n,
    const double *ap,
    double *x,
    const compensated_blas_blas_int *incx
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

    backend().dtpsv(uplo, trans, diag, n_ilp64, ap, x, incx_ilp64);
}

extern "C" void dger_(
    const compensated_blas_blas_int *m,
    const compensated_blas_blas_int *n,
    const double *alpha,
    const double *x,
    const compensated_blas_blas_int *incx,
    const double *y,
    const compensated_blas_blas_int *incy,
    double *a,
    const compensated_blas_blas_int *lda
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

    backend().dger(m_ilp64, n_ilp64, alpha, x, incx_ilp64, y, incy_ilp64, a, lda_ilp64);
}

extern "C" void dspr_(
    const char *uplo,
    const compensated_blas_blas_int *n,
    const double *alpha,
    const double *x,
    const compensated_blas_blas_int *incx,
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

    backend().dspr(uplo, n_ilp64, alpha, x, incx_ilp64, ap);
}

extern "C" void dsyr_(
    const char *uplo,
    const compensated_blas_blas_int *n,
    const double *alpha,
    const double *x,
    const compensated_blas_blas_int *incx,
    double *a,
    const compensated_blas_blas_int *lda
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

    backend().dsyr(uplo, n_ilp64, alpha, x, incx_ilp64, a, lda_ilp64);
}

extern "C" void dspr2_(
    const char *uplo,
    const compensated_blas_blas_int *n,
    const double *alpha,
    const double *x,
    const compensated_blas_blas_int *incx,
    const double *y,
    const compensated_blas_blas_int *incy,
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

    backend().dspr2(uplo, n_ilp64, alpha, x, incx_ilp64, y, incy_ilp64, ap);
}

extern "C" void dsyr2_(
    const char *uplo,
    const compensated_blas_blas_int *n,
    const double *alpha,
    const double *x,
    const compensated_blas_blas_int *incx,
    const double *y,
    const compensated_blas_blas_int *incy,
    double *a,
    const compensated_blas_blas_int *lda
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

    backend().dsyr2(uplo, n_ilp64, alpha, x, incx_ilp64, y, incy_ilp64, a, lda_ilp64);
}

extern "C" void cgemv_(
    const char *trans,
    const compensated_blas_blas_int *m,
    const compensated_blas_blas_int *n,
    const compensated_blas_complex_float *alpha,
    const compensated_blas_complex_float *a,
    const compensated_blas_blas_int *lda,
    const compensated_blas_complex_float *x,
    const compensated_blas_blas_int *incx,
    const compensated_blas_complex_float *beta,
    compensated_blas_complex_float *y,
    const compensated_blas_blas_int *incy
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

    backend().cgemv(trans, m_ilp64, n_ilp64, alpha, a, lda_ilp64, x, incx_ilp64, beta, y, incy_ilp64);
}

extern "C" void cgbmv_(
    const char *trans,
    const compensated_blas_blas_int *m,
    const compensated_blas_blas_int *n,
    const compensated_blas_blas_int *kl,
    const compensated_blas_blas_int *ku,
    const compensated_blas_complex_float *alpha,
    const compensated_blas_complex_float *a,
    const compensated_blas_blas_int *lda,
    const compensated_blas_complex_float *x,
    const compensated_blas_blas_int *incx,
    const compensated_blas_complex_float *beta,
    compensated_blas_complex_float *y,
    const compensated_blas_blas_int *incy
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

    backend().cgbmv(trans, m_ilp64, n_ilp64, kl_ilp64, ku_ilp64, alpha, a, lda_ilp64, x, incx_ilp64, beta, y, incy_ilp64);
}

extern "C" void chemv_(
    const char *uplo,
    const compensated_blas_blas_int *n,
    const compensated_blas_complex_float *alpha,
    const compensated_blas_complex_float *a,
    const compensated_blas_blas_int *lda,
    const compensated_blas_complex_float *x,
    const compensated_blas_blas_int *incx,
    const compensated_blas_complex_float *beta,
    compensated_blas_complex_float *y,
    const compensated_blas_blas_int *incy
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

    backend().chemv(uplo, n_ilp64, alpha, a, lda_ilp64, x, incx_ilp64, beta, y, incy_ilp64);
}

extern "C" void chbmv_(
    const char *uplo,
    const compensated_blas_blas_int *n,
    const compensated_blas_blas_int *k,
    const compensated_blas_complex_float *alpha,
    const compensated_blas_complex_float *a,
    const compensated_blas_blas_int *lda,
    const compensated_blas_complex_float *x,
    const compensated_blas_blas_int *incx,
    const compensated_blas_complex_float *beta,
    compensated_blas_complex_float *y,
    const compensated_blas_blas_int *incy
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

    backend().chbmv(uplo, n_ilp64, k_ilp64, alpha, a, lda_ilp64, x, incx_ilp64, beta, y, incy_ilp64);
}

extern "C" void chpmv_(
    const char *uplo,
    const compensated_blas_blas_int *n,
    const compensated_blas_complex_float *alpha,
    const compensated_blas_complex_float *ap,
    const compensated_blas_complex_float *x,
    const compensated_blas_blas_int *incx,
    const compensated_blas_complex_float *beta,
    compensated_blas_complex_float *y,
    const compensated_blas_blas_int *incy
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

    backend().chpmv(uplo, n_ilp64, alpha, ap, x, incx_ilp64, beta, y, incy_ilp64);
}

extern "C" void ctrmv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const compensated_blas_blas_int *n,
    const compensated_blas_complex_float *a,
    const compensated_blas_blas_int *lda,
    compensated_blas_complex_float *x,
    const compensated_blas_blas_int *incx
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

    backend().ctrmv(uplo, trans, diag, n_ilp64, a, lda_ilp64, x, incx_ilp64);
}

extern "C" void ctbmv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const compensated_blas_blas_int *n,
    const compensated_blas_blas_int *k,
    const compensated_blas_complex_float *a,
    const compensated_blas_blas_int *lda,
    compensated_blas_complex_float *x,
    const compensated_blas_blas_int *incx
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

    backend().ctbmv(uplo, trans, diag, n_ilp64, k_ilp64, a, lda_ilp64, x, incx_ilp64);
}

extern "C" void ctpmv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const compensated_blas_blas_int *n,
    const compensated_blas_complex_float *ap,
    compensated_blas_complex_float *x,
    const compensated_blas_blas_int *incx
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

    backend().ctpmv(uplo, trans, diag, n_ilp64, ap, x, incx_ilp64);
}

extern "C" void ctrsv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const compensated_blas_blas_int *n,
    const compensated_blas_complex_float *a,
    const compensated_blas_blas_int *lda,
    compensated_blas_complex_float *x,
    const compensated_blas_blas_int *incx
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

    backend().ctrsv(uplo, trans, diag, n_ilp64, a, lda_ilp64, x, incx_ilp64);
}

extern "C" void ctbsv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const compensated_blas_blas_int *n,
    const compensated_blas_blas_int *k,
    const compensated_blas_complex_float *a,
    const compensated_blas_blas_int *lda,
    compensated_blas_complex_float *x,
    const compensated_blas_blas_int *incx
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

    backend().ctbsv(uplo, trans, diag, n_ilp64, k_ilp64, a, lda_ilp64, x, incx_ilp64);
}

extern "C" void ctpsv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const compensated_blas_blas_int *n,
    const compensated_blas_complex_float *ap,
    compensated_blas_complex_float *x,
    const compensated_blas_blas_int *incx
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

    backend().ctpsv(uplo, trans, diag, n_ilp64, ap, x, incx_ilp64);
}

extern "C" void cgerc_(
    const compensated_blas_blas_int *m,
    const compensated_blas_blas_int *n,
    const compensated_blas_complex_float *alpha,
    const compensated_blas_complex_float *x,
    const compensated_blas_blas_int *incx,
    const compensated_blas_complex_float *y,
    const compensated_blas_blas_int *incy,
    compensated_blas_complex_float *a,
    const compensated_blas_blas_int *lda
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

    backend().cgerc(m_ilp64, n_ilp64, alpha, x, incx_ilp64, y, incy_ilp64, a, lda_ilp64);
}

extern "C" void cgeru_(
    const compensated_blas_blas_int *m,
    const compensated_blas_blas_int *n,
    const compensated_blas_complex_float *alpha,
    const compensated_blas_complex_float *x,
    const compensated_blas_blas_int *incx,
    const compensated_blas_complex_float *y,
    const compensated_blas_blas_int *incy,
    compensated_blas_complex_float *a,
    const compensated_blas_blas_int *lda
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

    backend().cgeru(m_ilp64, n_ilp64, alpha, x, incx_ilp64, y, incy_ilp64, a, lda_ilp64);
}

extern "C" void cher_(
    const char *uplo,
    const compensated_blas_blas_int *n,
    const float *alpha,
    const compensated_blas_complex_float *x,
    const compensated_blas_blas_int *incx,
    compensated_blas_complex_float *a,
    const compensated_blas_blas_int *lda
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

    backend().cher(uplo, n_ilp64, alpha, x, incx_ilp64, a, lda_ilp64);
}

extern "C" void chpr_(
    const char *uplo,
    const compensated_blas_blas_int *n,
    const float *alpha,
    const compensated_blas_complex_float *x,
    const compensated_blas_blas_int *incx,
    compensated_blas_complex_float *ap
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

    backend().chpr(uplo, n_ilp64, alpha, x, incx_ilp64, ap);
}

extern "C" void cher2_(
    const char *uplo,
    const compensated_blas_blas_int *n,
    const compensated_blas_complex_float *alpha,
    const compensated_blas_complex_float *x,
    const compensated_blas_blas_int *incx,
    const compensated_blas_complex_float *y,
    const compensated_blas_blas_int *incy,
    compensated_blas_complex_float *a,
    const compensated_blas_blas_int *lda
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

    backend().cher2(uplo, n_ilp64, alpha, x, incx_ilp64, y, incy_ilp64, a, lda_ilp64);
}

extern "C" void chpr2_(
    const char *uplo,
    const compensated_blas_blas_int *n,
    const compensated_blas_complex_float *alpha,
    const compensated_blas_complex_float *x,
    const compensated_blas_blas_int *incx,
    const compensated_blas_complex_float *y,
    const compensated_blas_blas_int *incy,
    compensated_blas_complex_float *ap
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

    backend().chpr2(uplo, n_ilp64, alpha, x, incx_ilp64, y, incy_ilp64, ap);
}

extern "C" void zgemv_(
    const char *trans,
    const compensated_blas_blas_int *m,
    const compensated_blas_blas_int *n,
    const compensated_blas_complex_double *alpha,
    const compensated_blas_complex_double *a,
    const compensated_blas_blas_int *lda,
    const compensated_blas_complex_double *x,
    const compensated_blas_blas_int *incx,
    const compensated_blas_complex_double *beta,
    compensated_blas_complex_double *y,
    const compensated_blas_blas_int *incy
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

    backend().zgemv(trans, m_ilp64, n_ilp64, alpha, a, lda_ilp64, x, incx_ilp64, beta, y, incy_ilp64);
}

extern "C" void zgbmv_(
    const char *trans,
    const compensated_blas_blas_int *m,
    const compensated_blas_blas_int *n,
    const compensated_blas_blas_int *kl,
    const compensated_blas_blas_int *ku,
    const compensated_blas_complex_double *alpha,
    const compensated_blas_complex_double *a,
    const compensated_blas_blas_int *lda,
    const compensated_blas_complex_double *x,
    const compensated_blas_blas_int *incx,
    const compensated_blas_complex_double *beta,
    compensated_blas_complex_double *y,
    const compensated_blas_blas_int *incy
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

    backend().zgbmv(trans, m_ilp64, n_ilp64, kl_ilp64, ku_ilp64, alpha, a, lda_ilp64, x, incx_ilp64, beta, y, incy_ilp64);
}

extern "C" void zhemv_(
    const char *uplo,
    const compensated_blas_blas_int *n,
    const compensated_blas_complex_double *alpha,
    const compensated_blas_complex_double *a,
    const compensated_blas_blas_int *lda,
    const compensated_blas_complex_double *x,
    const compensated_blas_blas_int *incx,
    const compensated_blas_complex_double *beta,
    compensated_blas_complex_double *y,
    const compensated_blas_blas_int *incy
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

    backend().zhemv(uplo, n_ilp64, alpha, a, lda_ilp64, x, incx_ilp64, beta, y, incy_ilp64);
}

extern "C" void zhbmv_(
    const char *uplo,
    const compensated_blas_blas_int *n,
    const compensated_blas_blas_int *k,
    const compensated_blas_complex_double *alpha,
    const compensated_blas_complex_double *a,
    const compensated_blas_blas_int *lda,
    const compensated_blas_complex_double *x,
    const compensated_blas_blas_int *incx,
    const compensated_blas_complex_double *beta,
    compensated_blas_complex_double *y,
    const compensated_blas_blas_int *incy
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

    backend().zhbmv(uplo, n_ilp64, k_ilp64, alpha, a, lda_ilp64, x, incx_ilp64, beta, y, incy_ilp64);
}

extern "C" void zhpmv_(
    const char *uplo,
    const compensated_blas_blas_int *n,
    const compensated_blas_complex_double *alpha,
    const compensated_blas_complex_double *ap,
    const compensated_blas_complex_double *x,
    const compensated_blas_blas_int *incx,
    const compensated_blas_complex_double *beta,
    compensated_blas_complex_double *y,
    const compensated_blas_blas_int *incy
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

    backend().zhpmv(uplo, n_ilp64, alpha, ap, x, incx_ilp64, beta, y, incy_ilp64);
}

extern "C" void ztrmv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const compensated_blas_blas_int *n,
    const compensated_blas_complex_double *a,
    const compensated_blas_blas_int *lda,
    compensated_blas_complex_double *x,
    const compensated_blas_blas_int *incx
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

    backend().ztrmv(uplo, trans, diag, n_ilp64, a, lda_ilp64, x, incx_ilp64);
}

extern "C" void ztbmv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const compensated_blas_blas_int *n,
    const compensated_blas_blas_int *k,
    const compensated_blas_complex_double *a,
    const compensated_blas_blas_int *lda,
    compensated_blas_complex_double *x,
    const compensated_blas_blas_int *incx
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

    backend().ztbmv(uplo, trans, diag, n_ilp64, k_ilp64, a, lda_ilp64, x, incx_ilp64);
}

extern "C" void ztpmv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const compensated_blas_blas_int *n,
    const compensated_blas_complex_double *ap,
    compensated_blas_complex_double *x,
    const compensated_blas_blas_int *incx
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

    backend().ztpmv(uplo, trans, diag, n_ilp64, ap, x, incx_ilp64);
}

extern "C" void ztrsv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const compensated_blas_blas_int *n,
    const compensated_blas_complex_double *a,
    const compensated_blas_blas_int *lda,
    compensated_blas_complex_double *x,
    const compensated_blas_blas_int *incx
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

    backend().ztrsv(uplo, trans, diag, n_ilp64, a, lda_ilp64, x, incx_ilp64);
}

extern "C" void ztbsv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const compensated_blas_blas_int *n,
    const compensated_blas_blas_int *k,
    const compensated_blas_complex_double *a,
    const compensated_blas_blas_int *lda,
    compensated_blas_complex_double *x,
    const compensated_blas_blas_int *incx
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

    backend().ztbsv(uplo, trans, diag, n_ilp64, k_ilp64, a, lda_ilp64, x, incx_ilp64);
}

extern "C" void ztpsv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const compensated_blas_blas_int *n,
    const compensated_blas_complex_double *ap,
    compensated_blas_complex_double *x,
    const compensated_blas_blas_int *incx
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

    backend().ztpsv(uplo, trans, diag, n_ilp64, ap, x, incx_ilp64);
}

extern "C" void zgerc_(
    const compensated_blas_blas_int *m,
    const compensated_blas_blas_int *n,
    const compensated_blas_complex_double *alpha,
    const compensated_blas_complex_double *x,
    const compensated_blas_blas_int *incx,
    const compensated_blas_complex_double *y,
    const compensated_blas_blas_int *incy,
    compensated_blas_complex_double *a,
    const compensated_blas_blas_int *lda
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

    backend().zgerc(m_ilp64, n_ilp64, alpha, x, incx_ilp64, y, incy_ilp64, a, lda_ilp64);
}

extern "C" void zgeru_(
    const compensated_blas_blas_int *m,
    const compensated_blas_blas_int *n,
    const compensated_blas_complex_double *alpha,
    const compensated_blas_complex_double *x,
    const compensated_blas_blas_int *incx,
    const compensated_blas_complex_double *y,
    const compensated_blas_blas_int *incy,
    compensated_blas_complex_double *a,
    const compensated_blas_blas_int *lda
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

    backend().zgeru(m_ilp64, n_ilp64, alpha, x, incx_ilp64, y, incy_ilp64, a, lda_ilp64);
}

extern "C" void zher_(
    const char *uplo,
    const compensated_blas_blas_int *n,
    const double *alpha,
    const compensated_blas_complex_double *x,
    const compensated_blas_blas_int *incx,
    compensated_blas_complex_double *a,
    const compensated_blas_blas_int *lda
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

    backend().zher(uplo, n_ilp64, alpha, x, incx_ilp64, a, lda_ilp64);
}

extern "C" void zhpr_(
    const char *uplo,
    const compensated_blas_blas_int *n,
    const double *alpha,
    const compensated_blas_complex_double *x,
    const compensated_blas_blas_int *incx,
    compensated_blas_complex_double *ap
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

    backend().zhpr(uplo, n_ilp64, alpha, x, incx_ilp64, ap);
}

extern "C" void zher2_(
    const char *uplo,
    const compensated_blas_blas_int *n,
    const compensated_blas_complex_double *alpha,
    const compensated_blas_complex_double *x,
    const compensated_blas_blas_int *incx,
    const compensated_blas_complex_double *y,
    const compensated_blas_blas_int *incy,
    compensated_blas_complex_double *a,
    const compensated_blas_blas_int *lda
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

    backend().zher2(uplo, n_ilp64, alpha, x, incx_ilp64, y, incy_ilp64, a, lda_ilp64);
}

extern "C" void zhpr2_(
    const char *uplo,
    const compensated_blas_blas_int *n,
    const compensated_blas_complex_double *alpha,
    const compensated_blas_complex_double *x,
    const compensated_blas_blas_int *incx,
    const compensated_blas_complex_double *y,
    const compensated_blas_blas_int *incy,
    compensated_blas_complex_double *ap
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

    backend().zhpr2(uplo, n_ilp64, alpha, x, incx_ilp64, y, incy_ilp64, ap);
}

extern "C" void sgemm_(
    const char *transa,
    const char *transb,
    const compensated_blas_blas_int *m,
    const compensated_blas_blas_int *n,
    const compensated_blas_blas_int *k,
    const float *alpha,
    const float *a,
    const compensated_blas_blas_int *lda,
    const float *b,
    const compensated_blas_blas_int *ldb,
    const float *beta,
    float *c,
    const compensated_blas_blas_int *ldc
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

    backend().sgemm(transa, transb, m_ilp64, n_ilp64, k_ilp64, alpha, a, lda_ilp64, b, ldb_ilp64, beta, c, ldc_ilp64);
}

extern "C" void ssymm_(
    const char *side,
    const char *uplo,
    const compensated_blas_blas_int *m,
    const compensated_blas_blas_int *n,
    const float *alpha,
    const float *a,
    const compensated_blas_blas_int *lda,
    const float *b,
    const compensated_blas_blas_int *ldb,
    const float *beta,
    float *c,
    const compensated_blas_blas_int *ldc
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

    backend().ssymm(side, uplo, m_ilp64, n_ilp64, alpha, a, lda_ilp64, b, ldb_ilp64, beta, c, ldc_ilp64);
}

extern "C" void ssyrk_(
    const char *uplo,
    const char *trans,
    const compensated_blas_blas_int *n,
    const compensated_blas_blas_int *k,
    const float *alpha,
    const float *a,
    const compensated_blas_blas_int *lda,
    const float *beta,
    float *c,
    const compensated_blas_blas_int *ldc
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

    backend().ssyrk(uplo, trans, n_ilp64, k_ilp64, alpha, a, lda_ilp64, beta, c, ldc_ilp64);
}

extern "C" void ssyr2k_(
    const char *uplo,
    const char *trans,
    const compensated_blas_blas_int *n,
    const compensated_blas_blas_int *k,
    const float *alpha,
    const float *a,
    const compensated_blas_blas_int *lda,
    const float *b,
    const compensated_blas_blas_int *ldb,
    const float *beta,
    float *c,
    const compensated_blas_blas_int *ldc
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

    backend().ssyr2k(uplo, trans, n_ilp64, k_ilp64, alpha, a, lda_ilp64, b, ldb_ilp64, beta, c, ldc_ilp64);
}

extern "C" void strsm_(
    const char *side,
    const char *uplo,
    const char *transa,
    const char *diag,
    const compensated_blas_blas_int *m,
    const compensated_blas_blas_int *n,
    const float *alpha,
    const float *a,
    const compensated_blas_blas_int *lda,
    float *b,
    const compensated_blas_blas_int *ldb
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

    backend().strsm(side, uplo, transa, diag, m_ilp64, n_ilp64, alpha, a, lda_ilp64, b, ldb_ilp64);
}

extern "C" void strmm_(
    const char *side,
    const char *uplo,
    const char *transa,
    const char *diag,
    const compensated_blas_blas_int *m,
    const compensated_blas_blas_int *n,
    const float *alpha,
    const float *a,
    const compensated_blas_blas_int *lda,
    float *b,
    const compensated_blas_blas_int *ldb
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

    backend().strmm(side, uplo, transa, diag, m_ilp64, n_ilp64, alpha, a, lda_ilp64, b, ldb_ilp64);
}

extern "C" void dgemm_(
    const char *transa,
    const char *transb,
    const compensated_blas_blas_int *m,
    const compensated_blas_blas_int *n,
    const compensated_blas_blas_int *k,
    const double *alpha,
    const double *a,
    const compensated_blas_blas_int *lda,
    const double *b,
    const compensated_blas_blas_int *ldb,
    const double *beta,
    double *c,
    const compensated_blas_blas_int *ldc
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

    backend().dgemm(transa, transb, m_ilp64, n_ilp64, k_ilp64, alpha, a, lda_ilp64, b, ldb_ilp64, beta, c, ldc_ilp64);
}

extern "C" void dsymm_(
    const char *side,
    const char *uplo,
    const compensated_blas_blas_int *m,
    const compensated_blas_blas_int *n,
    const double *alpha,
    const double *a,
    const compensated_blas_blas_int *lda,
    const double *b,
    const compensated_blas_blas_int *ldb,
    const double *beta,
    double *c,
    const compensated_blas_blas_int *ldc
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

    backend().dsymm(side, uplo, m_ilp64, n_ilp64, alpha, a, lda_ilp64, b, ldb_ilp64, beta, c, ldc_ilp64);
}

extern "C" void dsyrk_(
    const char *uplo,
    const char *trans,
    const compensated_blas_blas_int *n,
    const compensated_blas_blas_int *k,
    const double *alpha,
    const double *a,
    const compensated_blas_blas_int *lda,
    const double *beta,
    double *c,
    const compensated_blas_blas_int *ldc
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

    backend().dsyrk(uplo, trans, n_ilp64, k_ilp64, alpha, a, lda_ilp64, beta, c, ldc_ilp64);
}

extern "C" void dsyr2k_(
    const char *uplo,
    const char *trans,
    const compensated_blas_blas_int *n,
    const compensated_blas_blas_int *k,
    const double *alpha,
    const double *a,
    const compensated_blas_blas_int *lda,
    const double *b,
    const compensated_blas_blas_int *ldb,
    const double *beta,
    double *c,
    const compensated_blas_blas_int *ldc
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

    backend().dsyr2k(uplo, trans, n_ilp64, k_ilp64, alpha, a, lda_ilp64, b, ldb_ilp64, beta, c, ldc_ilp64);
}

extern "C" void dtrsm_(
    const char *side,
    const char *uplo,
    const char *transa,
    const char *diag,
    const compensated_blas_blas_int *m,
    const compensated_blas_blas_int *n,
    const double *alpha,
    const double *a,
    const compensated_blas_blas_int *lda,
    double *b,
    const compensated_blas_blas_int *ldb
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

    backend().dtrsm(side, uplo, transa, diag, m_ilp64, n_ilp64, alpha, a, lda_ilp64, b, ldb_ilp64);
}

extern "C" void dtrmm_(
    const char *side,
    const char *uplo,
    const char *transa,
    const char *diag,
    const compensated_blas_blas_int *m,
    const compensated_blas_blas_int *n,
    const double *alpha,
    const double *a,
    const compensated_blas_blas_int *lda,
    double *b,
    const compensated_blas_blas_int *ldb
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

    backend().dtrmm(side, uplo, transa, diag, m_ilp64, n_ilp64, alpha, a, lda_ilp64, b, ldb_ilp64);
}

extern "C" void cgemm_(
    const char *transa,
    const char *transb,
    const compensated_blas_blas_int *m,
    const compensated_blas_blas_int *n,
    const compensated_blas_blas_int *k,
    const compensated_blas_complex_float *alpha,
    const compensated_blas_complex_float *a,
    const compensated_blas_blas_int *lda,
    const compensated_blas_complex_float *b,
    const compensated_blas_blas_int *ldb,
    const compensated_blas_complex_float *beta,
    compensated_blas_complex_float *c,
    const compensated_blas_blas_int *ldc
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

    backend().cgemm(transa, transb, m_ilp64, n_ilp64, k_ilp64, alpha, a, lda_ilp64, b, ldb_ilp64, beta, c, ldc_ilp64);
}

extern "C" void csymm_(
    const char *side,
    const char *uplo,
    const compensated_blas_blas_int *m,
    const compensated_blas_blas_int *n,
    const compensated_blas_complex_float *alpha,
    const compensated_blas_complex_float *a,
    const compensated_blas_blas_int *lda,
    const compensated_blas_complex_float *b,
    const compensated_blas_blas_int *ldb,
    const compensated_blas_complex_float *beta,
    compensated_blas_complex_float *c,
    const compensated_blas_blas_int *ldc
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

    backend().csymm(side, uplo, m_ilp64, n_ilp64, alpha, a, lda_ilp64, b, ldb_ilp64, beta, c, ldc_ilp64);
}

extern "C" void chemm_(
    const char *side,
    const char *uplo,
    const compensated_blas_blas_int *m,
    const compensated_blas_blas_int *n,
    const compensated_blas_complex_float *alpha,
    const compensated_blas_complex_float *a,
    const compensated_blas_blas_int *lda,
    const compensated_blas_complex_float *b,
    const compensated_blas_blas_int *ldb,
    const compensated_blas_complex_float *beta,
    compensated_blas_complex_float *c,
    const compensated_blas_blas_int *ldc
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

    backend().chemm(side, uplo, m_ilp64, n_ilp64, alpha, a, lda_ilp64, b, ldb_ilp64, beta, c, ldc_ilp64);
}

extern "C" void csyrk_(
    const char *uplo,
    const char *trans,
    const compensated_blas_blas_int *n,
    const compensated_blas_blas_int *k,
    const compensated_blas_complex_float *alpha,
    const compensated_blas_complex_float *a,
    const compensated_blas_blas_int *lda,
    const compensated_blas_complex_float *beta,
    compensated_blas_complex_float *c,
    const compensated_blas_blas_int *ldc
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

    backend().csyrk(uplo, trans, n_ilp64, k_ilp64, alpha, a, lda_ilp64, beta, c, ldc_ilp64);
}

extern "C" void cherk_(
    const char *uplo,
    const char *trans,
    const compensated_blas_blas_int *n,
    const compensated_blas_blas_int *k,
    const float *alpha,
    const compensated_blas_complex_float *a,
    const compensated_blas_blas_int *lda,
    const float *beta,
    compensated_blas_complex_float *c,
    const compensated_blas_blas_int *ldc
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

    backend().cherk(uplo, trans, n_ilp64, k_ilp64, alpha, a, lda_ilp64, beta, c, ldc_ilp64);
}

extern "C" void csyr2k_(
    const char *uplo,
    const char *trans,
    const compensated_blas_blas_int *n,
    const compensated_blas_blas_int *k,
    const compensated_blas_complex_float *alpha,
    const compensated_blas_complex_float *a,
    const compensated_blas_blas_int *lda,
    const compensated_blas_complex_float *b,
    const compensated_blas_blas_int *ldb,
    const compensated_blas_complex_float *beta,
    compensated_blas_complex_float *c,
    const compensated_blas_blas_int *ldc
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

    backend().csyr2k(uplo, trans, n_ilp64, k_ilp64, alpha, a, lda_ilp64, b, ldb_ilp64, beta, c, ldc_ilp64);
}

extern "C" void cher2k_(
    const char *uplo,
    const char *trans,
    const compensated_blas_blas_int *n,
    const compensated_blas_blas_int *k,
    const compensated_blas_complex_float *alpha,
    const compensated_blas_complex_float *a,
    const compensated_blas_blas_int *lda,
    const compensated_blas_complex_float *b,
    const compensated_blas_blas_int *ldb,
    const float *beta,
    compensated_blas_complex_float *c,
    const compensated_blas_blas_int *ldc
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

    backend().cher2k(uplo, trans, n_ilp64, k_ilp64, alpha, a, lda_ilp64, b, ldb_ilp64, beta, c, ldc_ilp64);
}

extern "C" void ctrsm_(
    const char *side,
    const char *uplo,
    const char *transa,
    const char *diag,
    const compensated_blas_blas_int *m,
    const compensated_blas_blas_int *n,
    const compensated_blas_complex_float *alpha,
    const compensated_blas_complex_float *a,
    const compensated_blas_blas_int *lda,
    compensated_blas_complex_float *b,
    const compensated_blas_blas_int *ldb
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

    backend().ctrsm(side, uplo, transa, diag, m_ilp64, n_ilp64, alpha, a, lda_ilp64, b, ldb_ilp64);
}

extern "C" void ctrmm_(
    const char *side,
    const char *uplo,
    const char *transa,
    const char *diag,
    const compensated_blas_blas_int *m,
    const compensated_blas_blas_int *n,
    const compensated_blas_complex_float *alpha,
    const compensated_blas_complex_float *a,
    const compensated_blas_blas_int *lda,
    compensated_blas_complex_float *b,
    const compensated_blas_blas_int *ldb
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

    backend().ctrmm(side, uplo, transa, diag, m_ilp64, n_ilp64, alpha, a, lda_ilp64, b, ldb_ilp64);
}

extern "C" void zgemm_(
    const char *transa,
    const char *transb,
    const compensated_blas_blas_int *m,
    const compensated_blas_blas_int *n,
    const compensated_blas_blas_int *k,
    const compensated_blas_complex_double *alpha,
    const compensated_blas_complex_double *a,
    const compensated_blas_blas_int *lda,
    const compensated_blas_complex_double *b,
    const compensated_blas_blas_int *ldb,
    const compensated_blas_complex_double *beta,
    compensated_blas_complex_double *c,
    const compensated_blas_blas_int *ldc
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

    backend().zgemm(transa, transb, m_ilp64, n_ilp64, k_ilp64, alpha, a, lda_ilp64, b, ldb_ilp64, beta, c, ldc_ilp64);
}

extern "C" void zsymm_(
    const char *side,
    const char *uplo,
    const compensated_blas_blas_int *m,
    const compensated_blas_blas_int *n,
    const compensated_blas_complex_double *alpha,
    const compensated_blas_complex_double *a,
    const compensated_blas_blas_int *lda,
    const compensated_blas_complex_double *b,
    const compensated_blas_blas_int *ldb,
    const compensated_blas_complex_double *beta,
    compensated_blas_complex_double *c,
    const compensated_blas_blas_int *ldc
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

    backend().zsymm(side, uplo, m_ilp64, n_ilp64, alpha, a, lda_ilp64, b, ldb_ilp64, beta, c, ldc_ilp64);
}

extern "C" void zhemm_(
    const char *side,
    const char *uplo,
    const compensated_blas_blas_int *m,
    const compensated_blas_blas_int *n,
    const compensated_blas_complex_double *alpha,
    const compensated_blas_complex_double *a,
    const compensated_blas_blas_int *lda,
    const compensated_blas_complex_double *b,
    const compensated_blas_blas_int *ldb,
    const compensated_blas_complex_double *beta,
    compensated_blas_complex_double *c,
    const compensated_blas_blas_int *ldc
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

    backend().zhemm(side, uplo, m_ilp64, n_ilp64, alpha, a, lda_ilp64, b, ldb_ilp64, beta, c, ldc_ilp64);
}

extern "C" void zsyrk_(
    const char *uplo,
    const char *trans,
    const compensated_blas_blas_int *n,
    const compensated_blas_blas_int *k,
    const compensated_blas_complex_double *alpha,
    const compensated_blas_complex_double *a,
    const compensated_blas_blas_int *lda,
    const compensated_blas_complex_double *beta,
    compensated_blas_complex_double *c,
    const compensated_blas_blas_int *ldc
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

    backend().zsyrk(uplo, trans, n_ilp64, k_ilp64, alpha, a, lda_ilp64, beta, c, ldc_ilp64);
}

extern "C" void zherk_(
    const char *uplo,
    const char *trans,
    const compensated_blas_blas_int *n,
    const compensated_blas_blas_int *k,
    const double *alpha,
    const compensated_blas_complex_double *a,
    const compensated_blas_blas_int *lda,
    const double *beta,
    compensated_blas_complex_double *c,
    const compensated_blas_blas_int *ldc
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

    backend().zherk(uplo, trans, n_ilp64, k_ilp64, alpha, a, lda_ilp64, beta, c, ldc_ilp64);
}

extern "C" void zsyr2k_(
    const char *uplo,
    const char *trans,
    const compensated_blas_blas_int *n,
    const compensated_blas_blas_int *k,
    const compensated_blas_complex_double *alpha,
    const compensated_blas_complex_double *a,
    const compensated_blas_blas_int *lda,
    const compensated_blas_complex_double *b,
    const compensated_blas_blas_int *ldb,
    const compensated_blas_complex_double *beta,
    compensated_blas_complex_double *c,
    const compensated_blas_blas_int *ldc
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

    backend().zsyr2k(uplo, trans, n_ilp64, k_ilp64, alpha, a, lda_ilp64, b, ldb_ilp64, beta, c, ldc_ilp64);
}

extern "C" void zher2k_(
    const char *uplo,
    const char *trans,
    const compensated_blas_blas_int *n,
    const compensated_blas_blas_int *k,
    const compensated_blas_complex_double *alpha,
    const compensated_blas_complex_double *a,
    const compensated_blas_blas_int *lda,
    const compensated_blas_complex_double *b,
    const compensated_blas_blas_int *ldb,
    const double *beta,
    compensated_blas_complex_double *c,
    const compensated_blas_blas_int *ldc
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

    backend().zher2k(uplo, trans, n_ilp64, k_ilp64, alpha, a, lda_ilp64, b, ldb_ilp64, beta, c, ldc_ilp64);
}

extern "C" void ztrsm_(
    const char *side,
    const char *uplo,
    const char *transa,
    const char *diag,
    const compensated_blas_blas_int *m,
    const compensated_blas_blas_int *n,
    const compensated_blas_complex_double *alpha,
    const compensated_blas_complex_double *a,
    const compensated_blas_blas_int *lda,
    compensated_blas_complex_double *b,
    const compensated_blas_blas_int *ldb
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

    backend().ztrsm(side, uplo, transa, diag, m_ilp64, n_ilp64, alpha, a, lda_ilp64, b, ldb_ilp64);
}

extern "C" void ztrmm_(
    const char *side,
    const char *uplo,
    const char *transa,
    const char *diag,
    const compensated_blas_blas_int *m,
    const compensated_blas_blas_int *n,
    const compensated_blas_complex_double *alpha,
    const compensated_blas_complex_double *a,
    const compensated_blas_blas_int *lda,
    compensated_blas_complex_double *b,
    const compensated_blas_blas_int *ldb
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

    backend().ztrmm(side, uplo, transa, diag, m_ilp64, n_ilp64, alpha, a, lda_ilp64, b, ldb_ilp64);
}
