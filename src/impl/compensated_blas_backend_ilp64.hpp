#pragma once

#include <cstdint>

#include "compensated_blas_ilp64.h"

namespace compensated_blas::impl {

class blas_backend_t {
public:
    virtual ~blas_backend_t() = default;

#define COMPENSATEDBLAS_ILP64_FUNCTION(return_type, name, signature, args) \
    virtual return_type name signature = 0;

#include "compensated_blas_ilp64_functions.def"

#undef COMPENSATEDBLAS_ILP64_FUNCTION
};

blas_backend_t &get_active_backend();
void set_active_backend(blas_backend_t *backend);

inline blas_backend_t &get_active_ilp64_backend() {
    return get_active_backend();
}

inline void set_active_ilp64_backend(blas_backend_t *backend) {
    set_active_backend(backend);
}

}  // namespace compensated_blas::impl
