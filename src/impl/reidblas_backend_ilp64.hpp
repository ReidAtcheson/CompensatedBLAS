#pragma once

#include <cstdint>

#include "reidblas_blas_ilp64.h"

namespace reidblas::impl {

class BlasBackend {
public:
    virtual ~BlasBackend() = default;

#define REIDBLAS_ILP64_FUNCTION(return_type, name, signature, args) \
    virtual return_type name signature = 0;

#include "reidblas_ilp64_functions.def"

#undef REIDBLAS_ILP64_FUNCTION
};

BlasBackend &get_active_backend();
void set_active_backend(BlasBackend *backend);

inline BlasBackend &get_active_ilp64_backend() {
    return get_active_backend();
}

inline void set_active_ilp64_backend(BlasBackend *backend) {
    set_active_backend(backend);
}

}  // namespace reidblas::impl
