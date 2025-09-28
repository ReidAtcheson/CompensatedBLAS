#include <type_traits>

#include "impl/compensated_blas_backend_ilp64.hpp"

#if COMPENSATEDBLAS_USE_ILP64

#define COMPENSATEDBLAS_ILP64_FUNCTION(return_type, name, signature, args)       \
    extern "C" return_type name##_ signature {                                  \
        auto &backend = compensated_blas::impl::get_active_backend();            \
        if constexpr (std::is_void_v<return_type>) {                             \
            backend.name args;                                                   \
        } else {                                                                 \
            return backend.name args;                                            \
        }                                                                        \
    }

#include "impl/compensated_blas_ilp64_functions.def"

#undef COMPENSATEDBLAS_ILP64_FUNCTION

#endif  // COMPENSATEDBLAS_USE_ILP64
