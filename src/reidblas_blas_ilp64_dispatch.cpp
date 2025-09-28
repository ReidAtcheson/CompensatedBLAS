#include <type_traits>

#include "impl/reidblas_backend_ilp64.hpp"

#if REIDBLAS_USE_ILP64

#define REIDBLAS_ILP64_FUNCTION(return_type, name, signature, args)              \
    extern "C" return_type name##_ signature {                                  \
        auto &backend = reidblas::impl::get_active_backend();                    \
        if constexpr (std::is_void_v<return_type>) {                             \
            backend.name args;                                                   \
        } else {                                                                 \
            return backend.name args;                                            \
        }                                                                        \
    }

#include "impl/reidblas_ilp64_functions.def"

#undef REIDBLAS_ILP64_FUNCTION

#endif  // REIDBLAS_USE_ILP64
