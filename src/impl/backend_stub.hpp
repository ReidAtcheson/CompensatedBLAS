#pragma once

#include <type_traits>

#include "impl/compensated_blas_backend_ilp64.hpp"

namespace compensated_blas::impl {

namespace detail {

template <typename... Args>
inline void ignore(Args &&...args) {
    (void)sizeof...(args);
    ((void)args, ...);
}

template <typename T>
constexpr T default_value() {
    return T{};
}

template <>
inline void default_value<void>() {}

}  // namespace detail

class stub_backend_t : public blas_backend_t {
public:
    ~stub_backend_t() override = default;

#define COMPENSATEDBLAS_ILP64_FUNCTION(return_type, name, signature, args) \
    return_type name signature override;

#include "impl/compensated_blas_ilp64_functions.def"

#undef COMPENSATEDBLAS_ILP64_FUNCTION
};

#define COMPENSATEDBLAS_ILP64_FUNCTION(return_type, name, signature, args)                 \
    inline return_type stub_backend_t::name signature {                                   \
        detail::ignore args;                                                               \
        if constexpr (!std::is_void_v<return_type>) {                                      \
            return detail::default_value<return_type>();                                   \
        }                                                                                  \
    }

#include "impl/compensated_blas_ilp64_functions.def"

#undef COMPENSATEDBLAS_ILP64_FUNCTION

}  // namespace compensated_blas::impl
