#include "impl/compensated_blas_backend_ilp64.hpp"

#include <type_traits>

namespace compensated_blas::impl::detail {

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

}  // namespace compensated_blas::impl::detail

namespace {

class EmptyBackend final : public compensated_blas::impl::BlasBackend {
public:
#define COMPENSATEDBLAS_ILP64_FUNCTION(return_type, name, signature, args) \
    return_type name signature override {                                   \
        compensated_blas::impl::detail::ignore args;                        \
        if constexpr (!std::is_void_v<return_type>) {                       \
            return compensated_blas::impl::detail::default_value<return_type>(); \
        }                                                                   \
    }

#include "impl/compensated_blas_ilp64_functions.def"

#undef COMPENSATEDBLAS_ILP64_FUNCTION
};

EmptyBackend empty_backend_instance{};
compensated_blas::impl::BlasBackend *active_backend_ptr = &empty_backend_instance;

}  // namespace

namespace compensated_blas::impl {

BlasBackend &get_active_backend() {
    return *::active_backend_ptr;
}

void set_active_backend(BlasBackend *backend) {
    ::active_backend_ptr = backend != nullptr ? backend : &::empty_backend_instance;
}

}  // namespace compensated_blas::impl
