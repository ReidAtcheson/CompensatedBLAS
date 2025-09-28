#include "impl/reidblas_backend_ilp64.hpp"

#include <type_traits>

namespace reidblas::impl::detail {

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

}  // namespace reidblas::impl::detail

namespace {

class EmptyBackend final : public reidblas::impl::BlasBackend {
public:
#define REIDBLAS_ILP64_FUNCTION(return_type, name, signature, args) \
    return_type name signature override {                           \
        reidblas::impl::detail::ignore args;                        \
        if constexpr (!std::is_void_v<return_type>) {               \
            return reidblas::impl::detail::default_value<return_type>(); \
        }                                                           \
    }

#include "impl/reidblas_ilp64_functions.def"

#undef REIDBLAS_ILP64_FUNCTION
};

EmptyBackend empty_backend_instance{};
reidblas::impl::BlasBackend *active_backend_ptr = &empty_backend_instance;

}  // namespace

namespace reidblas::impl {

BlasBackend &get_active_backend() {
    return *::active_backend_ptr;
}

void set_active_backend(BlasBackend *backend) {
    ::active_backend_ptr = backend != nullptr ? backend : &::empty_backend_instance;
}

}  // namespace reidblas::impl
