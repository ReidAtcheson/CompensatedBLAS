#include "impl/backend_stub.hpp"
#include "impl/naive_blas_backend.hpp"

namespace {

class empty_backend_t final : public compensated_blas::impl::stub_backend_t {};

empty_backend_t empty_backend_instance{};
compensated_blas::impl::blas_backend_t *active_backend_ptr =
    compensated_blas::impl::detail::acquire_naive_backend();

}  // namespace

namespace compensated_blas::impl {

blas_backend_t &get_active_backend() {
    return *::active_backend_ptr;
}

void set_active_backend(blas_backend_t *backend) {
    ::active_backend_ptr = backend != nullptr ? backend : &::empty_backend_instance;
}

}  // namespace compensated_blas::impl
