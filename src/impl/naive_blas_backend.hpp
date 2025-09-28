#pragma once

#include "impl/compensated_blas_backend_ilp64.hpp"

namespace compensated_blas::impl::detail {

blas_backend_t *acquire_naive_backend();

}  // namespace compensated_blas::impl::detail
