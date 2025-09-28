#pragma once

#include <cstdint>

namespace compensated_blas {

#if COMPENSATEDBLAS_USE_ILP64
using index_t = std::int64_t;
#else
using index_t = int;
#endif

index_t identity(index_t value);

}  // namespace compensated_blas
