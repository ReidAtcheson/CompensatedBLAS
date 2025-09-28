#pragma once

#include <stdint.h>

// Fortran-style BLAS declarations using 32-bit indices.
#define COMPENSATEDBLAS_INDEX_TYPE int32_t
#define COMPENSATEDBLAS_INT_ALIAS compensated_blas_blas_int
#include "compensated_blas_template.h"
#undef COMPENSATEDBLAS_INT_ALIAS
#undef COMPENSATEDBLAS_INDEX_TYPE
