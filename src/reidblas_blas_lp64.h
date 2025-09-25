#pragma once

#include <stdint.h>

// Fortran-style BLAS declarations using 32-bit indices.
#define REIDBLAS_INDEX_TYPE int32_t
#define REIDBLAS_INT_ALIAS reidblas_blas_int
#include "reidblas_blas_template.h"
#undef REIDBLAS_INT_ALIAS
#undef REIDBLAS_INDEX_TYPE
