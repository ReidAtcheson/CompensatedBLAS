# CompensatedBLAS

BLAS implemented using tunable length compensation terms and optional deferred rounding


## Testing

```
./scripts/run_xblat_integration.shA
```
builds our public library interface (binary compatible BLAS) and LD_PRELOADs it
into the netlib xblat test suite. We expect many failures from this until
all of BLAS has been implemented.

```
./scripts/run_tests.sh
```
builds and runs our unit tests (gtest)
## Style

use snake case wherever possible. classes are also snake case with `_t` suffix, because I'm a psycho.

