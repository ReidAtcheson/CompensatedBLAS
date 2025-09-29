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

No critical path allocations or avoidable copies. Don't copy for example packed storage to general storage
(or banded to general) in order to reuse general case algorithm.

The main exceptions to the allocation rule above is
  1. to preallocate space for any necessary tiling (done once in thread local style, not repeatedly per call)
  2. allocating deferred compensation terms greedily at the time the user has requested them
  3. preallocating space for normal compensated arithmetic (just like in 1: thread local style, not every call)

BLAS routines should be fully compatible with netlib blas and pass the `xblat*` test suite, this includes

parameter checking and error conditions.

use snake case wherever possible. classes are also snake case with `_t` suffix, because I'm a psycho.

## Lessons Learned Implementing BLAS Routines

Written by codex 

- **Wire up parameter validation first.** Netlib’s xBLAT error-exit cases call routines with invalid `trans`, dimensions, leading dimensions, or increments and expect an early return through `xerbla`. Mirror LAPACK’s documented index ordering, report 1-based indices, and ensure you do this before touching data so the checks don’t accidentally pass.
- **Surface `xerbla` from our shared library.** The harness uses `LD_PRELOAD`; if we don’t provide a forwarding shim the error tests fail. Export an ILP64-friendly `xerbla_` that forwards to the next provider (or prints) so both LP64/ILP64 callers see the same behaviour.
- **Respect netlib’s indexing semantics.** Compute `nrow`/`ncol` based on transpose mode, require `lda >= max(1, nrow)`, and adjust vector pointers when increments are negative. xBLAT explicitly probes these edge conditions.
- **Keep compensation bins aligned with storage.** When deferred rounding is active, fetch bins via the runtime descriptors and update them in-place. For immediate paths, reuse thread-local scratch bins and finalise after each accumulation step; this avoids extra allocations and ensures deterministic results.
- **Match reference arithmetic ordering.** Scale `y` by `beta`, form the dot product row/column-wise according to transpose flags, and accumulate with compensated `two_sum` reductions. Skipping `beta` zeroing or reordering multiplies shows up quickly in the residual checks.
- **Test incrementally.** Run unit tests for immediate/deferred paths, then `run_xblat_integration.sh`. The latter highlights both error exits and numerical mismatches early, saving painful debugging once multiple kernels interact.
- **Fence higher precision scratch work.** When we collapse compensated values (e.g., to seed a division) keep the wider accumulator hidden inside helpers like `reconstruct_compensated_value`; all algorithmic paths should stay in working precision using the primary slot plus compensation bins so future refactors can swap the scratch type without touching the cores.
