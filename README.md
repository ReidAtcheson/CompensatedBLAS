# CompensatedBLAS
BLAS implemented using tunable length compensation terms and optional deferred rounding



# Disclaimer

This code is largely written by [codex](https://openai.com/codex/) where the primary input from me was the addition
of compensated arithmetic. The idea of compensated arithmetic is not novel in the context of BLAS having been
explored extensively in the literature (see references below). Furthermore I have expended _zero effort_ to
optimize these routines. They likely are _many times_ slower than associated optimized routines certainly
from vendor libraries like MKL, but also probably slower than available compensated BLAS libraries.

All of the above is to say: this is probably not fit for any real practical purpose in its current state. it is an experimental project
whose primary purpose was to satisfy my own curiosity.

# Features 

So why did I do this? I thought it would be nice to have a compensated blas library
that is fully binary compatible with ordinary BLAS, enabled tunable compensated accumulators,
and also enabled a sidecar where we could defer rounding across BLAS calls for those routines
which  can be called to accumulate repeatedly into the same output matrix.

So to summarize:

1. Fully binary compatible with blas (you can LD_PRELOAD this library into a project that links blas, and it will work).
2. Optional deferred rounding for routines that accumulate into an output matrix. e.g. GEMM,SYRK,axpy, etc..


# Current BLAS support

Codex has made steady and consistent progress and I only halted it because it was burning my token budget and
I wanted to experiment with other things too. I will probably return to this and complete the naive implementation.


1. All level 1 blas supported for real,complex,32/64bit
2. All level 2 blas supported for real 32/64bitA
3. `xSYRK` supported in level 3 blas for real,complex,32/64bit






# xBLAT Integration Harness

The helper script `scripts/run_xblat_integration.sh` automates a clean configure
and install to a temporary prefix before running the `libblas-test` Fortran
xBLAT executables under `LD_PRELOAD`. Install the canonical test binaries and
input decks with `apt-get install libblas-test` on Debian/Ubuntu systems. Example:

```
./scripts/run_xblat_integration.sh --build-type RelWithDebInfo --timeout 120
```

Use `--keep-temp` to inspect the staged install and per-test logs, or
`--search-dir` to point at alternative BLAS test directories.


# Adding a Backend

Backends are modeled as C++ subclasses of `compensated_blas::impl::BlasBackend`. Each
BLAS routine is a pure virtual method; the default implementation shipped with
the library (`EmptyBackend`) simply ignores its arguments and, for non-`void`
functions, returns a zero-initialized value.

To introduce a new backend:

1. **Implement the class** – Create a translation unit under `src/impl/` and
   derive from `BlasBackend`. The easiest way to provide the method bodies is to
   reuse the master macro list:
   ```c++
   class MyBackend final : public compensated_blas::impl::BlasBackend {
   public:
   #define COMPENSATEDBLAS_ILP64_FUNCTION(rt, name, signature, args) \
       rt name signature override;  // declare overrides
   #include "impl/compensated_blas_ilp64_functions.def"
   #undef COMPENSATEDBLAS_ILP64_FUNCTION
   };
   ```
   Then include the same list again to emit definitions, or write the bodies
   manually if you prefer.

2. **Register the backend** – Expose a static instance (or factory) and call
   `compensated_blas::impl::set_active_backend(&instance);` during library
   initialisation or from your own entry point. Tests can also substitute
   backends using this hook.

3. **Add the file to CMake** – Append your new implementation file to
   `COMPENSATEDBLAS_IMPL_SOURCES` in the top-level `CMakeLists.txt` so it is compiled
   into the `compensatedblas_impl` target (unit tests link against the same static
   archive).

This design keeps the exported BLAS symbols as plain C entry points while
allowing ergonomic C++ implementations and backend selection at runtime.


# References 

Extended / Mixed-Precision BLAS (XBLAS)

Li, X. S., Demmel, J., Bailey, D., Henry, G., Hida, Y., Iskandar, J., Kahan, W., et al. “Design, Implementation and Testing of Extended and Mixed Precision BLAS.” ACM Transactions on Mathematical Software 28(2), 2002. DOI: 10.1145/567806.567808.
PDF: https://portal.nersc.gov/project/sparse/xiaoye-web/p152-s_li.pdf
ACM page: https://dl.acm.org/doi/10.1145/567806.567808

Reproducible BLAS (binned accumulators / ReproBLAS)

Ahrens, P., Demmel, J., Nguyen, H. D., Riedy, E. “Efficient Reproducible Floating Point Summation and BLAS.” UCB/EECS-2015-229, 2015.
PDF: https://www2.eecs.berkeley.edu/Pubs/TechRpts/2015/EECS-2015-229.pdf

Ahrens, P., Demmel, J., Nguyen, H. D. “Algorithms for Efficient Reproducible Floating Point Summation.” ACM TOMS 46(3), 2020, Article 22. DOI: 10.1145/3389360.
PDF: https://people.eecs.berkeley.edu/~demmel/ma221_Fall23/J115_Efficient_Reproducible_Summation_TOMS_2020.pdf
ACM page: https://dl.acm.org/doi/10.1145/3389360

ExBLAS (superaccumulators + EFT; correct rounding & reproducibility)

Iakymchuk, R., Collange, S., Defour, D., Graillat, S. “ExBLAS: Reproducible and Accurate BLAS Library.” Poster/overview, 2015.
Poster: https://www-pequan.lip6.fr/~graillat/papers/poster.raim2015.pdf

Iakymchuk, R. “ExBLAS: Reproducible and Accurate BLAS Library.” NIST talk slides, 2015.
Slides: https://www.nist.gov/document/nre-2015-04-iakymchukpdf

Ozaki scheme (tunable up to correct rounding; built atop vendor BLAS)

Mukunoki, D., Ogita, T., Ozaki, K. “High-Performance Implementation of Reproducible and Accurate Matrix Multiplication on GPUs.” PMAA 2018 (slides/paper).
PDF: https://www-pequan.lip6.fr/~graillat/papers/pmaa18mukunoki.pdf

Mukunoki, D., et al. “Accurate Matrix Multiplication on Binary128 Format Using the Ozaki Scheme.” Proc. PASC ’21, 2021. DOI: 10.1145/3472456.3472493.
ACM page: https://dl.acm.org/doi/10.1145/3472456.3472493

IEEE-754 (Augmented arithmetic / twoSum-twoProd in the standard)

IEEE Std 754-2019. IEEE Standard for Floating-Point Arithmetic. Approved June 13, 2019.
PDF: https://www-users.cse.umn.edu/~vinals/tspot_files/phys4041/2020/IEEE%20Standard%20754-2019.pdf

IEEE MSC background note: “ANSI/IEEE-Std-754-2019 Background” (augmented {add, sub, mul} overview).
Page: https://grouper.ieee.org/groups/msc/ANSI_IEEE-Std-754-2019/background/
PDF summary: https://grouper.ieee.org/groups/msc/ANSI_IEEE-Std-754-2019/background/ieee-computer.pdf

Demmel, J. “A New IEEE 754 Standard for Floating-Point Arithmetic in an Ever-Changing World.” SIAM News, 2021.
Article: https://www.siam.org/publications/siam-news/articles/a-new-ieee-754-standard-for-floating-point-arithmetic-in-an-ever-changing-world/

Exact dot product / long accumulators (theory & hardware motivation)

Kulisch, U., Snyder, V. “The Exact Dot Product as Basic Tool for Long Interval Arithmetic.” Computing 91, 307–313 (2011). DOI: 10.1007/s00607-010-0127-7.
Springer page: https://link.springer.com/article/10.1007/s00607-010-0127-7
Open PDF mirror: https://scispace.com/pdf/the-exact-dot-product-as-basic-tool-for-long-interval-224g2jfuw3.pdf

Kulisch, U. “Very Fast and Exact Accumulation of Products.” Computing 91, 2011. DOI: 10.1007/s00607-010-0131-y.
Springer page: https://link.springer.com/article/10.1007/s00607-010-0131-y

Accurate/faithful summation foundations (NearSum etc.)

Rump, S. M. “Accurate Floating-Point Summation Part I: Faithful Rounding.” SIAM J. Sci. Comput., 2006. DOI: 10.1137/050645671.
DOI page: https://epubs.siam.org/doi/10.1137/050645671

Rump, S. M., Ogita, T., Oishi, S. “Accurate Floating-Point Summation Part II: Sign, K-Fold Faithful, and Rounding to Nearest.” SIAM J. Sci. Comput., 2008. DOI: 10.1137/07068816X.
DOI page: https://epubs.siam.org/doi/10.1137/07068816X
(Alt PDF for Part II): https://www.tuhh.de/ti3/paper/rump/RuOgOi07II.pdf
