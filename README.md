# ReidBLAS
BLAS implemented using tunable length compensation terms and optional deferred rounding


## xBLAT Integration Harness

The helper script `scripts/run_xblat_integration.sh` automates a clean configure
and install to a temporary prefix before running the `libblas-test` xBLAT
executables under `LD_PRELOAD`. Example:

```
./scripts/run_xblat_integration.sh --build-type RelWithDebInfo --timeout 120
```

Use `--keep-temp` to inspect the staged install and per-test logs, or
`--search-dir` to point at alternative BLAS test directories.


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
