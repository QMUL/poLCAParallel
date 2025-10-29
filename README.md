# poLCAParallel

## Polytomous Variable Latent Class Analysis

### With Bootstrap Likelihood Ratio Test

Sherman E. Lo, Queen Mary, University of London

A reimplementation of poLCA
\[[CRAN](https://cran.r-project.org/web/packages/poLCA/index.html),
[GitHub](https://github.com/dlinzer/poLCA)\] in C++. It attempts to reproduce
results and be as similar as possible to the original code, while running
faster, especially with multiple repetitions, by utilising multiple threads.

## About poLCAParallel

The package poLCAParallel reimplements the poLCA fitting, standard error
calculations, goodness of fit tests and the bootstrap log-likelihood ratio test
in C++. This was done using [Rcpp](https://cran.r-project.org/web/packages/Rcpp)
and [RcppArmadillo](https://cran.r-project.org/web/packages/RcppArmadillo) which
allows R to run fast C++ code. Additional notes include:

* The API remains the same as the original poLCA with a few additions
* It tries to reproduce results from the original poLCA
* The code uses [Armadillo](http://arma.sourceforge.net/) for linear algebra
* Multiple repetitions are done in parallel using
  [`std::thread`](https://www.cplusplus.com/reference/thread/) for multi-thread
  programming and [`std::mutex`](https://www.cplusplus.com/reference/mutex/) to
  prevent data races
* Direct inversion of matrices is avoided to improve numerical stability and
  performance
* Response probabilities are reordered to increase cache efficiency
* Use of [`std::map`](https://en.cppreference.com/w/cpp/container/map) for the
  chi-squared calculations to improve performance

Further reading is available on the
[QMUL ITS Research Blog](https://blog.hpc.qmul.ac.uk/speeding_up_r_packages.html).

## About poLCA

poLCA is a software package for the estimation of latent class models and latent
class regression models for polytomous outcome variables, implemented in the R
statistical computing environment.

Latent class analysis (also known as latent structure analysis) can be used to
identify clusters of similar "types" of individuals or observations from
multivariate categorical data, estimating the characteristics of these latent
groups, and returning the probability that each observation belongs to each
group. These models are also helpful in investigating sources of confounding and
nonindependence among a set of categorical variables, as well as for density
estimation in cross-classification tables. Typical applications include the
analysis of opinion surveys; rater agreement; lifestyle and consumer choice; and
other social and behavioral phenomena.

The basic latent class model is a finite mixture model in which the component
distributions are assumed to be multi-way cross-classification tables with all
variables mutually independent. The model stratifies the observed data by a
theoretical latent categorical variable, attempting to eliminate any spurious
relationships between the observed variables. The latent class regression model
makes it possible for the researcher to further estimate the effects of
covariates (or "concomitant" variables) on predicting latent class membership.

poLCA uses expectation-maximization and Newton-Raphson algorithms to find
maximum likelihood estimates of the parameters of the latent class and latent
class regression models.

## Recommended Installation Instructions

The easiest way to install poLCAParallel is to use R with
[remotes](https://cran.r-project.org/web/packages/remotes/index.html).

### Install From GitHub

Run the following in R to install the latest version

```r
remotes::install_github("QMUL/poLCAParallel@package")
```

or for a previous version, for example,

```r
remotes::install_github("QMUL/poLCAParallel@v1.2.4")
```

### Install From Releases

Download the `.zip` or `.tar.gz` file from the releases. Install it in R using

```r
remotes::install_local(<PATH TO .zip OR .tar.gz FILE>)
```

## User's Notes

### Citation

Please consider citing the corresponding
[QMUL ITS Research Blog](https://blog.hpc.qmul.ac.uk/speeding_up_r_packages.html)

* Lo, S.E. (2022). Speeding up and Parallelising R packages (using Rcpp and C++)
  | QMUL ITS Research Blog.
  [[link]](https://blog.hpc.qmul.ac.uk/speeding_up_r_packages.html)

and the publication below which this software was originally created for

* Eto F, Samuel M, Henkin R, Mahesh M, Ahmad T, et al. (2023). Ethnic
  differences in early onset multimorbidity and associations with health service
  use, long-term prescribing, years of life lost, and mortality: A
  cross-sectional study using clustering in the UK Clinical Practice Research
  Datalink. *PLOS Medicine,* 20(10): e1004300.
  <https://doi.org/10.1371/journal.pmed.1004300>

### Tips

* When using `model <- poLCAParallel::poLCA()`, set the parameters
  `calc.se=FALSE` and `calc.chisq=FALSE` to avoid doing standard error and
  goodness of fit calculations respectively. This will save time if you do not
  require those results. You can always calculate them afterwards using
  `model <- poLCAParallel::poLCAParallel.se(model)` and
  `model <- poLCAParallel::poLCAParallel.goodnessfit(model)`.
* Make use of multiple repetitions and threads. When using
  `poLCAParallel::poLCA()`, set `nrep=1` to do a test run and gauge how long it
  takes. Afterwards, set `nrep` to a bigger number to try different initial
  values in parallel.
* When using `poLCAParallel::poLCA()`, set `n.thread` to set the number of
  threads to be used by the computer. By default, it uses all detectable
  threads.
* There is an experimental option to use Laplace smoothing on the response
  probabilities when doing standard error calculations. This provides better
  numerical stability and avoids very small standard errors. To use it, either
  * In `poLCAParallel::poLCA()`, set `se.smooth=TRUE`
  * Or in `poLCAParallel::poLCAParallel.se()`, set `is_smooth=TRUE`
* When using the regression model, it is encouraged to normalise your data frame
  to provide better numerical stability.
* Use `set.seed()` before using `poLCAParallel::poLCA()` to set the seed for
  random number generation. This ensures reproducibility when reporting what
  seed you have used.

### Example Code

R scripts which compare poLCAParallel with poLCA are provided in `exec/`.
An example use of a bootstrap likelihood ratio test is shown in `exec/3_blrt.R`.

### Changes from the Original Code

* In `poLCAParallel::poLCA()`, the following arguments have been added:
  * `n.thread` is provided to specify the number of threads to use.
  * `calc.chisq` is provided to specify if you want to conduct goodness of fit
    tests or not.
  * `se.smooth` is provided if you wish to use Laplace smoothing on the response
    probabilities in the standard error calculations.
* The prior probabilities are a return value, accessible with `$prior`.
* The stopping condition of the EM algorithm has changed slightly. If the
  log-likelihood change after an iteration of EM is too small, the stopping
  condition is evaluated after the E step rather than the M step. This is so
  that the by-product of the E step is reused when calculating the
  log-likelihood.
* The Newton step uses a linear solver rather than directly inverting the
  Hessian matrix in the regression model.
* The output `probs.start` are the initial probabilities used to achieve the
  maximum log-likelihood from *any* repetition rather than from the first
  repetition.
* The output `eflag` is set to `TRUE` if *any* repetition has to be restarted,
  rather than the repetition which achieves maximum log-likelihood.
* The standard error is not calculated if `calc.se` is set to `FALSE` even in
  poLCA regression. Previously, the standard error is calculated regardless of
  `calc.se` in poLCA regression.
* In the standard error calculations, an SVD is done on the score matrix,
  rather than inverting the information matrix.
* Any errors in the input data will call `stop()` rather than return a `NULL`.
* No rounding in the return value `predcell`.

## Developer's and Maintainer's Notes

### Installing as a Developer

The following installation instructions are useful if you wish to develop the
code and install a locally modified version of the package.

Requires the R packages for compiling and testing:

* [Rcpp](https://cran.r-project.org/web/packages/Rcpp)
* [RcppArmadillo](https://cran.r-project.org/web/packages/RcppArmadillo)
* [roxygen2](https://cran.r-project.org/web/packages/roxygen2/index.html)
* [testthat](https://cran.r-project.org/web/packages/testthat/index.html)
* [usethis](https://cloud.r-project.org/web/packages/usethis/index.html)

Requires the dependent R packages:

* [MASS](https://cran.r-project.org/web/packages/MASS/index.html)
* [poLCA](https://cran.r-project.org/web/packages/poLCA/index.html)
* [scatterplot3d](https://cran.r-project.org/web/packages/scatterplot3d/)

Git clone this repository

```bash
git clone https://github.com/QMUL/poLCAParallel.git
```

and change directory into it

```bash
cd poLCAParallel
```

From there, in the repository root, run the following to generate additional
code and documentation so that the package can be compiled correctly

```bash
R -e "usethis::use_namespace()"
R -e "Rcpp::compileAttributes()"
R -e "roxygen2::roxygenize()"
```

Install the package using

```bash
R CMD INSTALL --preclean --no-multiarch .
```

### Testing

The testing of the C++ code is done using
[Catch2](https://github.com/catchorg/Catch2) and the R code using
[testthat](https://testthat.r-lib.org/). All test codes are in `tests/`.

#### C++ with Catch2

The tests for the C++ code are done by compiling the test code, isolated from
any R ecosystem, and running a compiled executable. It requires cmake, Catch2
and [armadillo](https://arma.sourceforge.net/). To compile the code, from the
repository root, make a new directory and use cmake inside it

```bash
mkdir build
cd build
cmake ..
cmake --build .
```

This will compile an executable called `test_polca_parallel`. Execute it to run
the tests. Pass names or tags to run specific tests, see `tests/*.cc`.

#### R with testthat

To test the R code, run the following at the repository root

```bash
R -e "testthat::test_local()"
```

### Apptainer

[Apptainer](https://apptainer.org/) definition files are provided, which can be
used to install the package inside a container. These may be useful for further
troubleshooting or development.

* The definition file `poLCAParallel.def` installs R and the package only. No
  version pinning
* The definition file `poLCAParallel-dev.def` installs the R package as well as
  generating documentation and running tests within the container. Versions of
  dependencies are pinned to the ones used during development or maintenance

To build the container, use the command (or similar)

```bash
apptainer build poLCAParallel-dev.sif poLCAParallel-dev.def
```

Within the container, the package is located in `/usr/src/poLCAParallel`. When
using the definition file `poLCAParallel-dev.def`, the C++ doxygen documentation
is located in `/usr/src/poLCAParallel/html`.

### Git/GitHub Workflow Guide

All generated documents and codes, eg from

```bash
R -e "Rcpp::compileAttributes('poLCAParallel')"
```

and

```bash
R -e "roxygen2::roxygenize('poLCAParallel')"
```

shall not be included in the `master` branch. Instead, they shall be in the
`package` branch so that this package can be installed using
`remotes::install_github("QMUL/poLCAParallel@package")`. This is to avoid
having duplicate documentation and generated code on the `master` branch.

Semantic versioning is used and tagged. Tags on the `master` branch shall have
`v` prepended and `-master` appended, eg. `v1.1.0-master`. The corresponding
tag on the `package` branch shall only have `v` prepended, eg. `v1.1.0`.

### Development Notes

* The likelihood calculation is done by iteratively multiplying probabilities
  together. In the commit `85ee419`, the multiplication starts from
  `DOUBLE_XMAX` to avoid underflows but was reverted. Consider investigating
  further in future releases.
* In the standard error calculations, the score matrix is typically
  ill-conditioned. Consider pre-conditioning the matrix.
* In the poLCA regression model, consider using multiple Newton steps instead
  of one single step in the EM algorithm.

### Actions for the Next Minor Version(s)

* Add a feature where the likelihood calculation can be optionally done by
  summing the log probabilities rather than multiplying probabilities together.
  This should avoid underflows, especially when there are a large number of
  manifest variables (aka categories) or very small probabilities. Though it
  should be noted that working in log space is slower.

### Actions for the Next Major Version

* The R package MASS is not required as a prerequisite.
* The default value for `n.thread` should be `1` instead of
  `parallel::detectCores()`

The following R functions, many of which are internal, are marked as deprecated
and should be deleted

* `poLCA.se()` and `poLCA.dLL2dBeta.C()` - no longer needed because the standard
  error calculations are reimplemented in `poLCAParallel.se()`
* `poLCA.probHat.C` - no longer needed because the goodness of fit test is
    reimplemented in `goodness_fit.cc`
* `poLCA.postClass.C()` and `poLCA.ylik.C()` - no longer needed and
  reimplemented in `polca_rcpp.cc`
* `poLCA.vectorize()` and `poLCA.unvectorize()` - no longer needed and
  reimplemented in `poLCAParallel.vectorize()` and `poLCAParallel.unvectorize()`
  respectively

All C code in `poLCA.C` is deprecated because they are reimplemented in C++.

The R code should follow the Tidyverse style guide. In particular, variables,
functions and parameters should be in snake case. This will result in

* Removing the `poLCA.` and `poLCAParallel.` prefix in function and file names
* Using a underscore instead of a dot in variable and parameter names, for
  example, `na.rm` should be called `na_rm`

### C++ Style Guide

There was an attempt to use the
[Google C++ style guide](https://google.github.io/styleguide/cppguide.html).

### C++ Source Code Documentation

The C++ code documentation can be created with [Doxygen](https://doxygen.nl/)
by running

```console
doxygen
```

and viewed at `html/index.html`.

## References

* Bandeen-roche, K., Miglioretti, D. L., Zeger, S. L., and Rathouz, P. J.
  (1997). Latent variable regression for multiple discrete outcomes. *Journal of
  the American Statistical Association*, 92(440):1375–1386.
  [[link]](https://doi.org/10.1080/01621459.1997.10473658)
* Dziak, J. J., Lanza, S. T., & Tan, X. (2014). Effect size, statistical power,
  and sample size requirements for the bootstrap likelihood ratio test in latent
  class analysis. *Structural Equation Modeling: A Multidisciplinary Journal*,
  21(4):534-552.
  [[link]](https://www.tandfonline.com/doi/full/10.1080/10705511.2014.919819?casa_token=LgaSzKeeB8MAAAAA%3AB80XwZEIkLOIVsD4Gvp6O0gfktOnIqA6dOBBvUZIjjhs-7ilLIZJC_TmxCh8Umh45d0sWez4-em9)
* Linzer, D.A. & Lewis, J. (2013). poLCA: Polytomous Variable Latent
  Class Analysis. R package version 1.4.
  [[link]](https://github.com/dlinzer/poLCA)
* Linzer, D.A. & Lewis, J.B. (2011). poLCA: An R package for polytomous
  variable latent class analysis. *Journal of Statistical Software*,
  42(10): 1-29.
  [[link]](http://www.jstatsoft.org/v42/i10)

## License

The software is under the GNU GPL 2.0 license, as with the original poLCA code,
stated in their
[documentation](https://cran.r-project.org/web/packages/poLCA/index.html).
