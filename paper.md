---
title: 'poLCAParallel: A C++ reimplementation and parallelisation of poLCA'
authors:
  - given-names: Sherman Edea
    surname: Lo
    affiliation: "1"
affiliations:
  - index: 1
    name: Queen Mary, University of London, UK
date: 27 October 2026
bibliography: paper.bib
---

The R package poLCAParallel is a fork and reimplementation of an existing R
package, poLCA. In this fork, the underlying algorithm was parallelised and
rewritten in C++. This offers a significant speedup, making it suitable for
large datasets using high-performance computing (HPC). The API was kept the same
as the original code to allow poLCAParallel to be a seamless drop-in replacement
of poLCA.

# Statement of need

The R [@r2025statistical] package poLCA [@linzer2011polca; @linzer2026polca]
stands for polytomous variable latent class analysis. This studies a mixture
model of random multivariate polytomous variables [@goodman1974exploratory;
@bandeen1997latent] and fits the model to observed data.

Let $Y$ be the multivariate polytomous random variable of interest with $K$
independent categories. This can be represented as a random vector $Y=\left(Y_1,
Y_2, \cdots, Y_K\right)$. Each category can take one of many independent
outcomes with values $Y_k \in\left\{1,2,\cdots, L_k\right\}$ for
$k=1,2,\cdots,K$ where $L_k$ is the number of outcomes for the $k$th category.
The outcomes do not necessarily need to have an ordering.

For the latent class model, with the number of classes $J$ and a prior
probability for each class $\pi_j$ for $j=1,2,\cdots,J$ where $0<\pi_j<1$ and
$\sum_{j=1}^J\pi_j=1$, the probability mass function is given as

$$ \mathbb{P}\left(Y=y\right)=\sum_{j=1}^J \pi_j \prod_{k=1}^K \prod_{l=1}^{L_k}
p_{j,k,l}^{\delta\left(y_k, l\right)} $$

where $y=\left(y_1,y_2,\cdots,y_K\right)$ is a possible instance of $Y$,
$p_{j,k,l}$ is the outcome probability for class $j$, category $k$ and outcome
$l$, and $\delta(\cdot ,\cdot)$ is the Kronecker delta. The outcome
probabilities are such that $0<p_{j,k,l}<1$ for all $j$, $k$ and $l$ denoted
above. They are also normalised such that $\sum_{l=1}^{L_k}p_{j,k,l}=1$ for all
$j$ and $k$ denoted above.

Suppose a dataset of $n$ observations of $Y$ was obtained, each observation
denoted as $y^{(1)}, y^{(2)},\cdots,y^{(n)}$. The log-likelihood is given as

$$ \ln L = \sum_{i=1}^n \ln {\left[ \mathbb{P}\left(Y=y^{(i)}\right)\right]} $$

For a given dataset, the expectation–maximisation (EM) algorithm
[@dempster1977likelihood] estimates the parameters of the model $p_{j,k,l}$ and
$\pi_j$ for all $j$, $k$ and $l$ as denoted previously. This is done by
maximising the log-likelihood using an iterative procedure, alternating between
taking an E step and an M step.

In the E step, the posterior probabilities for each observation belonging to
different classes, given the model parameters, are estimated. In the M step, the
values of $p_{j,k,l}$ and $\pi_j$ are updated to maximise the log-likelihood
given those posterior probabilities. The E and M steps are repeated until some
convergence criteria are met. The algorithm is known to converge to a local
maximum log-likelihood. Thus, different initial values should be trialled to
find solutions that maximise the log-likelihood globally.

The package poLCA also offers models where $\pi_j$ is a linear function of
observed predictor vectors [@bandeen1997latent]. The package can also calculate
the standard errors [@rao1992information; @cramer1999mathematical;
@bandeen1997latent] and goodness-of-fit [@pearson1900criterion;
@wilks1938likelihood] statistics.

The need for optimising poLCA was identified by an HPC user at Queen Mary,
University of London (QMUL). The HPC at QMUL is called Apocrita
[@king2017apocrita] and offers CPU and GPU compute facilities. It was identified
that a user’s job using poLCA took many days and exceeded the 10-day maximum
runtime.

# State of the field

Within the R ecosystem, poLCA [@linzer2011polca; @linzer2026polca] is heavily
used by statisticians and in data analysis, such as in @schimmenti2018trauma,
@tromp2018multimorbidity, @godfredcato2020covid, @whitaker2022covid,
@romiti2024comorbidities and @xiao2024suicide. The majority of the package is
written in R, with a few small functions in C.

The package poLCA showed some significant bottlenecks when our user used it for
a dataset consisting of about half a million observations. The code in the
package was designed to be run in serial. Critically, it was found that
inefficient operations were used which can bottleneck the package. For example,
in the standard error calculations, the pseudoinverse [@penrose1995inverse;
@barata2012pseudoinverse] was used to directly invert a matrix in a linear
problem involving symmetric matrices. In the goodness-of-fit calculations, a
sorting algorithm was used to find unique observations of $Y$ in the dataset,
rather than using tools specifically for finding unique items. These
inefficiencies caused the user's code to appear stalled during the 10-day
runtime.

We contributed to poLCA by implementing our optimisation as a fork. The
optimisations would change the internal code completely and poLCAParallel could
feasibly stand alone as its own package. This would avoid overwhelming the
original code repository and authors with large code suggestions and pull
requests.

# Software design

The underlying EM algorithm was written in C++ [@iso2020cpp] and parallelised by
trialing out different initial values in parallel. Hence, the package was named
poLCAParallel.

It was designed to be a drop-in replacement for poLCA, keeping the same API as
the original package with a few additional optional parameters and functions.
This allows regular users of poLCA to adopt poLCAParallel by only changing a few
lines of code.

The C++ code uses Armadillo [@sanderson2019matrices; @sanderson2025armadillo]
for linear algebra, which offers faster speeds than an implementation in R
[@eddelbuettel2014rcpparmadillo]. The C++ component of the code was written so
that it can be compiled on its own outside the R ecosystem when needed. The R
package RcppArmadillo [@eddelbuettel2014rcpparmadillo;
@eddelbuettel2026rcpparmadillo] was used so that the C++ implementation can be
compiled with Armadillo and run from R using Rcpp [@eddelbuettel2011rcpp;
@eddelbuettel2013rcpp; @eddelbuettel2018rcpp; @eddelbuettel2026rcpp].

To improve cache efficiency [@sehr2018cpp], some matrices and vectors from R
were reordered. For example, in the log-likelihood calculation, which involves
iterating $p_{j,k,l}$ over $j$, $k$ and $l$, the outcome probabilities were
reordered such that neighbouring elements in memory are accessed during
iterations, improving performance.

In terms of parallelisation, the class `std::thread` from the standard library
`<thread>` was used [@sehr2018cpp; @iso2020cpp]. Each thread uses different
initial values for the EM algorithm in parallel. By using more initial values,
the solution with the highest log-likelihood is more likely to achieve a global
maximum. And by running from these initial values in parallel, such a solution
can be calculated faster.

Another good candidate tool for multi-thread programming is OpenMP
[@dagum1998openmp] because it provides programmers a way to write thread-safe
code without having to use mutexes. However, using the C++ standard library was
decided instead because it was more familiar to the author at the time.

The standard error calculations and goodness-of-fit tests were also
reimplemented in C++. Many of the linear algebra operations were rewritten to
avoid direct or pseudo-inversion of a matrix. Because covariance matrices are
symmetric, this property was used to replace direct or pseudo-inversion of
matrices with appropriate Armadillo linear solvers and decompositions. This
offers better numerical stability and faster speeds [@kiusalaas2013numerical].

The goodness-of-fit statistics require counts of all unique observations of $Y$.
In the original code, the response variables were sorted and then counted
individually. The package poLCAParallel offers significant speed-up by using a
hash map [@sehr2018cpp] to keep track of and look up all unique instances of
$Y$.

## Documentation

The documentation in the R code and C++ code can be generated using the R
package roxygen2 [@wickham2026roxygen2] and the software doxygen
[@vanheesch2026doxygen], respectively. The C++ documentation is mainly aimed at
future developers and maintainers.

## Testing

The C++ component was designed to be isolated from the R ecosystem, and can be
compiled and tested by itself using Catch2 [@catch22026]. The package testthat
[@wickham2011testthat; @wickham2026testthat] was used to test the entire package
through the R interface. This allows the testing to be done at different levels,
pinpointing problems, should one occur, in the pipeline, whether it’s in the R
or C++ component.

Because poLCAParallel is a drop-in replacement for poLCA, tests include ensuring
the output of poLCA and poLCAParallel are the same or numerically similar. For
example, objects should have the same property names and content. However,
changes to the linear algebra operations and improvements to the numerical
stability may mean some standard error results will diverge from the original
code and cannot be compared for similarity in tests.

## Package availability

The package is available in a [GitHub
repository](https://github.com/QMUL/poLCAParallel) and from CRAN
[@lo2026polcaparallel]. The package poLCAParallel uses roxygen2 and
RcppArmadillo, which generate documentation and code for it to work in R. To
avoid committing and tracking changes of generated code, the default main branch
does not include any generated code. A branch, called `package`, was created,
which includes all generated code so that the package can be used from R
straight from that branch. A caveat is that if a user wishes to install
poLCAParallel directly via GitHub in R, they will have to specify the branch
name, such as `remotes::install_github("QMUL/poLCAParallel@package")`.

# Research impact statement

The Apocrita user, whose code was bottlenecked by poLCA, managed to run their
code using poLCAParallel in the order of hours rather than days. In
@eto2023ethnic, they acknowledged poLCAParallel and the author.

Apocrita’s main CPU compute facility consists of two Intel Xeon Platinum 8268
CPUs per node. A job can use up to 48 cores per node. A benchmark was conducted
using the user's data. The data is confidential but it has the following summary
statistics:

- $n = 489\,478$
- $K = 204$
- $L_k = 2$ for all $k$

A model with $J=10$ classes was fitted, using 48 initial values. By default,
poLCA calculates the standard errors and goodness-of-fit statistics in addition
to the model fit, so these will be included in the benchmarks.

To allow for poLCA to run in a feasible time, a random subsample of $n=10\,000$
was taken and used in the benchmarks. They are compared with poLCAParallel using
different numbers of threads, as shown in Figure \ref{fig:subsample}. A speedup
of at least 6&times; was observed in the single-thread performance when using
poLCAParallel over poLCA, with further improvements when using more threads.

![Benchmarks comparing poLCA with poLCAParallel fitting a model with $J=10$
classes onto a random subsample $n=10\,000$ of the data. For poLCAParallel,
different numbers of threads were used.\label{fig:subsample}](subsample.svg){
width=80% }

For the entire dataset of $n = 489\,478$, poLCA would take days to complete,
thus omitted in further benchmarks. Figure \ref{fig:full} shows the benchmarks
of poLCAParallel when fitting the model onto the entire dataset. The entire fit
was completed in the order of hours rather than days when using poLCAParallel.
Whilst using more threads does improve performance, diminishing returns were
observed when the entire node of 48 cores was used. In this realm, it is
bottlenecked by single-thread calculations such as standard errors and
goodness-of-fit statistics.

![Benchmark of poLCAParallel fitting a model with $J=10$ classes onto the entire
dataset. Different numbers of threads were used. \label{fig:full}](full.svg){
width=80% }

During development of poLCAParallel, blog posts were written about the process
on our departmental website, such as @lo2022speeding and @lo2026modernising. The
former was cited in @alvarezgalvez2023multimorbidity, @zhou2023latent,
@priya2026subtyping. Otherwise, the package was mentioned in @kassasombo2026icu
and @zhao2026latent. This shows that the package poLCAParallel was reached out
to and used by researchers who were also bottlenecked by poLCA.

# AI usage disclosure

No artificial intelligence tools were used in the creation of this paper or the
R package poLCAParallel.
