#' Estimates latent class and latent class regression models for polytomous
#' outcome variables
#'
#' Latent class analysis, also known as latent structure analysis, is a
#' technique for the analysis of clustering among observations in multi-way
#' tables of qualitative/categorical variables. The central idea is to fit a
#' model in which any confounding between the manifest variables can be
#' explained by a single unobserved "latent" categorical variable. `poLCA` uses
#' the assumption of local independence to estimate a mixture model of latent
#' multi-way tables, the number of which (`nclass`) is specified by the user.
#' Estimated parameters include the class-conditional response probabilities for
#' each manifest variable, the "mixing" proportions denoting population share of
#' observations corresponding to each latent multi-way table, and coefficients
#' on any class-predictor covariates, if specified in the model.
#'
#' Model specification: Latent class models have more than one manifest
#' variable, so the response variables are `cbind(dv1,dv2,dv3...)` where `dv#`
#' refer to variable names in the data frame. For models with no covariates, the
#' formula is `cbind(dv1,dv2,dv3)~1`. For models with covariates, replace the
#' `~1` with the desired function of predictors `iv1,iv2,iv3...` as, for
#' example, `cbind(dv1,dv2,dv3)~iv1+iv2*iv3`.
#'
#' `poLCA` treats all manifest variables as qualitative/categorical/nominal
#' -- NOT as ordinal.
#'
#' The implemention of this function in the package poLCAParallel is rewritten
#' in C++. Multiple threads are used, where each thread processes an initial
#' value or repetition.
#'
#' Notes:
#'
#' `poLCA` uses EM and Newton-Raphson algorithms to maximize the latent class
#' model log-likelihood function. Depending on the starting parameters, this
#' algorithm may only locate a local, rather than global, maximum. This becomes
#' more and more of a problem as `nclass` increases. It is therefore highly
#' advisable to run `poLCA` multiple times until you are relatively certain that
#' you have located the global maximum log-likelihood. As long as
#' `probs.start=NULL`, each function call will use different (random) initial
#' starting parameters. Alternatively, setting `nrep` to a value greater than
#' one enables the user to estimate the latent class model multiple times with a
#' single call to `poLCA`, thus conducting the search for the global maximizer
#' automatically.
#'
#' The term "Latent class regression" (LCR) can have two meanings. In this
#' package, LCR models refer to latent class models in which the probability of
#' class membership is predicted by one or more covariates. However, in other
#' contexts, LCR is also used to refer to regression models in which the
#' manifest variable is partitioned into some specified number of latent classes
#' as part of estimating the regression model. It is a way to simultaneously fit
#' more than one regression to the data when the latent data partition is
#' unknown. The `flexmix` function in package `flexmix` will estimate this other
#' type of LCR model. Because of these terminology issues, the LCR models this
#' package estimates are sometimes termed "latent class models with covariates"
#' or "concomitant-variable latent class analysis," both of which are accurate
#' descriptions of this model.
#'
#' The package poLCAParallel reimplements the poLCA fitting, standard error
#' calculations, goodness of fit tests and the bootstrap log-likelihood ratio
#' test in C++. This was done using
#' [Rcpp](https://cran.r-project.org/package=Rcpp) and
#' [RcppArmadillo](https://cran.r-project.org/package=RcppArmadillo) which
#' allows R to run fast C++ code. Additional notes include:
#'
#' * The API remains the same as the original poLCA with a few additions
#' * It tries to reproduce results from the original poLCA
#' * The code uses [Armadillo](https://arma.sourceforge.net/) for linear algebra
#' * Multiple repetitions are done in parallel using
#'   [`std::jthread`](https://en.cppreference.com/cpp/thread/jthread) for
#'   multi-thread programming and
#'   [`std::mutex`](https://en.cppreference.com/cpp/thread/mutex) to prevent
#'   data races
#' * Direct inversion of matrices is avoided to improve numerical stability and
#'   performance
#' * Response probabilities are reordered to increase cache efficiency
#' * Use of [`std::map`](https://en.cppreference.com/cpp/container/map) for the
#'   chi-squared calculations to improve performance
#'
#' Further reading is available on the
#' [QMUL ITS Research Blog](https://blog.hpc.qmul.ac.uk/speeding_up_r_packages/).
#'
#' References:
#' * Agresti, Alan. 2002. *Categorical Data Analysis, second edition*.
#'   Hoboken: John Wiley & Sons.
#' * Bandeen-Roche, Karen, Diana L. Miglioretti, Scott L. Zeger, and Paul J.
#'   Rathouz. 1997. "Latent Variable Regression for Multiple Discrete
#'   Outcomes." *Journal of the American Statistical Association*.
#'   92(440): 1375-1386.
#' * Hagenaars, Jacques A. and Allan L. McCutcheon, eds. 2002.
#'   *Applied Latent Class Analysis*. Cambridge: Cambridge University
#'   Press.
#' * McLachlan, Geoffrey J. and Thriyambakam Krishnan. 1997.
#'   *The EM Algorithm and Extensions*. New York: John Wiley & Sons.
#'
#' @param formula A formula expression of the form `response ~ predictors`. The
#' details of model specification are given below.
#' @param data A data frame containing variables in `formula`. Manifest
#' variables must contain *only* integer values, and must be coded with
#' consecutive values from 1 to the maximum number of outcomes for each
#' variable. All missing values should be entered as `NA`.
#' @param nclass The number of latent classes to assume in the model. Setting
#' `nclass=1` results in `poLCA` estimating the loglinear independence model.
#' The default is two.
#' @param maxiter The maximum number of iterations through which the estimation
#' algorithm will cycle.
#' @param graphs Logical, for whether `poLCA` should graphically display the
#' parameter estimates at the completion of the estimation algorithm. The
#' default is `FALSE`.
#' @param tol A tolerance value for judging when convergence has been reached.
#' When the one-iteration change in the estimated log-likelihood is less than
#' `tol`, the estimation algorithm stops updating and considers the maximum
#' log-lparameterikelihood to have been found.
#' @param na.rm Logical, for how `poLCA` handles cases with missing values on
#' the manifest variables. If `TRUE`, those cases are removed (listwise deleted)
#' before estimating the model. If `FALSE`, cases with missing values are
#' retained. Cases with missing covariates are always removed. The default is
#' `TRUE`.
#' @param probs.start A list of matrices of class-conditional response
#' probabilities to be used as the starting values for the estimation algorithm.
#' Each matrix in the list corresponds to one manifest variable, with one row
#' for each latent class, and one column for each outcome. The default is
#' `NULL`, producing random starting values. Note that if `nrep>1`, then any
#' user-specified `probs.start` values are only used in the first of the `nrep`
#' attempts.
#' @param nrep Number of times to estimate the model, using different values of
#' `probs.start`. The default is one. Setting `nrep`>1 automates the search for
#' the global---rather than just a local---maximum of the log-likelihood
#' function. `poLCA` returns the parameter estimates corresponding to the model
#' with the greatest log-likelihood.
#' @param verbose Logical, indicating whether `poLCA` should output to the
#' screen the results of the model. If `FALSE`, no output is produced. The
#' default is `TRUE`.
#' @param calc.se Logical, indicating whether `poLCA` should calculate the
#' standard errors of the estimated class-conditional response probabilities and
#' mixing proportions. The default is `TRUE`.
#' @param calc.chisq Logical, indicating whether to calculate the goodness of
#' fit statistics, the chi squared statistics and the log likelihood ratio. The
#' default is `TRUE`.
#' @param n.thread Integer, the number of threads to use. Each thread processes
#' a repetition. By default, all detectable threads are used.
#' @param se.smooth Logical, experimental, for calculating the standard errors,
#' whether to smooth the outcome probabilities to produce more numerical stable
#' results at the cost of bias.
#'
#' @return an object of class poLCA; a list containing the following elements:
#' * `y`: data frame of manifest variables.
#' * `x`: data frame of covariates, if specified.
#' * `N`: number of cases used in model.
#' * `Nobs`: number of fully observed cases (less than or equal to `N`).
#' * `probs`: estimated class-conditional response probabilities.
#' * `probs.se`: standard errors of estimated class-conditional response
#'   probabilities, in the same format as `probs`.
#' * `P`: sizes of each latent class; equal to the mixing proportions in the
#'   function basic latent class model, or the mean of the priors in the latent
#'   class regression model.
#' * `P.se`: the standard errors of the estimated `P`.
#' * `prior`: matrix of prior class membership probabilities
#' * `posterior`: matrix of posterior class membership probabilities; also see
#'   function `poLCA.posterior`.
#' * `predclass`: vector of predicted class memberships, by modal assignment.
#' * `predcell`: table of observed versus predicted cell counts for cases with
#'   no missing values; also see functions `poLCA.table` and `poLCA.predcell`
#' * `llik`: maximum value of the log-likelihood.
#' * `numiter`: number of iterations until reaching convergence.
#' * `maxiter`: maximum number of iterations through which the estimation
#'   algorithm was set to run.
#' * `coeff`: multinomial logit coefficient estimates on covariates (when
#'   estimated). `coeff` is a matrix with `nclass-1` columns, and one row for
#'   each covariate. All logit coefficients are calculated for classes with
#'   respect to class 1.
#' * `coeff.se`: standard errors of coefficient estimates on covariates (when
#'   estimated), in the same format as `coeff`.
#' * `coeff.V`: covariance matrix of coefficient estimates on covariates (when
#'   estimated).
#' * `aic`: Akaike Information Criterion.
#' * `bic`: Bayesian Information Criterion.
#' * `Gsq`: Likelihood ratio/deviance statistic.
#' * `Chisq`: Pearson Chi-square goodness of fit statistic for fitted vs.
#'   observed multiway tables.
#' * `time`: length of time it took to run the model.
#' * `npar`: number of degrees of freedom used by the model (estimated
#'   parameters).
#' * `resid.df`: number of residual degrees of freedom.
#' * `attempts`: a vector containing the maximum log-likelihood values found in
#'   each of the `nrep` attempts to fit the model.
#' * `eflag`: Logical, error flag. `TRUE` if estimation algorithm needed to
#'   automatically restart with new initial parameters. A restart is caused in
#'   the event of computational/rounding errors that result in nonsensical
#'   parameter estimates.
#' * `probs.start`: A list of matrices containing the class-conditional response
#'   probabilities used as starting values in the estimation algorithm. If the
#'   algorithm needed to restart (see `eflag`), then this contains the starting
#'   values used for the final, successful, run.
#' * `probs.start.ok`: Logical. `FALSE` if `probs.start` was incorrectly
#'   specified by the user, otherwise `TRUE`.
#' * `call`: function call to `poLCA`.
#'
#' @examples
#' ##
#' ## Three models without covariates:
#' ## M0: Loglinear independence model.
#' ## M1: Two-class latent class model.
#' ## M2: Three-class latent class model.
#' ##
#' data(values)
#' f <- cbind(A, B, C, D)~1
#' M0 <- poLCA(f, values, nclass = 1) # log-likelihood: -543.6498
#' M1 <- poLCA(f, values, nclass = 2) # log-likelihood: -504.4677
#' # log-likelihood: -503.3011
#' M2 <- poLCA(f, values, nclass = 3, maxiter = 8000)
#'
#' ##
#' ## Three-class model with a single covariate.
#' ##
#' data(election)
#' f2a <- cbind(
#'   MORALG, CARESG, KNOWG, LEADG, DISHONG, INTELG,
#'   MORALB, CARESB, KNOWB, LEADB, DISHONB, INTELB
#' )~PARTY
#' # log-likelihood: -16222.32
#' nes2a <- poLCA(f2a, election, nclass = 3, nrep = 5)
#' pidmat <- cbind(1, c(1:7))
#' exb <- exp(pidmat %*% nes2a$coeff)
#' matplot(c(1:7), (cbind(1, exb) / (1 + rowSums(exb))),
#'   ylim = c(0, 1), type = "l",
#'   main = "Party ID as a predictor of candidate affinity class",
#'   xlab = "Party ID: strong Democratic (1) to strong Republican (7)",
#'   ylab = "Probability of latent class membership", lwd = 2, col = 1
#' )
#' text(5.9, 0.35, "Other")
#' text(5.4, 0.7, "Bush affinity")
#' text(1.8, 0.6, "Gore affinity")
#'
#' @export
poLCA <- function(formula,
                  data,
                  nclass = 2,
                  maxiter = 1000,
                  graphs = FALSE,
                  tol = 1e-10,
                  na.rm = TRUE,
                  probs.start = NULL,
                  nrep = 1,
                  verbose = TRUE,
                  calc.se = TRUE,
                  calc.chisq = TRUE,
                  n.thread = parallel::detectCores(),
                  se.smooth = FALSE) {
  # nclass == 1 will use original code
  # poLCAParallel edits the original code for the nclass > 1 case
  if (nclass == 1) {
    warning("For nclass = 1, using the original poLCA code")
    return(poLCA::poLCA(
      formula, data, nclass, maxiter, graphs, tol, na.rm,
      probs.start, nrep, verbose, calc.se
    ))
  }

  starttime <- Sys.time()

  data_x_y <- extract_data(formula, data, na.rm)

  features <- data_x_y$x # features
  responses <- data_x_y$y # responses
  ndata <- nrow(responses) # number of data points
  ncategory <- ncol(responses) # number of categories
  # number of outcomes for each category
  noutcomes <- t(matrix(apply(responses, 2, max)))
  nfeature <- ncol(features) # number of features

  # check probs.start and generate any additional probs if needed
  # keep track if the user has provided probabilities or not
  probs.start <- check_and_generate_initial_probs(
    probs.start, nrep, noutcomes, nclass
  )

  # random seed required to generate new initial values when needed
  seed <- sample.int(
    as.integer(.Machine$integer.max), 5,
    replace = TRUE
  )
  # run C++ code here
  em_results <- EmAlgorithmRcpp(
    features,
    t(responses),
    probs.start$vector,
    ndata,
    nfeature,
    noutcomes,
    nclass,
    nrep,
    na.rm,
    n.thread,
    maxiter,
    tol,
    seed
  )

  # ========== EXTRACT RESULTS ========== #
  # Put all outputs in a list ret, to be returned, making up the poLCA object
  ret <- list()

  ret$posterior <- em_results[[1]]
  ret$prior <- em_results[[2]]

  ret$probs <- unvectorize_probs(em_results[[3]], noutcomes, nclass)
  names(ret$probs) <- colnames(responses)

  # labelling coeff
  if (nfeature > 1) {
    ret$coeff <- em_results[[4]]
    ret$coeff <- matrix(ret$coeff, nrow = nfeature)
    rownames(ret$coeff) <- colnames(features)
  } else {
    ret$coeff <- NA
  }

  ret$attempts <- em_results[[5]]
  # maximum value of the log-likelihood
  # em_results[[6]] is best_rep_index
  ret$llik <- ret$attempts[em_results[[6]]]
  ret$numiter <- em_results[[7]]
  # best starting values of class-conditional response probabilities
  ret$probs.start <- unvectorize_probs(em_results[[8]], noutcomes, nclass)
  ret$eflag <- em_results[[9]]

  # Nx1 vector of predicted class memberships, by modal assignment
  ret$predclass <- apply(ret$posterior, 1, which.max)
  # estimated class population shares
  ret$P <- colMeans(ret$posterior)
  # if starting probs specified, logical indicating proper entry format
  ret$probs.start.ok <- probs.start$ok

  # placeholder for standard error
  ret$P.se <- NA
  ret$probs.se <- NA
  ret$coeff.se <- NA
  ret$coeff.V <- NA

  # placeholder for goodness of fit
  ret$Chisq <- NA
  ret$Gsq <- NA
  ret$predcell <- NA

  # number of degrees of freedom used by the model (number of estimated
  # parameters)
  ret$npar <- (nclass * sum(noutcomes - 1)) + (nclass - 1)
  if (nfeature > 1) {
    ret$npar <- ret$npar + (nfeature * (nclass - 1)) - (nclass - 1)
  }
  # Akaike Information Criterion
  ret$aic <- (-2 * ret$llik) + (2 * ret$npar)
  # Schwarz-Bayesian Information Criterion
  ret$bic <- (-2 * ret$llik) + (log(ndata) * ret$npar)
  # number of fully observed cases (if na.rm=F)
  ret$Nobs <- sum(rowSums(responses == 0) == 0)

  responses[responses == 0] <- NA
  ret$y <- data.frame(responses) # outcome variables
  ret$x <- data.frame(features) # covariates, if specified
  for (j in 1:ncategory) {
    rownames(ret$probs[[j]]) <- paste("class ", 1:nclass, ": ", sep = "")
    if (is.factor(data[, match(colnames(responses), colnames(data))[j]])) {
      lev <- levels(data[, match(colnames(responses), colnames(data))[j]])
      colnames(ret$probs[[j]]) <- lev
      ret$y[, j] <- factor(ret$y[, j], labels = lev)
    } else {
      colnames(ret$probs[[j]]) <-
        paste("Pr(", 1:ncol(ret$probs[[j]]), ")", sep = "")
    }
  }
  ret$N <- ndata # number of observations

  # calculate the standard errors
  if (calc.se) {
    ret <- poLCAParallel.se(ret, se.smooth)
  }

  # if rows are fully observed and chi squared requested
  # do goodness of fit test
  if (!(all(rowSums(responses == 0) > 0)) && calc.chisq) {
    ret <- poLCAParallel.goodnessfit(ret)
  }

  ret$maxiter <- maxiter # maximum number of iterations specified by user
  # number of residual degrees of freedom
  ret$resid.df <- min(ret$N, (prod(noutcomes) - 1)) - ret$npar
  class(ret) <- "poLCA"
  if (graphs) {
    plot.poLCA(ret)
  }
  if (verbose) {
    print.poLCA(ret)
  }
  ret$time <- Sys.time() - starttime # how long it took to run the model
  ret$call <- match.call()
  return(ret)
}

#' Extract the responses and features given the data and formula
#'
#' Extract the responses and features given the data and formula from `poLCA()`.
#'
#' @param formula A formula expression of the form `response ~ predictors` -
#' see `poLCA()` for further details
#' @param data A data frame containing variables in `formula` - see `poLCA()`
#' for further details
#' @param na.rm boolean, to handle missing values or not - see `poLCA()` for
#' further details
#'
#' @return List with attributes `x` and `y`. `x` contain the features as a
#' matrix with size `ndata` x `nfeature`. `y` contain the responses as a matrix
#' with size `ndata` x `ncategory`
#'
#' @noRd
extract_data <- function(formula, data, na.rm) {
  mframe <- stats::model.frame(formula, data, na.action = NULL)
  mf <- stats::model.response(mframe)
  if (any(mf < 1, na.rm = TRUE) || any(round(mf) != mf, na.rm = TRUE)) {
    stop("Some manifest variables contain values that are not positive integers.
          For poLCA to run, please recode categorical outcome variables to
          increment from 1 to the maximum number of outcome categories for each
          variable.")
  }
  data <- data[rowSums(is.na(stats::model.matrix(formula, mframe))) == 0, ]
  if (na.rm) {
    mframe <- stats::model.frame(formula, data)
    y <- stats::model.response(mframe)
  } else {
    mframe <- stats::model.frame(formula, data, na.action = NULL)
    y <- stats::model.response(mframe)
    y[is.na(y)] <- 0
  }
  if (any(sapply(lapply(as.data.frame(y), table), length) == 1)) {
    y <- y[, !(sapply(apply(y, 2, table), length) == 1)]
    warning("At least one manifest variable contained only one outcome category,
             and has been removed from the analysis.")
  }
  x <- stats::model.matrix(formula, mframe)
  return(list(x = x, y = y))
}

#' Check and generate initial probabilities
#'
#' Checks the user provided `probs_start` and generate further initial
#' probabilities for the EM algorithm. If `probs_start` is invalid, it will
#' generate new initial probabilities
#'
#' @param probs_start A list of matrices of class-conditional response
#' probabilities, see `poLCA()` for further details
#' @param nrep int, number of repetitions (or initial values) requested by the
#' user
#' @param noutcomes vector of int, number of outcomes for each manifest/category
#' @param nclass int, number of classes or clusters
#'
#' @return list with attributes `vector` and `ok`
#' * `vector`: vector of outcome probabilities conditioned on the
#'   manifest/category, the class/cluster and the repetition. For the first
#'   repetition, it contains a copy of `probs_start` (or replaced with new
#'   random probabilities if invalid or not provided). For the remaining
#'   repetitions, they are randomly generated. Imagine a nested loop, from
#'   inner to outer, or a flatten column-major matrix, the probabilities are
#'   arranged in the following order:
#'   * dim 1: for each outcome
#'   * dim 2: for each manifest/category
#'   * dim 3: for each class/cluster
#'   * dim 4: for each repetition
#' * `ok`: boolean, `TRUE` if `probs_start` is valid or default `NULL`
#'
#' @noRd
check_and_generate_initial_probs <- function(probs_start,
                                             nrep, noutcomes, nclass) {
  # keep track if the user has not provide probs
  # if the user has not provide probs, ie when `probs_start` is `NULL`
  # then set the return value of `ok` to `TRUE`
  has_provide_probs <- is.null(probs_start)

  is_probs_valid <- check_probs(probs_start, noutcomes, nclass)

  # perpare initial values
  probs_vector <- c()
  irep <- 1

  # if can use user's provided probs_start
  if (is_probs_valid) {
    probs_list_i <- poLCAParallel.vectorize(probs_start)
    probs_vector <- c(
      probs_vector,
      probs_list_i$vecprobs
    )
    irep <- irep + 1
  }
  # if cannot use the user's provided probs_start, generate a new one

  # generate random probabilities
  if (nrep > 1 || irep == 1) {
    for (repl in irep:nrep) {
      probs_vector <- c(
        probs_vector,
        random_vectorized_probs(noutcomes, nclass)
      )
    }
  }

  # if the user has not provided probs, set `is_probs_valid` to `TRUE`, this is
  # used further down and set the attribute `probs.start.ok` later on, see
  # `print.poLCA.R` where this is used
  #
  # Setting `is_probs_valid = FALSE` assumes the user has provided probs in the
  # warning message in print.poLCA.R()
  #
  # Setting `is_probs_valid = TRUE` when the user has not provided probs is done
  # to not alert the user when they have not provided probs. The alert is only
  # needed if the user provided invalid probabilities
  if (has_provide_probs) {
    is_probs_valid <- TRUE
  }

  return(list(vector = probs_vector, ok = is_probs_valid))
}

#' Generate random initial probabilities
#'
#' @param noutcomes vector of int, number of outcomes for each manifest/category
#' @param nclass int, number of classes or clusters
#'
#' @return vector of outcome probabilities conditioned on the manifest/category
#' and the class/cluster. Imagine a nested loop, from inner to outer, or a
#' flatten column-major matrix, the probabilities are arranged in the following
#' order:
#' * dim 1: for each outcome
#' * dim 2: for each manifest/category
#' * dim 3: for each class/cluster
#'
#' @noRd
random_vectorized_probs <- function(noutcomes, nclass) {
  probs <- list()
  for (j in seq_len((length(noutcomes)))) {
    probs[[j]] <- matrix(
      stats::runif(nclass * noutcomes[j]),
      nrow = nclass, ncol = noutcomes[j]
    )
    probs[[j]] <- probs[[j]] / rowSums(probs[[j]])
  }
  probs_list_i <- poLCAParallel.vectorize(probs)
  return(probs_list_i$vecprobs)
}

#' Check if the user's provided `probs_start` is valid
#'
#' @param probs_start A list of matrices of class-conditional response
#' probabilities, see the parameter `probs.start` in `poLCA()` for further
#' details
#' @param noutcomes vector of int, number of outcomes for each manifest/category
#' @param nclass int, number of classes or clusters
#'
#' @return boolean, `TRUE` if `probs_start` is valid
#'
#' @noRd
check_probs <- function(probs_start, noutcomes, nclass) {
  ncategory <- length(noutcomes)
  if (is.null(probs_start)) {
    return(FALSE)
  }
  if (length(probs_start) != ncategory) {
    return(FALSE)
  }
  if (!is.list(probs_start)) {
    return(FALSE)
  }
  if (sum(sapply(probs_start, dim)[1, ] == nclass) != ncategory) {
    return(FALSE)
  }
  if (sum(sapply(probs_start, dim)[2, ] == noutcomes) != ncategory) {
    return(FALSE)
  }
  if (sum(round(sapply(probs_start, rowSums), 4) == 1) !=
    (nclass * ncategory)) {
    return(FALSE)
  }
  return(TRUE)
}

#' Unvectorize probabilities
#'
#' Wrapper function around `poLCAParallel.unvectorize()` so you can provide
#' `prob_vec`, `noutcomes` and `nclass` separately, rather than all in one list
#'
#' @param probs_vec vector of outcome probabilities conditioned on the
#' manifest/category and the class/cluster. Imagine a nested loop, from inner
#' to outer, or a flatten column-major matrix, the probabilities are arranged
#' in the following order:
#' * dim 1: for each outcome
#' * dim 2: for each manifest/category
#' * dim 3: for each class/cluster
#' @param noutcomes vector of int, number of outcomes for each manifest/category
#' @param nclass int, number of classes/clusters
#'
#' @return list for each category/manifest variable. For the `i`th entry, it
#' contains a matrix of outcome probabilities with dimensions `n_class` x
#' `n_outcomes[i]`
#'
#' @noRd
unvectorize_probs <- function(probs_vec, noutcomes, nclass) {
  return(poLCAParallel.unvectorize(
    list(vecprobs = probs_vec, numChoices = noutcomes, classes = nclass)
  ))
}
