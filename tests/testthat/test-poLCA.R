#' Test the contents of a `poLCA` object
#'
#' Test the Rcpp outputted contents of a `poLCA` object, this tests the prior
#' probabilities, posterior probabilities, outcome probabilities, log
#' likelihood, number of iterations done, initial probabilities which achieved
#' the maximum log likelihood and more...
#'
#' Provide the `poLCA` object and parameters which are used to test the object
#'
#' @param lc A model object estimated using the `poLCA` function (or a list
#'   which mocks it)
#' @param n_data Number of data points
#' @param n_outcomes Vector of integers, number of outcomes for each category
#' @param n_cluster Number of clusters fitted
#' @param n_rep Number of repetitions used
#' @param na_rm Logical, if to remove NA responses
#' @param maxiter Number of iterations used in the EM algorithm
test_polca_em_algorithm <- function(lc, n_data, n_outcomes, n_cluster, n_rep,
                                    na_rm, maxiter) {
  # if remove NA responses, use Nobs, number of fully observed data
  if (na_rm) {
    n_data <- lc$Nobs
  }
  # test the probabilities
  test_cluster_probs(lc$prior, n_data, n_cluster)
  test_cluster_probs(lc$posterior, n_data, n_cluster)
  test_outcome_probs(lc$probs, n_outcomes, n_cluster)
  test_outcome_probs(lc$probs.start, n_outcomes, n_cluster)

  # test the log likelihoods
  expect_identical(length(lc$attempts), as.integer(n_rep))
  expect_lte(lc$llik, 0)
  for (ln_l_i in lc$attempts) {
    expect_lte(ln_l_i, 0)
  }
  expect_equal(max(lc$attempts), lc$llik)

  # test the number of iterations
  expect_gte(lc$numiter, 0)
  expect_lte(lc$numiter, maxiter)
  expect_identical(lc$maxiter, maxiter)

  expect_equal(is.logical(lc$eflag), TRUE)
}

#' Test the other contents of a `poLCA` object
#'
#' Test the outputted contents of a `poLCA` object not tested in
#' `test_polca_em_algorithm()`. It tests the R outputs (not Rcpp) created on
#' `poLCA()` such as the features, responses, number of data points and time
#' taken... etc
#'
#' Provide the `poLCA` object and parameters which are used to test the object
#'
#' @param lc A model object estimated using the `poLCA` function (or a list
#'   which mocks it)
#' @param n_data Number of data points
#' @param n_feature Number of features
#' @param n_outcomes Vector of integers, number of outcomes for each category
#' @param n_cluster Number of clusters fitted
#' @param na_rm Logical, if to remove NA responses
test_polca_other <- function(lc, n_data, n_feature, n_outcomes, n_cluster,
                             na_rm) {
  # if remove NA responses, use Nobs, number of fully observed data
  if (na_rm) {
    n_data <- lc$Nobs
  }
  expect_identical(lc$N, as.integer(n_data))

  # test design matrix of features
  expect_identical(nrow(lc$x), as.integer(n_data))
  expect_identical(ncol(lc$x), as.integer(n_feature + 1))
  expect_identical(all(lc$x[, 1] == 1), TRUE)

  # test design matrix of responses
  expect_identical(nrow(lc$y), as.integer(n_data))
  expect_identical(ncol(lc$y), as.integer(length(n_outcomes)))
  if (na_rm) {
    expect_identical(all(lc$y >= 1), TRUE)
    expect_identical(lc$Nobs, lc$N)
  } else {
    response_mat <- as.matrix(lc$y)
    expect_identical(all(response_mat[!is.na(response_mat)] >= 1), TRUE)
    expect_identical(
      lc$Nobs,
      as.integer(sum(rowSums(is.na(response_mat)) == 0))
    )
  }

  # test the types of attributes
  is.logical(lc$probs.start.ok)
  inherits(lc$time, "difftime")
}

#' Test using `poLCA()` for the non-regression problem
#'
#' Test using `poLCA()` for the non-regression problem. Random responses are
#' generated and then passed to `poLCA()` to be fitted onto. The results are
#' then tested
#'
#' @param n_data Number of data points
#' @param n_outcomes Vector of integers, number of outcomes for each category
#' @param n_cluster Number of clusters fitted
#' @param n_rep Number of different initial values to try
#' @param na_rm Logical, if to remove NA responses
#' @param n_thread Number of threads to use
#' @param maxiter Number of iterations used in the EM algorithm
#' @param tol Tolerance used in the EM algorithm
#' @param prob_na Probability of missing data
test_non_regression <- function(n_data, n_outcomes, n_cluster, n_rep, na_rm,
                                n_thread, maxiter, tol, prob_na) {
  responses <- random_response(n_data, n_outcomes, prob_na, NaN)
  formula <- get_non_regression_formula(responses)
  lc <- poLCAParallel::poLCA(formula, responses, n_cluster,
    maxiter = maxiter, tol = tol, na.rm = na_rm, nrep = n_rep,
    verbose = FALSE, n.thread = n_thread
  )
  test_polca_em_algorithm(
    lc, n_data, n_outcomes, n_cluster, n_rep, na_rm, maxiter
  )
  test_polca_other(lc, n_data, 0, n_outcomes, n_cluster, na_rm)
  test_polca_goodnessfit(lc, n_outcomes)
  test_standard_error(lc, n_outcomes, n_cluster)
}

#' Test using `poLCA()` for the regression problem
#'
#' Test using `poLCA()` for the regression problem. Random responses and
#' features are generated and then passed to `poLCA()` to be fitted onto. The
#' results are then tested
#'
#' @param n_data Number of data points
#' @param n_feature Number of features
#' @param n_outcomes Vector of integers, number of outcomes for each category
#' @param n_cluster Number of clusters fitted
#' @param n_rep Number of different initial values to try
#' @param na_rm Logical, if to remove NA responses
#' @param n_thread Number of threads to use
#' @param maxiter Number of iterations used in the EM algorithm
#' @param tol Tolerance used in the EM algorithm
#' @param prob_na Probability of missing data
test_regression <- function(n_data, n_feature, n_outcomes, n_cluster, n_rep,
                            na_rm, n_thread, maxiter, tol, prob_na) {
  features <- random_features(n_data, n_feature)
  responses <- random_response(n_data, n_outcomes, prob_na, NaN)
  formula <- get_regression_formula(responses, features)
  data <- cbind(responses, features)

  lc <- poLCAParallel::poLCA(formula, data, n_cluster,
    maxiter = maxiter, tol = tol, na.rm = na_rm, nrep = n_rep,
    verbose = FALSE, n.thread = n_thread
  )

  test_polca_em_algorithm(
    lc, n_data, n_outcomes, n_cluster, n_rep, na_rm, maxiter
  )
  test_polca_other(lc, n_data, n_feature, n_outcomes, n_cluster, na_rm)
  test_polca_goodnessfit(lc, n_outcomes)
  test_standard_error(lc, n_outcomes, n_cluster)

  # test coefficients
  # one extra feature as poLCA adds an intercept term
  # one less cluster as only need (n_cluster - 1) probabilities to work out the
  # remaining one
  expect_identical(nrow(lc$coeff), as.integer(n_feature + 1))
  expect_identical(ncol(lc$coeff), as.integer(n_cluster - 1))
}

#' Test if the two fitted model are the same
#'
#' Test if the two fitted model are the same, with tolerance, one using
#' poLCAParallel and the other using either `poLCA` or `poLCAParallel`. It tests
#' the attributes of the fitted models
#'
#' Use the parameter `is_strict` to specify if the fitted model `lc_compare` is
#' from either `poLCA` or `poLCAParallel`. The comparisions are softer for
#' `poLCA` because some attributes are not available and implmentations may
#' differ (which then require a larger tolerance)
#'
#' `poLCA` and `poLCAParallel` differ quite a lot in terms of the regression
#' problem, hence not every variable is tested for equality in the regression
#' problem
#'
#' @param lc_parallel A model object estimated using the `poLCAParallel::poLCA`
#'   function (or a list which mocks it)
#' @param lc_compare A model object estimated using either `poLCA::poLCA`
#'   or `poLCAParallel::poLCA` functions (or a list which mocks it)
#' @param is_regression boolean, if the problem is a regression problem
#' @param is_strict boolean set to `TRUE` if the two provided models are from
#'   `poLCAParallel::poLCA`, else set to `FALSE` if `lc_compare` is from
#'   `poLCA::poLCA`
test_equal <- function(lc_parallel, lc_compare, is_regression, is_strict) {
  if (is_strict) {
    equal_tol_gof <- NULL
    equal_tol_prob <- NULL
  } else {
    # choosen as the following fails
    # * equal_tol_gof <- 1e0 * sqrt(.Machine$double.eps)
    # * equal_tol_prob <- 1e2 * sqrt(.Machine$double.eps)
    equal_tol_gof <- 1e2 * sqrt(.Machine$double.eps)
    equal_tol_prob <- 1e4 * sqrt(.Machine$double.eps)
  }

  # test if all attributes in the og code is in our code
  for (attribute_i in names(lc_compare)) {
    expect_identical(attribute_i %in% names(lc_parallel), TRUE)
  }

  if (!is_regression) {
    # test if results are the same
    expect_equal(lc_parallel$llik, lc_compare$llik)
    expect_equal(lc_parallel$aic, lc_compare$aic)
    expect_equal(lc_parallel$bic, lc_compare$bic)
    expect_equal(lc_parallel$Nobs, lc_compare$Nobs)

    expect_equal(lc_parallel$Chisq, lc_compare$Chisq,
      tolerance = equal_tol_gof
    )
    expect_equal(lc_parallel$Gsq, lc_compare$Gsq,
      tolerance = equal_tol_gof
    )

    expect_equal(lc_parallel$probs, lc_compare$probs,
      tolerance = equal_tol_prob
    )
    expect_equal(lc_parallel$P, lc_compare$P,
      tolerance = equal_tol_prob
    )
    expect_equal(lc_parallel$posterior, lc_compare$posterior,
      tolerance = equal_tol_prob
    )

    # test $predcell
    # all columns except for the last are integers, describing the observed
    # outcomes and the count
    # the last column is the expected count which is a double
    #
    # split the test comparisions into two parts, use identical for integers and
    # equal for doubles

    # test integers, test the column names, extract and test all columns except
    # for the last
    column_names <- names(lc_parallel$predcell)
    expect_identical(column_names, names(lc_compare$predcell))
    identical_columns <- column_names[seq_len(length(column_names) - 1)]
    expect_identical(
      lc_parallel[identical_columns],
      lc_compare[identical_columns]
    )

    # the original poLCA code rounds the expected counts to 3 decimal places
    if (!is_strict) {
      parallel_expected <- round(lc_parallel$predcell$expected, 3)
      compare_expected <- lc_compare$predcell$expected
      # results may differ on the 3rd decimal place
      diff <- (abs(parallel_expected - compare_expected) <=
        0.001 + sqrt(.Machine$double.eps))
      expect_identical(all(diff), TRUE)
    } else {
      expect_equal(lc_parallel$predcell$expected, lc_compare$predcell$expected)
    }
  }

  expect_equal(lc_parallel$probs.start.ok, lc_compare$probs.start.ok)
  expect_equal(lc_parallel$npar, lc_compare$npar)
  expect_identical(lc_parallel$y, lc_compare$y)
  expect_identical(lc_parallel$x, lc_compare$x)

  if (is_strict) {
    expect_equal(lc_parallel$prior, lc_compare$prior)
  }
}

#' Test if results can be reproduced
#'
#' Test if results are the same, or at least similar, as the original poLCA code
#' for the non-regression problem. Generate data and pass it to `poLCA::poLCA()`
#' and `poLCAParallel::poLCA()` and compare results
#'
#' When calling `poLCA::poLCA()` is uses `nrep` to `1` so that each call of
#' `poLCA()` uses the same initial probabilities
#'
#' Also tests if results can be reproduced when using `n.thread = 1`
#'
#' @param n_data Number of data points
#' @param n_outcomes Vector of integers, number of outcomes for each category
#' @param n_cluster Number of clusters fitted
#' @param n_rep Number of different initial values to try
#' @param na_rm Logical, if to remove NA responses
#' @param n_thread Number of threads to use
#' @param maxiter Number of iterations used in the EM algorithm
#' @param tol Tolerance used in the EM algorithm
#' @param prob_na Probability of missing data
#' @param seed Seed to generate random data and seed poLCA
test_reproduce_non_regression <- function(n_data, n_outcomes, n_cluster, n_rep,
                                          na_rm, n_thread, maxiter, tol,
                                          prob_na, seed) {
  set.seed(seed)
  responses <- random_response(n_data, n_outcomes, prob_na, NaN)
  formula <- get_non_regression_formula(responses)

  # set the seed before each poLCA() call so that they generate the same initial
  # random probabilities within the function call
  # THIS ASSUMES THE IMPLEMENTATION OF GENERATING INITIAL RANDOM PROBABILITES
  # IS THE SAME AS THE ORIGINAL CODE

  set.seed(seed)
  lc_polca <- poLCA::poLCA(formula, responses, n_cluster,
    maxiter = maxiter, tol = tol, na.rm = na_rm, nrep = 1,
    verbose = FALSE
  )

  set.seed(seed)
  lc_parallel <- poLCAParallel::poLCA(formula, responses, n_cluster,
    maxiter = maxiter, tol = tol, na.rm = na_rm, nrep = 1,
    verbose = FALSE, n.thread = n_thread
  )
  # test if results are the same
  test_equal(lc_parallel, lc_polca, FALSE, FALSE)

  # compare with poLCAParallel single thread
  # nrep is greater than 1

  set.seed(seed)
  lc_parallel <- poLCAParallel::poLCA(formula, responses, n_cluster,
    maxiter = maxiter, tol = tol, na.rm = na_rm, nrep = n_rep,
    verbose = FALSE, n.thread = n_thread
  )
  set.seed(seed)
  lc_single <- poLCAParallel::poLCA(formula, responses, n_cluster,
    maxiter = maxiter, tol = tol, na.rm = na_rm, nrep = n_rep,
    verbose = FALSE, n.thread = 1
  )
  # test if results are the same
  test_equal(lc_parallel, lc_single, FALSE, TRUE)
}

#' Test if results can be reproduced
#'
#' Test if results are the same, or at least similar, as the original poLCA code
#' for the regression problem. Generate data and pass it to `poLCA::poLCA()`
#' and `poLCAParallel::poLCA()` and compare results
#'
#' The EM algorithm does depend on the initial values and how many different
#' initials were tried. A failed test could be fixed by either using a high
#' repetition count or ensure the initial values are the same
#'
#' Also tests if results can be reproduced when using `n.thread = 1`
#'
#' @param n_data Number of data points
#' @param n_feature Number of features
#' @param n_outcomes Vector of integers, number of outcomes for each category
#' @param n_cluster Number of clusters fitted
#' @param n_rep Number of different initial values to try
#' @param na_rm Logical, if to remove NA responses
#' @param n_thread Number of threads to use
#' @param maxiter Number of iterations used in the EM algorithm
#' @param tol Tolerance used in the EM algorithm
#' @param prob_na Probability of missing data
#' @param seed Seed to generate random data and seed poLCA
test_reproduce_regression <- function(n_data, n_feature, n_outcomes, n_cluster,
                                      n_rep, na_rm, n_thread, maxiter, tol,
                                      prob_na, seed) {
  set.seed(seed)
  features <- random_features(n_data, n_feature)
  responses <- random_response(n_data, n_outcomes, prob_na, NaN)
  formula <- get_regression_formula(responses, features)
  data <- cbind(responses, features)

  # set the seed before each poLCA() call so that they generate the same initial
  # random probabilities within the function call

  set.seed(seed)
  lc_polca <- poLCA::poLCA(formula, data, n_cluster,
    maxiter = maxiter, tol = tol, na.rm = na_rm, nrep = n_rep,
    verbose = FALSE
  )

  set.seed(seed)
  lc_parallel <- poLCAParallel::poLCA(formula, data, n_cluster,
    maxiter = maxiter, tol = tol, na.rm = na_rm, nrep = n_rep,
    verbose = FALSE, n.thread = n_thread
  )
  # test if results are the same
  test_equal(lc_parallel, lc_polca, TRUE, FALSE)

  # compare with poLCAParallel single thread
  set.seed(seed)
  lc_single <- poLCAParallel::poLCA(formula, data, n_cluster,
    maxiter = maxiter, tol = tol, na.rm = na_rm, nrep = n_rep,
    verbose = FALSE, n.thread = 1
  )
  # test if results are the same
  test_equal(lc_parallel, lc_single, TRUE, TRUE)
}

#' Test if results are the same as original poLCA code (provide initial probs)
#'
#' Test if results are the same, or at least similar, as the original poLCA code
#' for the non-regression problem. Generate data and initial probabilities
#' before hand. These are passed to `poLCA::poLCA()` and
#' `poLCAParallel::poLCA()` and the results are compared
#'
#' This test sets `nrep` to `1` so that each call of `poLCA()` uses the same
#' initial probabilities
#'
#' @param n_data Number of data points
#' @param n_outcomes Vector of integers, number of outcomes for each category
#' @param n_cluster Number of clusters fitted
#' @param na_rm Logical, if to remove NA responses
#' @param n_thread Number of threads to use
#' @param maxiter Number of iterations used in the EM algorithm
#' @param tol Tolerance used in the EM algorithm
#' @param prob_na Probability of missing data
test_probs_start_non_regression <- function(n_data, n_outcomes, n_cluster,
                                            na_rm, n_thread, maxiter,
                                            tol, prob_na) {
  responses <- random_response(n_data, n_outcomes, prob_na, NaN)
  formula <- get_non_regression_formula(responses)

  probs_start <- random_unvectorized_probs(n_outcomes, n_cluster)

  # do not set seed here
  # this tests if the same results can be reproduced by using the provided
  # initial probabilities

  lc_polca <- poLCA::poLCA(formula, responses, n_cluster,
    maxiter = maxiter, tol = tol, na.rm = na_rm, probs.start = probs_start,
    nrep = 1, verbose = FALSE
  )

  lc_parallel <- poLCAParallel::poLCA(formula, responses, n_cluster,
    maxiter = maxiter, tol = tol, na.rm = na_rm, probs.start = probs_start,
    nrep = 1, verbose = FALSE, n.thread = n_thread
  )

  # test if results are the same
  test_equal(lc_parallel, lc_polca, FALSE, FALSE)
}

test_that("non-regression-full-data", {
  # test using na_rm = TRUE and FALSE
  # with no missing data, they both should work in the same way
  set.seed(-1012646258)
  seeds <- sample.int(.Machine$integer.max, N_REPEAT)
  for (i in seq_len(N_REPEAT)) {
    set.seed(seeds[i])
    expect_no_error(test_non_regression(
      100,
      c(2, 3, 5, 2, 2),
      3,
      4,
      TRUE,
      N_THREAD,
      DEFAULT_MAXITER,
      DEFAULT_TOL,
      0
    ))
  }

  set.seed(-2057561765)
  seeds <- sample.int(.Machine$integer.max, N_REPEAT)
  for (i in seq_len(N_REPEAT)) {
    set.seed(seeds[i])
    expect_no_error(test_non_regression(
      100,
      c(2, 3, 5, 2, 2),
      3,
      4,
      FALSE,
      N_THREAD,
      DEFAULT_MAXITER,
      DEFAULT_TOL,
      0
    ))
  }
})

test_that("non-regression-missing-data", {
  # test using na_rm = TRUE and FALSE
  # with missing data, both will produce different results
  set.seed(-1554950958)
  seeds <- sample.int(.Machine$integer.max, N_REPEAT)
  for (i in seq_len(N_REPEAT)) {
    set.seed(seeds[i])
    expect_no_error(test_non_regression(
      100,
      c(2, 3, 5, 2, 2),
      3,
      4,
      TRUE,
      N_THREAD,
      DEFAULT_MAXITER,
      DEFAULT_TOL,
      0.1
    ))
  }

  set.seed(984792451)
  seeds <- sample.int(.Machine$integer.max, N_REPEAT)
  for (i in seq_len(N_REPEAT)) {
    set.seed(seeds[i])
    expect_no_error(test_non_regression(
      100,
      c(2, 3, 5, 2, 2),
      3,
      4,
      FALSE,
      N_THREAD,
      DEFAULT_MAXITER,
      DEFAULT_TOL,
      0.1
    ))
  }
})

test_that("regression-full-data", {
  set.seed(-590845051)
  seeds <- sample.int(.Machine$integer.max, N_REPEAT)
  for (i in seq_len(N_REPEAT)) {
    set.seed(seeds[i])
    expect_no_error(test_regression(
      100,
      4,
      c(2, 3, 5, 2, 2),
      3,
      4,
      TRUE,
      N_THREAD,
      DEFAULT_MAXITER,
      DEFAULT_TOL,
      0
    ))
  }

  set.seed(1785517768)
  seeds <- sample.int(.Machine$integer.max, N_REPEAT)
  for (i in seq_len(N_REPEAT)) {
    set.seed(seeds[i])
    expect_no_error(test_regression(
      100,
      4,
      c(2, 3, 5, 2, 2),
      3,
      4,
      FALSE,
      N_THREAD,
      DEFAULT_MAXITER,
      DEFAULT_TOL,
      0
    ))
  }
})


test_that("regression-missing-data", {
  set.seed(-85141069)
  seeds <- sample.int(.Machine$integer.max, N_REPEAT)
  for (i in seq_len(N_REPEAT)) {
    set.seed(seeds[i])
    expect_no_error(test_regression(
      100,
      4,
      c(2, 3, 5, 2, 2),
      3,
      4,
      TRUE,
      N_THREAD,
      DEFAULT_MAXITER,
      DEFAULT_TOL,
      0.1
    ))
  }

  set.seed(-2070313423)
  seeds <- sample.int(.Machine$integer.max, N_REPEAT)
  for (i in seq_len(N_REPEAT)) {
    set.seed(seeds[i])
    expect_no_error(test_regression(
      100,
      4,
      c(2, 3, 5, 2, 2),
      3,
      4,
      FALSE,
      N_THREAD,
      DEFAULT_MAXITER,
      DEFAULT_TOL,
      0.1
    ))
  }
})


test_that("reproduce-non-regression-full-data", {
  set.seed(-683307112)
  seeds <- sample.int(.Machine$integer.max, N_REPEAT)
  for (i in seq_len(N_REPEAT)) {
    expect_no_error(test_reproduce_non_regression(
      100,
      c(2, 3, 5, 2, 2),
      3,
      4,
      TRUE,
      N_THREAD,
      DEFAULT_MAXITER,
      DEFAULT_TOL,
      0,
      seeds[i]
    ))
  }

  set.seed(-1855018758)
  seeds <- sample.int(.Machine$integer.max, N_REPEAT)
  for (i in seq_len(N_REPEAT)) {
    expect_no_error(test_reproduce_non_regression(
      100,
      c(2, 3, 5, 2, 2),
      3,
      4,
      FALSE,
      N_THREAD,
      DEFAULT_MAXITER,
      DEFAULT_TOL,
      0,
      seeds[i]
    ))
  }

  set.seed(980213281)
  seeds <- sample.int(.Machine$integer.max, N_REPEAT)
  for (i in seq_len(N_REPEAT)) {
    set.seed(seeds[i])
    expect_no_error(test_probs_start_non_regression(
      100,
      c(2, 3, 5, 2, 2),
      3,
      TRUE,
      N_THREAD,
      DEFAULT_MAXITER,
      DEFAULT_TOL,
      0
    ))
  }

  set.seed(1619464396)
  seeds <- sample.int(.Machine$integer.max, N_REPEAT)
  for (i in seq_len(N_REPEAT)) {
    set.seed(seeds[i])
    expect_no_error(test_probs_start_non_regression(
      100,
      c(2, 3, 5, 2, 2),
      3,
      FALSE,
      N_THREAD,
      DEFAULT_MAXITER,
      DEFAULT_TOL,
      0
    ))
  }
})

test_that("reproduce-non-regression-missing-data", {
  set.seed(-1391069936)
  seeds <- sample.int(.Machine$integer.max, N_REPEAT)
  for (i in seq_len(N_REPEAT)) {
    expect_no_error(test_reproduce_non_regression(
      100,
      c(2, 3, 5, 2, 2),
      3,
      4,
      TRUE,
      N_THREAD,
      DEFAULT_MAXITER,
      DEFAULT_TOL,
      0.1,
      seeds[i]
    ))
  }

  set.seed(799350486)
  seeds <- sample.int(.Machine$integer.max, N_REPEAT)
  for (i in seq_len(N_REPEAT)) {
    expect_no_error(test_reproduce_non_regression(
      100,
      c(2, 3, 5, 2, 2),
      3,
      4,
      FALSE,
      N_THREAD,
      DEFAULT_MAXITER,
      DEFAULT_TOL,
      0.1,
      seeds[i]
    ))
  }

  set.seed(-1158272799)
  seeds <- sample.int(.Machine$integer.max, N_REPEAT)
  for (i in seq_len(N_REPEAT)) {
    set.seed(seeds[i])
    expect_no_error(test_probs_start_non_regression(
      100,
      c(2, 3, 5, 2, 2),
      3,
      TRUE,
      N_THREAD,
      DEFAULT_MAXITER,
      DEFAULT_TOL,
      0.1
    ))
  }

  set.seed(16136553)
  seeds <- sample.int(.Machine$integer.max, N_REPEAT)
  for (i in seq_len(N_REPEAT)) {
    set.seed(seeds[i])
    expect_no_error(test_probs_start_non_regression(
      100,
      c(2, 3, 5, 2, 2),
      3,
      FALSE,
      N_THREAD,
      DEFAULT_MAXITER,
      DEFAULT_TOL,
      0.1
    ))
  }
})


test_that("reproduce-regression-full-data", {
  set.seed(-425222977)
  seeds <- sample.int(.Machine$integer.max, N_REPEAT)
  for (i in seq_len(N_REPEAT)) {
    expect_no_error(test_reproduce_regression(
      100,
      4,
      c(2, 3, 5, 2, 2),
      3,
      1,
      TRUE,
      N_THREAD,
      DEFAULT_MAXITER,
      DEFAULT_TOL,
      0,
      seeds[i]
    ))
  }

  set.seed(-257866430)
  seeds <- sample.int(.Machine$integer.max, N_REPEAT)
  for (i in seq_len(N_REPEAT)) {
    expect_no_error(test_reproduce_regression(
      100,
      4,
      c(2, 3, 5, 2, 2),
      3,
      1,
      FALSE,
      N_THREAD,
      DEFAULT_MAXITER,
      DEFAULT_TOL,
      0,
      seeds[i]
    ))
  }
})

test_that("reproduce-regression-missing-data", {
  set.seed(1117500770)
  seeds <- sample.int(.Machine$integer.max, N_REPEAT)
  for (i in seq_len(N_REPEAT)) {
    expect_no_error(test_reproduce_regression(
      100,
      4,
      c(2, 3, 5, 2, 2),
      3,
      1,
      TRUE,
      N_THREAD,
      DEFAULT_MAXITER,
      DEFAULT_TOL,
      0.1,
      seeds[i]
    ))
  }

  set.seed(1405265156)
  seeds <- sample.int(.Machine$integer.max, N_REPEAT)
  for (i in seq_len(N_REPEAT)) {
    expect_no_error(test_reproduce_regression(
      100,
      4,
      c(2, 3, 5, 2, 2),
      3,
      1,
      FALSE,
      N_THREAD,
      DEFAULT_MAXITER,
      DEFAULT_TOL,
      0.1,
      seeds[i]
    ))
  }
})
