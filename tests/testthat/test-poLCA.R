#' Test the contents of a poLCA object
#'
#' Test the Rcpp outputted contents of a poLCA object, this tests the prior
#' probabilities, posterior probabilities, outcome probabilities, log
#' likelihood, number of iterations done, initial probabilities which achieved
#' the maximum log likelihood and more...
#'
#' Provide the poLCA object and parameters which are used to test the object
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

#' Test the other contents of a poLCA object
#'
#' Test the outputted contents of a poLCA object not tested in
#' test_polca_em_algorithm(). It tests the R outputs (not Rcpp) created on
#' poLCA() such as the features, responses, number of data points and time
#' taken... etc
#'
#' Provide the poLCA object and parameters which are used to test the object
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

#' Test using poLCA() for the non-regression problem
#'
#' Test using poLCA() for the non-regression problem. Random responses are
#' generated and then passed to poLCA() to be fitted onto. The results are then
#' tested
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

#' Test using poLCA() for the regression problem
#'
#' Test using poLCA() for the regression problem. Random responses and features
#' are generated and then passed to poLCA() to be fitted onto. The results are
#' then tested
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

#' Test if the results fitted model is the same as the original code
#'
#' Test if the results fitted models, using poLCA and poLCAParallel, are the
#' same. It tests the attributes of the fitted models
#'
#' @param lc_parallel A model object estimated using the `poLCAParallel::poLCA`
#'   function (or a list which mocks it)
#' @param lc_polca A model object estimated using the `poLCA::poLCA` function
#'   (or a list which mocks it)
#' @param is_regression boolean, if the problem is a regression problem
test_equal <- function(lc_parallel, lc_polca, is_regression) {
  equal_tol_gof <- 1e2 * sqrt(.Machine$double.eps)
  equal_tol_prob <- 1e3 * sqrt(.Machine$double.eps)

  # test if all attributes in the og code is in our code
  for (attribute_i in names(lc_polca)) {
    expect_identical(attribute_i %in% names(lc_parallel), TRUE)
  }

  if (!is_regression) {
    # test if results are the same
    expect_equal(lc_parallel$llik, lc_polca$llik)
    expect_equal(lc_parallel$aic, lc_polca$aic)
    expect_equal(lc_parallel$bic, lc_polca$bic)
    expect_equal(lc_parallel$Nobs, lc_polca$Nobs)

    expect_equal(lc_parallel$Chisq, lc_polca$Chisq,
      tolerance = equal_tol_gof
    )
    expect_equal(lc_parallel$Gsq, lc_polca$Gsq,
      tolerance = equal_tol_gof
    )

    expect_equal(lc_parallel$probs, lc_polca$probs,
      tolerance = equal_tol_prob
    )
    expect_equal(lc_parallel$P, lc_polca$P,
      tolerance = equal_tol_prob
    )
    expect_equal(lc_parallel$posterior, lc_polca$posterior,
      tolerance = equal_tol_prob
    )

    # in predcell, the og code rounds the expected frequency
    pred_cell_rounded <- lc_parallel$predcell
    pred_cell_rounded$expected <- round(lc_parallel$predcell$expected, 3)
    expect_identical(all.equal(pred_cell_rounded, lc_polca$predcell), TRUE)
  }

  expect_equal(lc_parallel$probs.start.ok, lc_polca$probs.start.ok)
  expect_equal(lc_parallel$npar, lc_polca$npar)
  expect_identical(all.equal(lc_parallel$y, lc_polca$y), TRUE)
  expect_identical(all.equal(lc_parallel$x, lc_polca$x), TRUE)
}

#' Test if results are the same as original poLCA code
#'
#' Test if results are the same, or at least similar, as the original poLCA code
#' for the non-regression problem. Generate data and pass it to `poLCA::poLCA()`
#' and `poLCAParallel::poLCA()` and compare results
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
#' @param seed Seed to generate random data and seed poLCA
test_reproduce_non_regression <- function(n_data, n_outcomes, n_cluster,
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
  test_equal(lc_parallel, lc_polca, FALSE)
}

#' Test if results are the same as original poLCA code
#'
#' Test if results are the same, or at least similar, as the original poLCA code
#' for the regression problem. Generate data and pass it to poLCA::poLCA()
#' and poLCAParallel::poLCA() and compare results
#'
#' The EM algorithm does depend on the initial values and how many different
#' initials were tried. A failed test could be fixed by either using a high
#' repetition count or ensure the initial values are the same
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
  test_equal(lc_parallel, lc_polca, TRUE)
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
  test_equal(lc_parallel, lc_polca, FALSE)
}

test_that("non-regression-full-data", {
  # test using na_rm = TRUE and FALSE
  # with no missing data, they both should work in the same way
  set.seed(-1012646258)
  expect_no_error(test_non_regression(
    100,
    c(2, 3, 5, 2, 2),
    3,
    4,
    TRUE,
    4,
    1000,
    1e-10,
    0
  ))

  set.seed(-2057561765)
  expect_no_error(test_non_regression(
    100,
    c(2, 3, 5, 2, 2),
    3,
    4,
    FALSE,
    4,
    1000,
    1e-10,
    0
  ))
})

test_that("non-regression-missing-data", {
  # test using na_rm = TRUE and FALSE
  # with missing data, both will produce different results
  set.seed(-1554950958)
  expect_no_error(test_non_regression(
    100,
    c(2, 3, 5, 2, 2),
    3,
    4,
    TRUE,
    4,
    1000,
    1e-10,
    0.1
  ))

  set.seed(984792451)
  expect_no_error(test_non_regression(
    100,
    c(2, 3, 5, 2, 2),
    3,
    4,
    FALSE,
    4,
    1000,
    1e-10,
    0.1
  ))
})

test_that("regression-full-data", {
  set.seed(-590845051)
  expect_no_error(test_regression(
    100,
    4,
    c(2, 3, 5, 2, 2),
    3,
    4,
    TRUE,
    4,
    1000,
    1e-10,
    0
  ))
  set.seed(1785517768)
  expect_no_error(test_regression(
    100,
    4,
    c(2, 3, 5, 2, 2),
    3,
    4,
    FALSE,
    4,
    1000,
    1e-10,
    0
  ))
})


test_that("regression-missing-data", {
  set.seed(-85141069)
  expect_no_error(test_regression(
    100,
    4,
    c(2, 3, 5, 2, 2),
    3,
    4,
    TRUE,
    4,
    1000,
    1e-10,
    0.1
  ))

  set.seed(-2070313423)
  expect_no_error(test_regression(
    100,
    4,
    c(2, 3, 5, 2, 2),
    3,
    4,
    FALSE,
    4,
    1000,
    1e-10,
    0.1
  ))
})


test_that("reproduce-non-regression-full-data", {
  expect_no_error(test_reproduce_non_regression(
    100,
    c(2, 3, 5, 2, 2),
    3,
    TRUE,
    4,
    1000,
    1e-10,
    0,
    -683307112
  ))

  expect_no_error(test_reproduce_non_regression(
    100,
    c(2, 3, 5, 2, 2),
    3,
    FALSE,
    4,
    1000,
    1e-10,
    0,
    -1855018758
  ))

  set.seed(980213281)
  expect_no_error(test_probs_start_non_regression(
    100,
    c(2, 3, 5, 2, 2),
    3,
    TRUE,
    4,
    1000,
    1e-10,
    0
  ))

  set.seed(1619464396)
  expect_no_error(test_probs_start_non_regression(
    100,
    c(2, 3, 5, 2, 2),
    3,
    FALSE,
    4,
    1000,
    1e-10,
    0
  ))
})

test_that("reproduce-non-regression-missing-data", {
  expect_no_error(test_reproduce_non_regression(
    100,
    c(2, 3, 5, 2, 2),
    3,
    TRUE,
    4,
    1000,
    1e-10,
    0.1,
    -1391069936
  ))

  expect_no_error(test_reproduce_non_regression(
    100,
    c(2, 3, 5, 2, 2),
    3,
    FALSE,
    4,
    1000,
    1e-10,
    0.1,
    799350486
  ))


  set.seed(-1158272799)
  expect_no_error(test_probs_start_non_regression(
    100,
    c(2, 3, 5, 2, 2),
    3,
    TRUE,
    4,
    1000,
    1e-10,
    0.1
  ))

  set.seed(16136553)
  expect_no_error(test_probs_start_non_regression(
    100,
    c(2, 3, 5, 2, 2),
    3,
    FALSE,
    4,
    1000,
    1e-10,
    0.1
  ))
})


test_that("reproduce-regression-full-data", {
  expect_no_error(test_reproduce_regression(
    100,
    4,
    c(2, 3, 5, 2, 2),
    3,
    1,
    TRUE,
    4,
    1000,
    1e-10,
    0,
    -425222977
  ))

  expect_no_error(test_reproduce_regression(
    100,
    4,
    c(2, 3, 5, 2, 2),
    3,
    1,
    FALSE,
    4,
    1000,
    1e-10,
    0,
    -257866430
  ))
})

test_that("reproduce-regression-missing-data", {
  expect_no_error(test_reproduce_regression(
    100,
    4,
    c(2, 3, 5, 2, 2),
    3,
    1,
    TRUE,
    4,
    1000,
    1e-10,
    0.1,
    1117500770
  ))

  expect_no_error(test_reproduce_regression(
    100,
    4,
    c(2, 3, 5, 2, 2),
    3,
    1,
    FALSE,
    4,
    1000,
    1e-10,
    0.1,
    1405265156
  ))
})
