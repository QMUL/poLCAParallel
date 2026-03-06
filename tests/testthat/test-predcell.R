#' Test if the resulting `poLCA.predcell()` are the same
#'
#' Test if the resulting `poLCA::poLCA.predcell()` and
#' `poLCAParallel::poLCA.predcell()` are the same. The original code can produce
#' NaN of Inf, these should be ignored as the poLCAParallel implementation
#' should be more robust
#'
#' @param predcell_parallel Resulting `poLCAParallel::poLCA.predcell()`
#' @param predcell_polca Resulting `poLCA::poLCA.predcell()`
test_predcell <- function(predcell_parallel, predcell_polca) {
  # expect same results when using poLCA::poLCA.predcell()
  # and poLCAParallel::poLCA.predcell()
  # however the original poLCA::poLCA.predcell() can produce NaN or Inf
  # ignore them
  is_finite_index <- is.finite(predcell_polca)
  expect_equal(
    predcell_parallel[is_finite_index],
    predcell_polca[is_finite_index]
  )
  # test if all values are finite
  expect_identical(all(is.finite(predcell_parallel)), TRUE)
}

#' Test the function `poLCA.predcell()` for the non-regression problem
#'
#' Test the function `poLCA.predcell()` for the non-regression problem. The
#' model is fitted on data and then passed to the function with fully observed
#' data. The test compares the results with the original poLCA code
#'
#' #############################################################################
#' As with the original code, partially observed responses are not supported
#' #############################################################################
#'
#' @param n_data Number of data points
#' @param n_outcomes Vector of integers, number of outcomes for each category
#' @param n_cluster Number of clusters fitted
#' @param n_rep Number of different initial values to try
#' @param na_rm Logical, if to remove NA responses
#' @param n_thread Number of threads to use
#' @param maxiter Number of iterations used in the EM algorithm
#' @param tol Tolerance used in the EM algorithm
#' @param prob_na_train Probability of missing data in the training data
#' @param n_data_test Number of data points in the unseen test data
#' @param prob_na_test Probability of missing data in the unseen test data
test_non_regress_predcell <- function(n_data, n_outcomes, n_cluster, n_rep,
                                      na_rm, n_thread, maxiter, tol,
                                      prob_na_train, n_data_test,
                                      prob_na_test) {
  responses <- as.data.frame(
    random_response(n_data, n_outcomes, prob_na_train, NaN)
  )
  formula <- get_non_regression_formula(responses)
  lc <- poLCAParallel::poLCA(formula, responses, n_cluster,
    maxiter = maxiter, tol = tol, na.rm = na_rm, nrep = n_rep,
    verbose = FALSE, n.thread = n_thread
  )

  # using training data
  test_predcell(
    poLCAParallel::poLCA.predcell(lc, lc$y),
    poLCA::poLCA.predcell(lc, lc$y)
  )

  # fully observed data
  responses <- random_response(n_data_test, n_outcomes, 0, NaN)
  test_predcell(
    poLCAParallel::poLCA.predcell(lc, responses),
    poLCA::poLCA.predcell(lc, responses)
  )

  # partially observed data not supported
}

#' Test the function `poLCA.predcell()` for the regression problem
#'
#' Test the function `poLCA.predcell()` for the non-regression problem. The
#' model is fitted on data and then passed to the function with fully observed
#' data. The test compares the results with the original poLCA code
#'
#' #############################################################################
#' As with the original code, partially observed responses are not supported
#' #############################################################################
#'
#' @param n_data Number of data points
#' @param n_outcomes Vector of integers, number of outcomes for each category
#' @param n_cluster Number of clusters fitted
#' @param n_rep Number of different initial values to try
#' @param na_rm Logical, if to remove NA responses
#' @param n_thread Number of threads to use
#' @param maxiter Number of iterations used in the EM algorithm
#' @param tol Tolerance used in the EM algorithm
#' @param prob_na_train Probability of missing data in the training data
#' @param n_data_test Number of data points in the unseen test data
#' @param prob_na_test Probability of missing data in the unseen test data
test_regress_predcell <- function(n_data, n_feature, n_outcomes, n_cluster,
                                  n_rep, na_rm, n_thread, maxiter, tol,
                                  prob_na_train, n_data_test, prob_na_test) {
  features <- random_features(n_data, n_feature)
  responses <- random_response(n_data, n_outcomes, prob_na_train, NaN)
  formula <- get_regression_formula(responses, features)
  data <- cbind(responses, features)

  lc <- poLCAParallel::poLCA(formula, data, n_cluster,
    maxiter = maxiter, tol = tol, na.rm = na_rm, nrep = n_rep,
    verbose = FALSE, n.thread = n_thread
  )

  # using training data
  test_predcell(
    poLCAParallel::poLCA.predcell(lc, lc$y),
    poLCA::poLCA.predcell(lc, lc$y)
  )

  # fully observed data
  responses <- random_response(n_data_test, n_outcomes, 0, NaN)
  test_predcell(
    poLCAParallel::poLCA.predcell(lc, responses),
    poLCA::poLCA.predcell(lc, responses)
  )

  # partially observed data not supported
}

test_that("non-regression-full-data", {
  # test using na_rm = TRUE and FALSE
  set.seed(1183913236)
  seeds <- sample.int(.Machine$integer.max, N_REPEAT)
  for (i in seq_len(N_REPEAT)) {
    set.seed(seeds[i])
    expect_no_error(test_non_regress_predcell(
      100,
      c(2, 3, 5, 2, 2),
      3,
      4,
      TRUE,
      N_THREAD,
      DEFAULT_MAXITER,
      DEFAULT_TOL,
      0,
      50,
      0.01
    ))
  }

  set.seed(-1141474643)
  seeds <- sample.int(.Machine$integer.max, N_REPEAT)
  for (i in seq_len(N_REPEAT)) {
    set.seed(seeds[i])
    expect_no_error(test_non_regress_predcell(
      100,
      c(2, 3, 5, 2, 2),
      3,
      4,
      FALSE,
      N_THREAD,
      DEFAULT_MAXITER,
      DEFAULT_TOL,
      0,
      50,
      0.01
    ))
  }
})

test_that("non-regression-missing-data", {
  # na_rm = FALSE not supported with missing data
  set.seed(-1688010496)
  seeds <- sample.int(.Machine$integer.max, N_REPEAT)
  for (i in seq_len(N_REPEAT)) {
    set.seed(seeds[i])
    expect_no_error(test_non_regress_predcell(
      100,
      c(2, 3, 5, 2, 2),
      3,
      4,
      TRUE,
      N_THREAD,
      DEFAULT_MAXITER,
      DEFAULT_TOL,
      0.1,
      50,
      0.01
    ))
  }
})

test_that("regression-full-data", {
  # test using na_rm = TRUE and FALSE
  set.seed(-377644738)
  seeds <- sample.int(.Machine$integer.max, N_REPEAT)
  for (i in seq_len(N_REPEAT)) {
    set.seed(seeds[i])
    expect_no_error(test_regress_predcell(
      100,
      4,
      c(2, 3, 5, 2, 2),
      3,
      4,
      TRUE,
      N_THREAD,
      DEFAULT_MAXITER,
      DEFAULT_TOL,
      0,
      50,
      0.01
    ))
  }

  set.seed(-1620100671)
  seeds <- sample.int(.Machine$integer.max, N_REPEAT)
  for (i in seq_len(N_REPEAT)) {
    set.seed(seeds[i])
    expect_no_error(test_regress_predcell(
      100,
      4,
      c(2, 3, 5, 2, 2),
      3,
      4,
      FALSE,
      N_THREAD,
      DEFAULT_MAXITER,
      DEFAULT_TOL,
      0,
      50,
      0.01
    ))
  }
})

test_that("regression-missing-data", {
  # na_rm = FALSE not supported with missing data
  set.seed(215886219)
  seeds <- sample.int(.Machine$integer.max, N_REPEAT)
  for (i in seq_len(N_REPEAT)) {
    set.seed(seeds[i])
    expect_no_error(test_regress_predcell(
      100,
      4,
      c(2, 3, 5, 2, 2),
      3,
      4,
      TRUE,
      N_THREAD,
      DEFAULT_MAXITER,
      DEFAULT_TOL,
      0.1,
      50,
      0.01
    ))
  }
})
