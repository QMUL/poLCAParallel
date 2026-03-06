#' Test and compare the resulting `poLCA.posterior()` function
#'
#' Test and compare the resulting `poLCAParallel::poLCA.posterior()` with the
#' original `poLCA::poLCA.posterior()`. They should produce the same results
#'
#' Also tests if the resulting `poLCAParallel::poLCA.posterior()` are valid
#' probabilities and finite
#'
#' It is possible the original `poLCA::poLCA.posterior()` may produce NaN or
#' Inf. They are ignored in the comparison
#'
#' @param posterior_parallel Return value of `poLCAParallel::poLCA.posterior()`
#' @param posterior_polca Return value of `poLCA::poLCA.posterior()`
test_posterior <- function(posterior_parallel,
                           posterior_polca) {
  expect_identical(all(is.finite(posterior_parallel)), TRUE)
  test_cluster_probs(
    posterior_parallel,
    nrow(posterior_parallel), ncol(posterior_parallel)
  )
  # only do a comparison test if the original code produce finite results
  if (all(is.finite(posterior_polca))) {
    expect_equal(posterior_parallel, posterior_polca)
  }
}

#' Test the function `poLCA.posterior()` for the non-regression problem
#'
#' Test the function `poLCA.posterior()` for the non-regression problem. The
#' model is fitted on data and then used to work out the posterior for the
#' training data, unseen no-missing test data and unseen with-missing test data.
#' The test compares the results with the original poLCA code
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
test_non_regress_posterior <- function(n_data, n_outcomes, n_cluster, n_rep,
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
  test_posterior(
    poLCAParallel::poLCA.posterior(lc, lc$y),
    poLCA::poLCA.posterior(lc, lc$y)
  )

  # fully observed data
  responses <- random_response(n_data_test, n_outcomes, 0, NaN)
  test_posterior(
    poLCAParallel::poLCA.posterior(lc, responses),
    poLCA::poLCA.posterior(lc, responses)
  )

  # partially observed data
  responses <- random_response(n_data_test, n_outcomes, prob_na_test, NaN)
  test_posterior(
    poLCAParallel::poLCA.posterior(lc, responses),
    poLCA::poLCA.posterior(lc, responses)
  )
}

#' Test the function `poLCA.posterior()` for the regression problem
#'
#' Test the function `poLCA.posterior()` for the non-regression problem. The
#' model is fitted on data and then used to work out the posterior for the
#' training data, unseen no-missing test data and unseen with-missing test data.
#' The test compares the results with the original poLCA code
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
#' @param prob_na_train Probability of missing data in the training data
#' @param n_data_test Number of data points in the unseen test data
#' @param prob_na_test Probability of missing data in the unseen test data
test_regress_posterior <- function(n_data, n_feature, n_outcomes, n_cluster,
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
  test_posterior(
    poLCAParallel::poLCA.posterior(lc, lc$y),
    poLCA::poLCA.posterior(lc, lc$y)
  )

  # fully observed data
  responses <- random_response(n_data_test, n_outcomes, 0, NaN)
  test_posterior(
    poLCAParallel::poLCA.posterior(lc, responses),
    poLCA::poLCA.posterior(lc, responses)
  )

  # partially observed data
  responses <- random_response(n_data_test, n_outcomes, prob_na_test, NaN)
  test_posterior(
    poLCAParallel::poLCA.posterior(lc, responses),
    poLCA::poLCA.posterior(lc, responses)
  )
}


test_that("non-regression-full-data", {
  # test using na_rm = TRUE and FALSE
  set.seed(-1381922797)
  seeds <- sample.int(.Machine$integer.max, N_REPEAT)
  for (i in seq_len(N_REPEAT)) {
    set.seed(seeds[i])
    expect_no_error(test_non_regress_posterior(
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

  set.seed(481136649)
  seeds <- sample.int(.Machine$integer.max, N_REPEAT)
  for (i in seq_len(N_REPEAT)) {
    set.seed(seeds[i])
    expect_no_error(test_non_regress_posterior(
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
  # test using na_rm = TRUE and FALSE
  set.seed(1210610989)
  seeds <- sample.int(.Machine$integer.max, N_REPEAT)
  for (i in seq_len(N_REPEAT)) {
    set.seed(seeds[i])
    expect_no_error(test_non_regress_posterior(
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

  set.seed(1304862690)
  seeds <- sample.int(.Machine$integer.max, N_REPEAT)
  for (i in seq_len(N_REPEAT)) {
    set.seed(seeds[i])
    expect_no_error(test_non_regress_posterior(
      100,
      c(2, 3, 5, 2, 2),
      3,
      4,
      FALSE,
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
  set.seed(-1529442620)
  seeds <- sample.int(.Machine$integer.max, N_REPEAT)
  for (i in seq_len(N_REPEAT)) {
    set.seed(seeds[i])
    expect_no_error(test_regress_posterior(
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

  set.seed(81779870)
  seeds <- sample.int(.Machine$integer.max, N_REPEAT)
  for (i in seq_len(N_REPEAT)) {
    set.seed(seeds[i])
    expect_no_error(test_regress_posterior(
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
  # test using na_rm = TRUE and FALSE
  set.seed(-1396271961)
  seeds <- sample.int(.Machine$integer.max, N_REPEAT)
  for (i in seq_len(N_REPEAT)) {
    set.seed(seeds[i])
    expect_no_error(test_regress_posterior(
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

  set.seed(63195066)
  seeds <- sample.int(.Machine$integer.max, N_REPEAT)
  for (i in seq_len(N_REPEAT)) {
    set.seed(seeds[i])
    expect_no_error(test_regress_posterior(
      100,
      4,
      c(2, 3, 5, 2, 2),
      3,
      4,
      FALSE,
      N_THREAD,
      DEFAULT_MAXITER,
      DEFAULT_TOL,
      0.1,
      50,
      0.01
    ))
  }
})
