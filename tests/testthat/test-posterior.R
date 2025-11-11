#' Test the function poLCA.posterior() for the non-regression problem
#'
#' Test the function poLCA.posterior() for the non-regression problem. The model
#' is fitted on data and then used to work out the posterior for the training
#' data, unseen no-missing test data and unseen with-missing test data. The test
#' compares the results with the orginal poLCA code
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
  posterior_polca <- poLCA::poLCA.posterior(lc, lc$y)
  posterior_polcaparallel <- poLCAParallel::poLCA.posterior(lc, lc$y)
  expect_equal(posterior_polcaparallel, posterior_polca)

  # fully observed data
  responses <- random_response(n_data_test, n_outcomes, 0, NaN)
  posterior_polca <- poLCA::poLCA.posterior(lc, responses)
  posterior_polcaparallel <- poLCAParallel::poLCA.posterior(lc, responses)
  expect_equal(posterior_polcaparallel, posterior_polca)

  # partially observed data
  responses <- random_response(n_data_test, n_outcomes, prob_na_test, NaN)
  posterior_polca <- poLCA::poLCA.posterior(lc, responses)
  posterior_polcaparallel <- poLCAParallel::poLCA.posterior(lc, responses)
  expect_equal(posterior_polcaparallel, posterior_polca)
}

#' Test the function poLCA.posterior() for the regression problem
#'
#' Test the function poLCA.posterior() for the non-regression problem. The model
#' is fitted on data and then used to work out the posterior for the training
#' data, unseen no-missing test data and unseen with-missing test data. The test
#' compares the results with the original poLCA code
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

  posterior_polca <- poLCA::poLCA.posterior(lc, lc$y)
  posterior_polcaparallel <- poLCAParallel::poLCA.posterior(lc, lc$y)
  expect_equal(posterior_polcaparallel, posterior_polca)

  responses <- random_response(n_data_test, n_outcomes, 0, NaN)
  posterior_polca <- poLCA::poLCA.posterior(lc, responses)
  posterior_polcaparallel <- poLCAParallel::poLCA.posterior(lc, responses)
  expect_equal(posterior_polcaparallel, posterior_polca)

  responses <- random_response(n_data_test, n_outcomes, prob_na_test, NaN)
  posterior_polca <- poLCA::poLCA.posterior(lc, responses)
  posterior_polcaparallel <- poLCAParallel::poLCA.posterior(lc, responses)
  expect_equal(posterior_polcaparallel, posterior_polca)
}


test_that("non-regression-full-data", {
  # test using na_rm = TRUE and FALSE
  set.seed(-1381922797)
  expect_no_error(test_non_regress_posterior(
    100,
    c(2, 3, 5, 2, 2),
    3,
    4,
    TRUE,
    4,
    1000,
    1e-10,
    0,
    50,
    0.01
  ))

  set.seed(481136649)
  expect_no_error(test_non_regress_posterior(
    100,
    c(2, 3, 5, 2, 2),
    3,
    4,
    FALSE,
    4,
    1000,
    1e-10,
    0,
    50,
    0.01
  ))
})

test_that("non-regression-missing-data", {
  # test using na_rm = TRUE and FALSE
  set.seed(1210610989)
  expect_no_error(test_non_regress_posterior(
    100,
    c(2, 3, 5, 2, 2),
    3,
    4,
    TRUE,
    4,
    1000,
    1e-10,
    0.1,
    50,
    0.01
  ))

  set.seed(1304862690)
  expect_no_error(test_non_regress_posterior(
    100,
    c(2, 3, 5, 2, 2),
    3,
    4,
    FALSE,
    4,
    1000,
    1e-10,
    0.1,
    50,
    0.01
  ))
})

test_that("regression-full-data", {
  # test using na_rm = TRUE and FALSE
  set.seed(-1529442620)
  expect_no_error(test_regress_posterior(
    100,
    4,
    c(2, 3, 5, 2, 2),
    3,
    4,
    TRUE,
    4,
    1000,
    1e-10,
    0,
    50,
    0.01
  ))

  set.seed(81779870)
  expect_no_error(test_regress_posterior(
    100,
    4,
    c(2, 3, 5, 2, 2),
    3,
    4,
    FALSE,
    4,
    1000,
    1e-10,
    0,
    50,
    0.01
  ))
})

test_that("regression-missing-data", {
  # test using na_rm = TRUE and FALSE
  set.seed(-1396271961)
  expect_no_error(test_regress_posterior(
    100,
    4,
    c(2, 3, 5, 2, 2),
    3,
    4,
    TRUE,
    4,
    1000,
    1e-10,
    0.1,
    50,
    0.01
  ))

  set.seed(63195066)
  expect_no_error(test_regress_posterior(
    100,
    4,
    c(2, 3, 5, 2, 2),
    3,
    4,
    FALSE,
    4,
    1000,
    1e-10,
    0.1,
    50,
    0.01
  ))
})
