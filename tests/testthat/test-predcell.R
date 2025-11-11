#' Test the function poLCA.predcell() for the non-regression problem
#'
#' Test the function poLCA.predcell() for the non-regression problem. The model
#' is fitted on data and then passed to the function with fully observed data.
#' The test compares the results with the original poLCA code
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
  predcell_polca <- poLCA::poLCA.predcell(lc, lc$y)
  predcell_polcaparallel <- poLCAParallel::poLCA.predcell(lc, lc$y)
  expect_equal(predcell_polcaparallel, predcell_polca)

  # fully observed data
  responses <- random_response(n_data_test, n_outcomes, 0, NaN)
  predcell_polca <- poLCA::poLCA.predcell(lc, responses)
  predcell_polcaparallel <- poLCAParallel::poLCA.predcell(lc, responses)
  expect_equal(predcell_polcaparallel, predcell_polca)

  # partially observed data not supported
}

#' Test the function poLCA.posterior() for the regression problem
#'
#' Test the function poLCA.predcell() for the non-regression problem. The model
#' is fitted on data and then passed to the function with fully observed data.
#' The test compares the results with the original poLCA code
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
  predcell_polca <- poLCA::poLCA.predcell(lc, lc$y)
  predcell_polcaparallel <- poLCAParallel::poLCA.predcell(lc, lc$y)
  expect_equal(predcell_polcaparallel, predcell_polca)

  # fully observed data
  responses <- random_response(n_data_test, n_outcomes, 0, NaN)
  predcell_polca <- poLCA::poLCA.predcell(lc, responses)
  predcell_polcaparallel <- poLCAParallel::poLCA.predcell(lc, responses)
  expect_equal(predcell_polcaparallel, predcell_polca)

  # partially observed data not supported
}

test_that("non-regression-full-data", {
  # test using na_rm = TRUE and FALSE
  set.seed(1183913236)
  expect_no_error(test_non_regress_predcell(
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

  set.seed(-1141474643)
  expect_no_error(test_non_regress_predcell(
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
  # na_rm = FALSE not supported with missing data
  set.seed(-1688010496)
  expect_no_error(test_non_regress_predcell(
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
})

test_that("regression-full-data", {
  # test using na_rm = TRUE and FALSE
  set.seed(-377644738)
  expect_no_error(test_non_regress_predcell(
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

  set.seed(-1620100671)
  expect_no_error(test_non_regress_predcell(
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

test_that("regression-missing-data", {
  # na_rm = FALSE not supported with missing data
  set.seed(215886219)
  expect_no_error(test_non_regress_predcell(
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
})
