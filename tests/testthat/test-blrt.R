#' Test `blrt()`
#'
#' Test the function `blrt()`. Two models (null and alt) are fitted onto random
#' data and then passed to the function. Test the output of that function
#'
#' Test if results can be reproduced using single thread and multiple threads
#'
#' Some notes about blrt
#'
#' - Only the non-regression problem is supported
#' - Missing data is supported during the fitting of the null and alt model but
#'   missing-ness is not considered in `blrt()`
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
#' @param n_bootstrap Number of bootstrap samples
#' @param seed Seed to generate random data and seed poLCA and blrt
test_blrt <- function(n_data, n_outcomes, n_cluster, n_rep, na_rm,
                      n_thread, maxiter, tol, prob_na, n_bootstrap, seed) {
  set.seed(seed)
  responses <- random_response(n_data, n_outcomes, prob_na, NaN)
  formula <- get_non_regression_formula(responses)
  lc_null <- poLCAParallel::poLCA(formula, responses, n_cluster,
    maxiter = maxiter, tol = tol, na.rm = na_rm, nrep = n_rep,
    verbose = FALSE, n.thread = n_thread
  )
  lc_alt <- poLCAParallel::poLCA(formula, responses, n_cluster + 1,
    maxiter = maxiter, tol = tol, na.rm = na_rm, nrep = n_rep,
    verbose = FALSE, n.thread = n_thread
  )

  set.seed(seed)
  bootstrap_results <- poLCAParallel::blrt(
    lc_null, lc_alt, n_bootstrap,
    n_thread = n_thread, n_rep = n_rep
  )
  expect_identical(
    length(bootstrap_results$bootstrap_log_ratio),
    as.integer(n_bootstrap)
  )

  expect_lte(bootstrap_results$p_value, 1)
  expect_gte(bootstrap_results$p_value, 0)

  # check results can be reproduced with n.thread = 1
  set.seed(seed)
  bootstrap_results_single <- poLCAParallel::blrt(
    lc_null, lc_alt, n_bootstrap,
    n_thread = 1, n_rep = n_rep
  )

  expect_equal(
    bootstrap_results$fitted_log_ratio,
    bootstrap_results_single$fitted_log_ratio
  )
  expect_equal(
    bootstrap_results$bootstrap_log_ratio,
    bootstrap_results_single$bootstrap_log_ratio
  )
  expect_equal(
    bootstrap_results$p_value,
    bootstrap_results_single$p_value
  )
}


test_that("full-data", {
  set.seed(-15347082)
  seeds <- sample.int(.Machine$integer.max, N_REPEAT)
  for (i in seq_len(N_REPEAT)) {
    expect_no_error(test_blrt(
      100,
      c(2, 3, 5, 2, 2),
      3,
      1,
      TRUE,
      N_THREAD,
      DEFAULT_MAXITER,
      DEFAULT_TOL,
      0,
      20,
      seeds[i]
    ))
  }

  set.seed(-880779424)
  seeds <- sample.int(.Machine$integer.max, N_REPEAT)
  for (i in seq_len(N_REPEAT)) {
    expect_no_error(test_blrt(
      100,
      c(2, 3, 5, 2, 2),
      3,
      1,
      FALSE,
      N_THREAD,
      DEFAULT_MAXITER,
      DEFAULT_TOL,
      0,
      20,
      seeds[i]
    ))
  }
})

test_that("missing-data", {
  set.seed(-494042289)
  seeds <- sample.int(.Machine$integer.max, N_REPEAT)
  for (i in seq_len(N_REPEAT)) {
    expect_no_error(test_blrt(
      100,
      c(2, 3, 5, 2, 2),
      3,
      1,
      TRUE,
      N_THREAD,
      DEFAULT_MAXITER,
      DEFAULT_TOL,
      0.1,
      20,
      seeds[i]
    ))
  }

  set.seed(-15347082)
  seeds <- sample.int(.Machine$integer.max, N_REPEAT)
  for (i in seq_len(N_REPEAT)) {
    expect_no_error(test_blrt(
      100,
      c(2, 3, 5, 2, 2),
      3,
      1,
      FALSE,
      N_THREAD,
      DEFAULT_MAXITER,
      DEFAULT_TOL,
      0.1,
      20,
      seeds[i]
    ))
  }
})
