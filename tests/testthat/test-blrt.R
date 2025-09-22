#' Test blrt()
#'
#' Test the function blrt(). Two models (null and alt) are fitted onto random
#' data and then passed to the function. Test the output of that function
#'
#' Some notes about blrt
#'
#' - Only the non-regression problem is supported
#' - Missing data is supported duirng the fitting of the null and alt model but
#'   missing-ness is not considered in blrt()
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
test_blrt <- function(n_data, n_outcomes, n_cluster, n_rep, na_rm,
                      n_thread, maxiter, tol, prob_na, n_bootstrap) {
  responses <- as.data.frame(random_response(n_data, n_outcomes, prob_na, NaN))
  formula <- formula(
    paste0("cbind(", paste(colnames(responses), collapse = ","), ")~1")
  )
  null_model <- poLCAParallel::poLCA(formula, responses, n_cluster,
    maxiter = maxiter, tol = tol, na.rm = na_rm, nrep = n_rep,
    verbose = FALSE, n.thread = n_thread
  )
  alt_model <- poLCAParallel::poLCA(formula, responses, n_cluster + 1,
    maxiter = maxiter, tol = tol, na.rm = na_rm, nrep = n_rep,
    verbose = FALSE, n.thread = n_thread
  )

  bootstrap_results <- poLCAParallel::blrt(
    null_model, alt_model, n_bootstrap,
    n_thread = n_thread, n_rep = n_rep
  )
  expect_identical(
    length(bootstrap_results$bootstrap_log_ratio),
    as.integer(n_bootstrap)
  )

  expect_lte(bootstrap_results$p_value, 1)
  expect_gte(bootstrap_results$p_value, 0)
}


test_that("full-data", {
  set.seed(-15347082)
  expect_no_error(test_blrt(
    100,
    c(2, 3, 5, 2, 2),
    3,
    1,
    TRUE,
    4,
    1000,
    1e-10,
    0,
    20
  ))

  set.seed(-15347082)
  expect_no_error(test_blrt(
    100,
    c(2, 3, 5, 2, 2),
    3,
    1,
    FALSE,
    4,
    1000,
    1e-10,
    0,
    20
  ))
})

test_that("missing-data", {
  set.seed(-15347082)
  expect_no_error(test_blrt(
    100,
    c(2, 3, 5, 2, 2),
    3,
    1,
    TRUE,
    4,
    1000,
    1e-10,
    0.1,
    20
  ))

  set.seed(-15347082)
  expect_no_error(test_blrt(
    100,
    c(2, 3, 5, 2, 2),
    3,
    1,
    FALSE,
    4,
    1000,
    1e-10,
    0.1,
    20
  ))
})
