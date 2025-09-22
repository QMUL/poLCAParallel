#' Test poLCAParallel.se() for the non-regression problem
#'
#' Test the function poLCAParallel.se() for the non-regression problem. A list
#' containing random data and random resulting poLCA fit is generated and then
#' passed to the function, which modifies the list. The modified contents are
#' then tested
#'
#' @param n_data Number of data points
#' @param n_outcomes Vector of integers, number of outcomes for each category
#' @param n_cluster Number of clusters fitted
#' @param prob_na Probability of missing data
#' @param is_smooth Logical, if to smooth the probabilities when calculating the
#' standard errors
test_non_regression_se <- function(n_data, n_outcomes, n_cluster,
                                   prob_na, is_smooth) {
  features <- matrix(0, nrow = n_data, ncol = 1)
  responses <- as.data.frame(random_response(n_data, n_outcomes, prob_na))

  probs <- random_unvectorized_probs(n_outcomes, n_cluster)

  # random prior
  # the prior is the same for each data point in the non-regression problem
  # so use rep()
  prior <- runif(n_cluster)
  prior <- prior / sum(prior)
  prior <- matrix(rep(prior, each = n_data), nrow = n_data, ncol = n_cluster)

  # random posterior
  posterior <- random_cluster_probs(n_data, n_cluster)

  results <- list(
    x = features, y = responses, probs = probs, prior = prior,
    posterior = posterior
  )
  results <- poLCAParallel::poLCAParallel.se(results, is_smooth)

  test_standard_error(results, n_outcomes, n_cluster)
}

#' Test poLCAParallel.se() for the regression problem
#'
#' Test the function poLCAParallel.se() for the non-regression problem. A list
#' containing random data and random resulting poLCA fit is generated and then
#' passed to the function, which modifies the list. The modified contents are
#' then tested
#'
#' @param n_data Number of data points
#' @param n_feature Number of features
#' @param n_outcomes Vector of integers, number of outcomes for each category
#' @param n_cluster Number of clusters fitted
#' @param prob_na Probability of missing data
#' @param is_smooth Logical, if to smooth the probabilities when calculating the
#' standard errors
test_regression_se <- function(n_data, n_feature, n_outcomes, n_cluster,
                               prob_na, is_smooth) {
  features <- matrix(rnorm(n_data * n_feature),
    nrow = n_data, ncol = n_feature
  )

  responses <- as.data.frame(random_response(n_data, n_outcomes, prob_na))

  probs <- random_unvectorized_probs(n_outcomes, n_cluster)

  prior <- random_cluster_probs(n_data, n_cluster)

  posterior <- random_cluster_probs(n_data, n_cluster)

  results <- list(
    x = features, y = responses, probs = probs, prior = prior,
    posterior = posterior
  )
  results <- poLCAParallel::poLCAParallel.se(results, is_smooth)

  test_standard_error(results, n_outcomes, n_cluster)

  test_standard_coeff_error(results, n_feature, n_cluster)
}

# tests vary if have missing data, non-regression or regression problem and if
# to use smoothing or not

test_that("non-regression-full-data", {
  set.seed(-750018826)
  expect_no_error(test_non_regression_se(100, c(2, 3, 5, 2, 2), 3, 0, FALSE))
})

test_that("non-regression-missing-data", {
  set.seed(784260511)
  expect_no_error(test_non_regression_se(100, c(2, 3, 5, 2, 2), 3, 0.1, FALSE))
})

test_that("smooth-non-regression-full-data", {
  set.seed(-1423264693)
  expect_no_error(test_non_regression_se(100, c(2, 3, 5, 2, 2), 3, 0, TRUE))
})

test_that("smooth-non-regression-missing-data", {
  set.seed(1406829310)
  expect_no_error(test_non_regression_se(100, c(2, 3, 5, 2, 2), 3, 0.1, TRUE))
})

test_that("regression-full-data", {
  set.seed(1932351167)
  expect_no_error(test_regression_se(100, 4, c(2, 3, 5, 2, 2), 3, 0, FALSE))
})

test_that("regression-missing-data", {
  set.seed(-2016351101)
  expect_no_error(test_regression_se(100, 4, c(2, 3, 5, 2, 2), 3, 0.1, FALSE))
})

test_that("smooth-regression-full-data", {
  set.seed(-1661617308)
  expect_no_error(test_regression_se(100, 4, c(2, 3, 5, 2, 2), 3, 0, TRUE))
})

test_that("smooth-regression-missing-data", {
  set.seed(1797624161)
  expect_no_error(test_regression_se(100, 4, c(2, 3, 5, 2, 2), 3, 0.1, TRUE))
})
