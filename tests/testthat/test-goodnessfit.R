#' Test poLCAParallel.goodnessfit()
#'
#' Test the function poLCAParallel.goodnessfit(). A list containing random data
#' and random resulting poLCA fit is generated and then passed to the function,
#' which modifies the list. The modified contents are then tested
#'
#' @param n_data Number of data points
#' @param n_outcomes Vector of integers, number of outcomes for each category
#' @param n_cluster Number of clusters fitted
#' @param prob_na Probability of missing data
test_goodnessfit <- function(n_data, n_outcomes, n_cluster,
                             prob_na) {
  responses <- as.data.frame(random_response(n_data, n_outcomes, prob_na))
  probs <- random_unvectorized_probs(n_outcomes, n_cluster)

  prior <- runif(n_cluster)
  prior <- prior / sum(prior)

  polca <- list(
    y = responses,
    P = prior,
    N = n_data,
    probs = probs,
    Nobs = sum(rowSums(responses == 0) == 0)
  )
  polca <- poLCAParallel.goodnessfit(polca)

  test_polca_goodnessfit(polca, n_outcomes)
}

test_that("full-data", {
  set.seed(125826165)
  expect_no_error(test_goodnessfit(
    100,
    c(2, 3, 5, 2, 2),
    3,
    0
  ))
})

test_that("missing-data", {
  set.seed(680730606)
  expect_no_error(test_goodnessfit(
    100,
    c(2, 3, 5, 2, 2),
    3,
    0.1
  ))
})
