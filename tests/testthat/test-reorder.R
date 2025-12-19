#' Test the function poLCA.reorder()
#'
#' Test the function poLCAParallel::poLCA.reorder()() against the original
#' poLCA::poLCA.reorder(). Randomly generated probabilities and a random order
#' are passed to the function
#'
#' @param n_outcomes number of outcomes
#' @param n_cluster number of clusters
test_reproduce <- function(n_outcomes, n_cluster) {
  n_sample <- 10

  probs <- random_unvectorized_probs(n_outcomes, n_cluster)

  # repeat for different orderings
  for (i in seq_len(n_sample)) {
    order <- sample(seq_len(n_cluster))
    probs_reorder_polca <- poLCA::poLCA.reorder(probs, order)
    probs_reorder_parallel <- poLCAParallel::poLCA.reorder(probs, order)
    expect_equal(probs_reorder_parallel, probs_reorder_polca)
  }
}

#' Test if the resulting reordered probabilities can be re-used
#'
#' Test if the poLCA attributes `probs.start` and `probs` can be used in the
#' function `poLCA.reorder()`. Also tests if the return value of
#' `poLCA.reorder()` can be used as the `probs.start` argument of the function
#' `poLCA()`
#'
#' @param n_data Number of data points
#' @param n_outcomes Vector of integers, number of outcomes for each category
#' @param n_cluster Number of clusters to fit
#' @param n_rep Number of different initial values to try
#' @param na_rm Logical, if to remove NA responses
#' @param n_thread Number of threads to use
#' @param maxiter Number of iterations used in the EM algorithm
#' @param tol Tolerance used in the EM algorithm
#' @param prob_na Probability of missing data
test_reuse <- function(n_data, n_outcomes, n_cluster, n_rep,
                       na_rm, n_thread, maxiter, tol,
                       prob_na) {
  responses <- random_response(n_data, n_outcomes, prob_na, NaN)
  formula <- get_non_regression_formula(responses)

  lc <- poLCAParallel::poLCA(formula, responses, n_cluster,
    maxiter = maxiter, tol = tol, na.rm = na_rm, nrep = n_rep,
    verbose = FALSE, n.thread = n_thread
  )

  probs_reorder <- poLCAParallel::poLCA.reorder(
    lc$probs.start,
    order(lc$P, decreasing = TRUE)
  )

  # one repetition as the purpose is to use the reordered probabilities only
  lc <- poLCAParallel::poLCA(formula, responses, n_cluster,
    maxiter = maxiter, tol = tol, na.rm = na_rm, probs.start = probs_reorder,
    nrep = 1, verbose = FALSE, n.thread = n_thread
  )

  probs_reorder <- poLCAParallel::poLCA.reorder(
    lc$probs, order(lc$P, decreasing = TRUE)
  )

  # one repetition as the purpose is to use the reordered probabilities only
  lc <- poLCAParallel::poLCA(formula, responses, n_cluster,
    maxiter = maxiter, tol = tol, na.rm = na_rm, probs.start = probs_reorder,
    nrep = 1, verbose = FALSE, n.thread = n_thread
  )
}

test_that("reproduce", {
  set.seed(-648072421)
  for (i in seq_len(20)) {
    n_cluster <- rpois(1, 10) + 2
    n_category <- rpois(1, 10) + 2
    n_outcomes <- rpois(n_category, 0.5) + 2
    expect_no_error(test_reproduce(n_outcomes, n_cluster))
  }
})

test_that("reorder", {
  set.seed(966670512)
  expect_no_error(test_reuse(
    100,
    c(2, 3, 5, 2, 2),
    3,
    4,
    TRUE,
    2,
    1000,
    1e-10,
    0
  ))
})
