#' Calculates the likelihood for each data point conditioned on latent classes
#'
#' Calculates the likelihood for each data point conditioned for each latent
#' class
#'
#' @param vectorized_probs  Can be the output of `poLCAParallel.vectorize()`.
#' List of three items (`vecprobs`, `numChoices`, `classes`) where
#' * `vecprobs`: vector of outcome probabilities conditioned on the
#'   manifest/category and the class/cluster. Imagine a nested loop, from inner
#'   to outer, or a flatten column-major matrix, the probabilities are arranged
#'   in the following order:
#'   * dim 1: for each outcome
#'   * dim 2: for each manifest/category
#'   * dim 3: for each class/cluster
#' * `numChoices`: integer vector, number of outcomes for each category/manifest
#'   variable
#' * `classes`: integer, number of latent classes (or clusters)
#' @param responses A design matrix of responses with dimensions
#' * dim 1: for each data
#' * dim 2: for each category/manifest variable
#' @return A matrix of likelihoods with dimensions
#' * dim 1: for each data
#' * dim 2: for each class/cluster
#'
#' @export
likelihood <- function(vectorized_probs, responses) {
  likelihood_ <- LikelihoodRcpp(
    t(responses),
    vectorized_probs$vecprobs,
    vectorized_probs$numChoices,
    dim(responses)[1],
    vectorized_probs$classes
  )
  return(likelihood_)
}
