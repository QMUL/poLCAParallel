#' Calculates the likelihood for each data point and each latent class
#'
#' Calculates the likelihood for each data point and each latent class
#'
#' @param vectorized_probs A list containing:
#'   - vecprobs: vector of outcome probabilities, a flattened list of matrices
#'     - dim 0: for each outcome
#'     - dim 1: for each category
#'     - dim 2: for each cluster
#'     - in other words, imagine a nested loop, from outer to inner:
#'         - for each cluster, for each category, for each outcome
#'   - numChoices: vector, number of outcomes for each category
#'   - classes: integer, number of classes (or clusters)
#' Can be the output of poLCAParallel.vectorize()
#' @param responses A design matrix of responses
#' @return A matrix of likelihoodswith dimensions n_data x n_cluster
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
