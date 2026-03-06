#' Put all outcome probabilities into a single vector
#'
#' Given a list of matrices of outcome probabilities, flatten it into a vector,
#' suitable for `EmAlgorithmRcpp()`. It mimics `poLCA.vectorize()` but with some
#' of the dimensions swapped to improve cache efficiency in the C++ code.
#'
#' @param probs list for each category/manifest variable. For the `i`th entry,
#' it contains a matrix of outcome probabilities with dimensions `n_class` x
#' `n_outcomes[i]`
#' @return list of three items (`vecprobs`, `numChoices`, `classes`) where
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
#' @noRd
poLCAParallel.vectorize <- function(probs) {
  classes <- nrow(probs[[1]])
  vecprobs <- c()
  for (m in seq_len(classes)) {
    for (j in seq_len(length(probs))) {
      vecprobs <- c(vecprobs, probs[[j]][m, ])
    }
  }
  num_choices <- sapply(probs, ncol)
  return(list(
    vecprobs = vecprobs, numChoices = num_choices, classes = classes
  ))
}
