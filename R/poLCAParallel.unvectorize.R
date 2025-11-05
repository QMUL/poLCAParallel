#' Reverse the operations of `poLCAParallel.vectorize()`
#'
#' Mimics `poLCA.unvectorize()` but with some of the dimensions swapped. Given
#' the return value, or even modified, of `poLCAParallel.vectorize()`, return a
#' list which contains matrices containing outcome probabilities and other
#' information.
#'
#' @param vp list of three items (`vecprobs`, `numChoices`, `classes`) where
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
#' @return list for each category/manifest variable. For the `i`th entry, it
#' contains a matrix of outcome probabilities with dimensions `n_class` x
#' `n_outcomes[i]`
#' @noRd
poLCAParallel.unvectorize <- function(vp) {
  num_choices <- vp$numChoices
  n_category <- length(num_choices)
  # allocate matrices to the return list
  probs <- list()
  for (j in seq_len(n_category)) {
    probs[[j]] <- matrix(nrow = vp$classes, ncol = num_choices[j])
  }
  # copy over probabilities
  index <- 1
  for (m in 1:vp$classes) {
    for (j in seq_len(n_category)) {
      next_index <- index + num_choices[j] - 1
      probs[[j]][m, ] <- vp$vecprobs[index:next_index]
      index <- next_index + 1
    }
  }
  return(probs)
}
