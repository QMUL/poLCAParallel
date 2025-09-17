#' Generate random responses
#'
#' @param n_data Number of data points
#' @param n_outcomes Vector of integers, number of outcomes for each category
#' @param prob_na Probability of missing data
#' @param na_encode Value to use for missing data
#'
#' @return Matrix of responses, dim 1 for each data point, dim 2 for each
#' category
random_response <- function(n_data, n_outcomes, prob_na = 0, na_encode = 0) {
  responses <- matrix(0, nrow = n_data, ncol = length(n_outcomes))
  for (i in seq_len(length(n_outcomes))) {
    responses[, i] <- sample(n_outcomes[i], n_data, replace = TRUE)
  }

  rand <- runif(n_data * length(n_outcomes))
  rand <- matrix(rand, nrow = n_data, ncol = length(n_outcomes))
  responses[rand < prob_na] <- na_encode

  return(responses)
}

#' Generate random features
#'
#' Generate random Normally distributed random features
#'
#' @param n_data Number of data points
#' @param n_features Number of features
#'
#' @return Matrix of features, dim 1 for each data point, dim 2 for each feature
random_features <- function(n_data, n_features) {
  features <- runif(n_data * n_features)
  features <- matrix(features, nrow = n_data, ncol = n_features)
  return(features)
}

#' Generate random outcome probabilities
#'
#' Generate random outcome probabilities in an unvectorized form
#'
#' @param n_outcomes Vector of integers, number of outcomes for each category
#' @param n_cluster Number of clusters fitted
#'
#' @return List of three items (vecprobs, numChoices, classes) where
#'   * vecprobs: vector of outcome probabilities, a flattened list of matrices
#'      * dim 0: for each outcome
#'      * dim 1: for each category
#'      * dim 2: for each cluster
#'      * in other words, imagine a nested loop, from outer to inner:
#'         * for each cluster, for each category, for each outcome
#'   * numChoices: integer vector, number of outcomes for each category
#'   * classes: integer, number of classes (or clusters)
random_unvectorized_probs <- function(n_outcomes, n_cluster) {
  probs <- list(
    vecprobs = random_vectorized_probs(n_outcomes, n_cluster),
    numChoices = n_outcomes, classes = n_cluster
  )
  probs <- poLCAParallel.unvectorize(probs)
  return(probs)
}

#' Generate random cluster probabilities
#'
#' @param n_data Number of data points
#' @param n_cluster Number of clusters fitted
#'
#' @return Matrix of cluster probabilities, dim 1 for each data point, dim 2 for
#' each cluster
random_cluster_probs <- function(n_data, n_cluster) {
  probs <- runif(n_data * n_cluster)
  probs <- matrix(probs, nrow = n_data, ncol = n_cluster)
  probs <- probs / matrix(
    rep(rowSums(probs), n_cluster),
    nrow = n_data, ncol = n_cluster
  )
  return(probs)
}


#' Test the cluster probabilities
#'
#' Test the prior and posterior probabilities. Tests the shape of the provided
#' matrix of probabilities, they are in [0, 1] and sum to 1
#'
#' @param probs Matrix of cluster probabilities, can be prior or posterior, dim
#' 1 for each data point,dim 2 for each cluster
#' @param n_data Number of data points
#' @param n_cluster Number of clusters fitted
test_cluster_probs <- function(probs, n_data, n_cluster) {
  expect_identical(nrow(probs), as.integer(n_data))
  expect_identical(ncol(probs), as.integer(n_cluster))
  expect_gte(min(probs), 0)
  expect_lte(max(probs), 1)
  prob_sum <- rowSums(probs)
  for (i in seq_len(length(prob_sum))) {
    expect_equal(prob_sum[i], 1)
  }
}

#' Test the outcome probabilities
#'
#' Test the outcome probabilities, they have the correct shape, are in [0, 1]
#' and sum to 1
#'
#' @param probs list of length n_category. For the ith entry, it contains a
#' matrix of outcome probabilities with dimensions n_cluster x n_outcomes[i]
#' @param n_outcomes Vector of integers, number of outcomes for each category
#' @param n_cluster Number of clusters fitted
test_outcome_probs <- function(probs, n_outcomes, n_cluster) {
  expect_identical(length(probs), as.integer(length(n_outcomes)))
  tol <- 1e-12

  i_category <- 1
  for (i in names(probs)) {
    probs_i <- probs[[i]]

    expect_identical(nrow(probs_i), as.integer(n_cluster))
    expect_identical(ncol(probs_i), as.integer(n_outcomes[i_category]))

    expect_gte(min(probs_i), 0)
    expect_lte(max(probs_i), 1 + tol)
    prob_sum <- rowSums(probs_i)
    for (j in seq_len(length(prob_sum))) {
      expect_equal(prob_sum[j][[1]], 1)
    }
    i_category <- i_category + 1
  }
}

#' Test the poLCA object's standard error
#'
#' Test the poLCA object's (or a list which mocks it) standard error for the
#' prior probabilities and outcome probabilities
#'
#' @param polca The poLCA object to test (or a list which mocks it)
#' @param n_outcomes Vector of integers, number of outcomes for each category
#' @param n_cluster Number of clusters fitted
test_standard_error <- function(polca, n_outcomes, n_cluster) {
  expect_identical("P.se" %in% names(polca), TRUE)
  expect_identical("probs.se" %in% names(polca), TRUE)

  expect_identical(length(polca$P.se), as.integer(n_cluster))
  expect_identical(all(polca$P.se >= 0), TRUE)

  expect_identical(length(polca$probs.se), as.integer(length(n_outcomes)))
  for (i in seq_len(length(n_outcomes))) {
    expect_identical(nrow(polca$probs.se[[i]]), as.integer(n_cluster))
    expect_identical(ncol(polca$probs.se[[i]]), as.integer(n_outcomes[i]))
    expect_identical(all(polca$probs.se[[i]] >= 0), TRUE)
  }
}

#' Test the poLCA object's regression coefficient standard error
#'
#' Test the poLCA object's (or a list which mocks it) regression coefficient
#' standard error
#'
#' @param polca The poLCA object to test (or a list which mocks it)
#' @param n_feature Number of features
#' @param n_cluster Number of clusters fitted
test_standard_coeff_error <- function(polca, n_feature, n_cluster) {
  n_parameter <- n_feature * (n_cluster - 1)
  expect_identical(ncol(polca$coeff.V), as.integer(n_parameter))
  expect_identical(nrow(polca$coeff.V), as.integer(n_parameter))
}

#' Test the poLCA object's goodness of fit
#'
#' Test the poLCA object's (or a list which mocks it) goodness of fit outputs
#' such as the table of expected and observed frequencies (predcell) and
#' statistics such as Gsq and Chisq
#'
#' @param polca The poLCA object to test (or a list which mocks it)
#' @param n_outcomes Vector of integers, number of outcomes for each category
test_polca_goodnessfit <- function(polca, n_outcomes) {
  expect_identical("predcell" %in% names(polca), TRUE)
  expect_identical("Gsq" %in% names(polca), TRUE)
  expect_identical("Chisq" %in% names(polca), TRUE)

  unique_responses <- as.matrix(polca$predcell)[, seq_len(length(n_outcomes))]
  expect_identical(all(unique_responses > 0), TRUE)
  for (i in seq_len(length(n_outcomes))) {
    expect_identical(all(unique_responses[, i] <= n_outcomes[i]), TRUE)
  }

  expect_identical(sum(as.integer(polca$predcell$observed)), polca$Nobs)

  expect_lte(sum(polca$expected), polca$Nobs)
}
