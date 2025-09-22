#' Test the contents of a poLCA object
#'
#' Test the Rcpp outputted contents of a poLCA object, this tests the prior
#' probabilities, posterior probabilities, outcome probabilities, log
#' likelihood, number of iterations done, initial probabilities which achieved
#' the maximum log likelihood and more...
#'
#' Provide the poLCA object and parameters which are used to test the object
#'
#' @param polca The poLCA object to test
#' @param n_data Number of data points
#' @param n_outcomes Vector of integers, number of outcomes for each category
#' @param n_cluster Number of clusters fitted
#' @param n_rep Number of repetitions used
#' @param na_rm Logical, if to remove NA responses
#' @param maxiter Number of iterations used in the EM algorithm
test_polca_em_algorithm <- function(polca, n_data, n_outcomes, n_cluster, n_rep,
                                    na_rm, maxiter) {
  # if remove NA responses, use Nobs, number of fully observed data
  if (na_rm) {
    n_data <- polca$Nobs
  }
  # test the probabilities
  test_cluster_probs(polca$prior, n_data, n_cluster)
  test_cluster_probs(polca$posterior, n_data, n_cluster)
  test_outcome_probs(polca$probs, n_outcomes, n_cluster)
  test_outcome_probs(polca$probs.start, n_outcomes, n_cluster)

  # test the log likelihoods
  expect_identical(length(polca$attempts), as.integer(n_rep))
  expect_lte(polca$llik, 0)
  for (ln_l_i in polca$attempts) {
    expect_lte(ln_l_i, 0)
  }
  expect_equal(max(polca$attempts), polca$llik)

  # test the number of iterations
  expect_gte(polca$numiter, 0)
  expect_lte(polca$numiter, maxiter)
  expect_identical(polca$maxiter, maxiter)

  expect_equal(is.logical(polca$eflag), TRUE)
}

#' Test the other contents of a poLCA object
#'
#' Test thecpp outputted contents of a poLCA object not tested in
#' test_polca_em_algorithm(). It tests the R outputs (not Rcpp) created on
#' poLCA() such as the features, responses, number of data points and time
#' taken
#'
#' Provide the poLCA object and parameters which are used to test the object
#'
#' @param polca The poLCA object to test
#' @param n_data Number of data points
#' @param n_features Number of features
#' @param n_outcomes Vector of integers, number of outcomes for each category
#' @param n_cluster Number of clusters fitted
#' @param na_rm Logical, if to remove NA responses
test_polca_other <- function(polca, n_data, n_features, n_outcomes, n_cluster,
                             na_rm) {
  # if remove NA responses, use Nobs, number of fully observed data
  if (na_rm) {
    n_data <- polca$Nobs
  }
  expect_identical(polca$N, as.integer(n_data))

  # test design matrix of features
  expect_identical(nrow(polca$x), as.integer(n_data))
  expect_identical(ncol(polca$x), as.integer(n_features + 1))
  expect_identical(all(polca$x[, 1] == 1), TRUE)

  # test design matrix of responses
  expect_identical(nrow(polca$y), as.integer(n_data))
  expect_identical(ncol(polca$y), as.integer(length(n_outcomes)))
  if (na_rm) {
    expect_identical(all(polca$y >= 1), TRUE)
    expect_identical(polca$Nobs, polca$N)
  } else {
    response_mat <- as.matrix(polca$y)
    expect_identical(all(response_mat[!is.na(response_mat)] >= 1), TRUE)
    expect_identical(
      polca$Nobs,
      as.integer(sum(rowSums(is.na(response_mat)) == 0))
    )
  }

  inherits(polca$time, "difftime")
}

#' Test using poLCA() for the non-regression problem
#'
#' Test using poLCA() for the non-regression problem. Random responses are
#' generated and then passed to poLCA() to be fitted onto. The results are then
#' tested
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
test_non_regression <- function(n_data, n_outcomes, n_cluster, n_rep, na_rm,
                                n_thread, maxiter, tol, prob_na) {
  responses <- as.data.frame(random_response(n_data, n_outcomes, prob_na, NaN))
  formula <- formula(
    paste0("cbind(", paste(colnames(responses), collapse = ","), ")~1")
  )
  polca <- poLCAParallel::poLCA(formula, responses, n_cluster,
    maxiter = maxiter, tol = tol, na.rm = na_rm, nrep = n_rep,
    verbose = FALSE, n.thread = n_thread
  )
  test_polca_em_algorithm(
    polca, n_data, n_outcomes, n_cluster, n_rep, na_rm, maxiter
  )
  test_polca_other(polca, n_data, 0, n_outcomes, n_cluster, na_rm)
  test_polca_goodnessfit(polca, n_outcomes)
  test_standard_error(polca, n_outcomes, n_cluster)
}

#' Test using poLCA() for the regression problem
#'
#' Test using poLCA() for the regression problem. Random responses and features
#' are generated and then passed to poLCA() to be fitted onto. The results are
#' then tested
#'
#' @param n_data Number of data points
#' @param n_features Number of features
#' @param n_outcomes Vector of integers, number of outcomes for each category
#' @param n_cluster Number of clusters fitted
#' @param n_rep Number of different initial values to try
#' @param na_rm Logical, if to remove NA responses
#' @param n_thread Number of threads to use
#' @param maxiter Number of iterations used in the EM algorithm
#' @param tol Tolerance used in the EM algorithm
#' @param prob_na Probability of missing data
test_regression <- function(n_data, n_features, n_outcomes, n_cluster, n_rep,
                            na_rm, n_thread, maxiter, tol, prob_na) {
  # random features
  features <- as.data.frame(matrix(rnorm(n_data * n_features),
    nrow = n_data, ncol = n_features
  ))
  # column names to have prefix U, this ensures the column names are different
  # from responses
  colnames(features) <- paste0(rep("U", n_features), paste(seq_len(n_features)))

  # random responses
  responses <- as.data.frame(random_response(n_data, n_outcomes, prob_na, NaN))

  data <- cbind(responses, features)

  formula <- formula(
    paste0(
      "cbind(", paste(colnames(responses), collapse = ","), ")~",
      paste0(colnames(features), collapse = "+"),
      ""
    )
  )
  polca <- poLCAParallel::poLCA(formula, data, n_cluster,
    maxiter = maxiter, tol = tol, na.rm = na_rm, nrep = n_rep,
    verbose = FALSE, n.thread = n_thread
  )

  test_polca_em_algorithm(
    polca, n_data, n_outcomes, n_cluster, n_rep, na_rm, maxiter
  )
  test_polca_other(polca, n_data, n_features, n_outcomes, n_cluster, na_rm)
  test_polca_goodnessfit(polca, n_outcomes)
  test_standard_error(polca, n_outcomes, n_cluster)

  # test coefficients
  # one extra feature as poLCA adds an intercept term
  # one less cluster as only need (n_cluster - 1) probabilities to work out the
  # remaining one
  expect_identical(nrow(polca$coeff), as.integer(n_features + 1))
  expect_identical(ncol(polca$coeff), as.integer(n_cluster - 1))
}

#' Test if results are the same as original poLCA code
#'
#' Test if results are the same, or at least similar, as the original poLCA code
#' for the non-regression problem. Generate data and pass it to poLCA::poLCA()
#' and poLCAParallel::poLCA() and compare results
#'
#' The EM algorithm does depend on the initial values and how many different
#' initials were tried. A failed test could be fixed by either using a high
#' repetition count or ensure the initial values are the same
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
#' @param seed Seed to generate random data and seed poLCA
test_reproduce_non_regression <- function(n_data, n_outcomes, n_cluster, n_rep,
                                          na_rm, n_thread, maxiter, tol,
                                          prob_na, seed) {
  equal_tol <- 1e1 * sqrt(.Machine$double.eps)

  responses <- as.data.frame(random_response(n_data, n_outcomes, prob_na, NaN))
  formula <- formula(
    paste0("cbind(", paste(colnames(responses), collapse = ","), ")~1")
  )

  set.seed(seed)
  polca_og <- poLCA::poLCA(formula, responses, n_cluster,
    maxiter = maxiter, tol = tol, na.rm = na_rm, nrep = n_rep,
    verbose = FALSE
  )

  set.seed(seed)
  polca <- poLCAParallel::poLCA(formula, responses, n_cluster,
    maxiter = maxiter, tol = tol, na.rm = na_rm, nrep = n_rep,
    verbose = FALSE
  )

  # test if all attributes in the og code is in our code
  for (attribute_i in names(polca_og)) {
    expect_identical(attribute_i %in% names(polca), TRUE)
  }

  # test if results are the same
  expect_equal(polca$llik, polca_og$llik)
  expect_equal(polca$aic, polca_og$aic)
  expect_equal(polca$bic, polca_og$bic)
  expect_equal(polca$Nobs, polca_og$Nobs)

  expect_equal(polca$Chisq, polca_og$Chisq, tolerance = equal_tol)
  expect_equal(polca$Gsq, polca_og$Gsq, tolerance = equal_tol)

  # in predcell, the og code rounds the expected frequency
  pred_cell_rounded <- polca$predcell
  pred_cell_rounded$expected <- round(polca$predcell$expected, 3)
  expect_identical(all.equal(pred_cell_rounded, polca_og$predcell), TRUE)

  expect_identical(all.equal(polca$y, polca_og$y), TRUE)
  expect_identical(all.equal(polca$x, polca_og$x), TRUE)
}

test_that("non-regression-full-data", {
  # test using na_rm = TRUE and FALSE
  # with no missing data, they both should work in the same way
  set.seed(-1012646258)
  expect_no_error(test_non_regression(
    100,
    c(2, 3, 5, 2, 2),
    3,
    4,
    TRUE,
    4,
    1000,
    1e-10,
    0
  ))

  set.seed(-2057561765)
  expect_no_error(test_non_regression(
    100,
    c(2, 3, 5, 2, 2),
    3,
    4,
    FALSE,
    4,
    1000,
    1e-10,
    0
  ))
})

test_that("non-regression-missing-data", {
  # test using na_rm = TRUE and FALSE
  # with missing data, both will produce different results
  set.seed(-1554950958)
  expect_no_error(test_non_regression(
    100,
    c(2, 3, 5, 2, 2),
    3,
    4,
    TRUE,
    4,
    1000,
    1e-10,
    0.1
  ))

  set.seed(984792451)
  expect_no_error(test_non_regression(
    100,
    c(2, 3, 5, 2, 2),
    3,
    4,
    FALSE,
    4,
    1000,
    1e-10,
    0.1
  ))
})

test_that("regression-full-data", {
  set.seed(-590845051)
  expect_no_error(test_regression(
    100,
    4,
    c(2, 3, 5, 2, 2),
    3,
    4,
    TRUE,
    4,
    1000,
    1e-10,
    0
  ))
  set.seed(1785517768)
  expect_no_error(test_regression(
    100,
    4,
    c(2, 3, 5, 2, 2),
    3,
    4,
    FALSE,
    4,
    1000,
    1e-10,
    0
  ))
})


test_that("regression-missing-data", {
  set.seed(-85141069)
  expect_no_error(test_regression(
    100,
    4,
    c(2, 3, 5, 2, 2),
    3,
    4,
    TRUE,
    4,
    1000,
    1e-10,
    0.1
  ))

  set.seed(-2070313423)
  expect_no_error(test_regression(
    100,
    4,
    c(2, 3, 5, 2, 2),
    3,
    4,
    FALSE,
    4,
    1000,
    1e-10,
    0.1
  ))
})


test_that("reproduce-non-regression-full-data", {
  expect_no_error(test_reproduce_non_regression(
    100,
    c(2, 3, 5, 2, 2),
    3,
    1,
    TRUE,
    4,
    1000,
    1e-10,
    0,
    -683307112
  ))

  expect_no_error(test_reproduce_non_regression(
    100,
    c(2, 3, 5, 2, 2),
    3,
    1,
    FALSE,
    4,
    1000,
    1e-10,
    0,
    -1855018758
  ))
})

test_that("reproduce-non-regression-missing-data", {
  expect_no_error(test_reproduce_non_regression(
    100,
    c(2, 3, 5, 2, 2),
    3,
    1,
    TRUE,
    4,
    1000,
    1e-10,
    0.1,
    -1391069936
  ))

  expect_no_error(test_reproduce_non_regression(
    100,
    c(2, 3, 5, 2, 2),
    3,
    1,
    FALSE,
    4,
    1000,
    1e-10,
    0.1,
    799350486
  ))
})
