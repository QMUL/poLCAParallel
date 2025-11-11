# Number of samples of the condition to test
n_sample_conditions <- 10
# Probability of a variable being in the condition
prob_in_condition <- 0.5

#' Test the function poLCA.table() given a fitted model
#'
#' Test the function poLCA.table() given a fitted model (can be non-regression
#' or regression model).
#'
#' This test will test all possible one-way and two-way tables by cycling
#' through the column names. In addition, it will test a sample of conditions,
#' randomly selected, to pass to the function, one of which is empty.
#'
#' The test compares the results with the original poLCA.
#'
#' @param columns Vector of strings, names of the columns of the responses
#' @param n_outcomes Vector of integers, number of outcomes for each category
#' @param lc A model object estimated using the `poLCA` function (or a list
#'   which mocks it)
test_table_given_model <- function(columns, n_outcomes, lc) {
  # for every pair category
  for (i_category in seq_len(length(n_outcomes))) {
    for (j_category in seq_len(length(n_outcomes))) {
      if (i_category == j_category) {
        # one way
        formula_ <- formula(paste0(columns[i_category], "~1"))
      } else {
        # two way
        formula_ <- formula(
          paste0(columns[i_category], "~", columns[j_category])
        )
      }

      for (i in seq_len(n_sample_conditions)) {
        condition <- list()
        # for the first iteration, the condition is empty
        if (i != 0) {
          # randomly sample a condition
          for (k_category in seq_len(length(n_outcomes))) {
            if (k_category != i_category && k_category != j_category) {
              if (runif(1) < prob_in_condition) {
                condition[[columns[k_category]]] <-
                  sample(seq_len(n_outcomes[k_category]), 1)
              }
            }
          }
        }

        # test function here
        table_polca <- poLCA::poLCA.table(formula_, condition, lc)
        table_polcaparallel <- poLCAParallel::poLCA.table(
          formula_, condition, lc
        )
        expect_equal(table_polcaparallel, table_polca)
      }
    }
  }
}

#' Test the function poLCA.table() for the non-regression problem
#'
#' Test the function poLCA.table() for the non-regression problem. The model is
#' fitted onto simulated data and then passed to the function. The test compares
#' the results with the original poLCA code
#'
#' See test_table_given_model() for further details
#'
#' @param n_data Number of data points
#' @param n_outcomes Vector of integers, number of outcomes for each category
#' @param n_cluster Number of clusters fitted
#' @param n_rep Number of different initial values to try
#' @param na_rm Logical, if to remove NA responses
#' @param n_thread Number of threads to use
#' @param maxiter Number of iterations used in the EM algorithm
#' @param tol Tolerance used in the EM algorithm
#' @param prob_na_train Probability of missing data in the training data
#' @param n_data_test Number of data points in the unseen test data
#' @param prob_na_test Probability of missing data in the unseen test data
test_non_regress_table <- function(n_data, n_outcomes, n_cluster, n_rep,
                                   na_rm, n_thread, maxiter, tol,
                                   prob_na_train, n_data_test,
                                   prob_na_test) {
  responses <- as.data.frame(
    random_response(n_data, n_outcomes, prob_na_train, NaN)
  )
  formula_ <- get_non_regression_formula(responses)
  lc <- poLCAParallel::poLCA(formula_, responses, n_cluster,
    maxiter = maxiter, tol = tol, na.rm = na_rm, nrep = n_rep,
    verbose = FALSE, n.thread = n_thread
  )

  test_table_given_model(colnames(responses), n_outcomes, lc)
}

#' Test the function poLCA.table() for the non-regression problem
#'
#' Test the function poLCA.table() for the non-regression problem. The model is
#' fitted onto simulated data and then passed to the function. The test compares
#' the results with the original poLCA code
#'
#' See test_table_given_model() for further details
#'
#' @param n_data Number of data points
#' @param n_feature Number of features
#' @param n_outcomes Vector of integers, number of outcomes for each category
#' @param n_cluster Number of clusters fitted
#' @param n_rep Number of different initial values to try
#' @param na_rm Logical, if to remove NA responses
#' @param n_thread Number of threads to use
#' @param maxiter Number of iterations used in the EM algorithm
#' @param tol Tolerance used in the EM algorithm
#' @param prob_na_train Probability of missing data in the training data
#' @param n_data_test Number of data points in the unseen test data
#' @param prob_na_test Probability of missing data in the unseen test data
test_regress_table <- function(n_data, n_feature, n_outcomes, n_cluster, n_rep,
                               na_rm, n_thread, maxiter, tol,
                               prob_na_train, n_data_test,
                               prob_na_test) {
  features <- random_features(n_data, n_feature)
  responses <- random_response(n_data, n_outcomes, prob_na_train, NaN)
  formula_ <- get_regression_formula(responses, features)
  data <- cbind(responses, features)

  model <- poLCAParallel::poLCA(formula_, data, n_cluster,
    maxiter = maxiter, tol = tol, na.rm = na_rm, nrep = n_rep,
    verbose = FALSE, n.thread = n_thread
  )

  test_table_given_model(colnames(responses), n_outcomes, model)
}


test_that("non-regression-full-data", {
  # test using na_rm = TRUE and FALSE
  set.seed(-507817496)
  expect_no_error(test_non_regress_table(
    100,
    c(2, 3, 5, 2, 2),
    3,
    4,
    TRUE,
    4,
    1000,
    1e-10,
    0,
    50,
    0.01
  ))

  set.seed(-2093133234)
  expect_no_error(test_non_regress_table(
    100,
    c(2, 3, 5, 2, 2),
    3,
    4,
    FALSE,
    4,
    1000,
    1e-10,
    0,
    50,
    0.01
  ))
})

test_that("non-regression-missing-data", {
  # test using na_rm = TRUE and FALSE
  set.seed(1354513976)
  expect_no_error(test_non_regress_table(
    100,
    c(2, 3, 5, 2, 2),
    3,
    4,
    TRUE,
    4,
    1000,
    1e-10,
    0.1,
    50,
    0.01
  ))

  set.seed(-647551612)
  expect_no_error(test_non_regress_table(
    100,
    c(2, 3, 5, 2, 2),
    3,
    4,
    FALSE,
    4,
    1000,
    1e-10,
    0.1,
    50,
    0.01
  ))
})


test_that("regression-full-data", {
  # test using na_rm = TRUE and FALSE
  set.seed(1449910432)
  expect_no_error(test_regress_table(
    100,
    4,
    c(2, 3, 5, 2, 2),
    3,
    4,
    TRUE,
    4,
    1000,
    1e-10,
    0,
    50,
    0.01
  ))

  set.seed(-346397158)
  expect_no_error(test_regress_table(
    100,
    4,
    c(2, 3, 5, 2, 2),
    3,
    4,
    FALSE,
    4,
    1000,
    1e-10,
    0,
    50,
    0.01
  ))
})

test_that("regression-missing-data", {
  # test using na_rm = TRUE and FALSE
  set.seed(-749216122)
  expect_no_error(test_regress_table(
    100,
    4,
    c(2, 3, 5, 2, 2),
    3,
    4,
    TRUE,
    4,
    1000,
    1e-10,
    0.1,
    50,
    0.01
  ))

  set.seed(-1284213522)
  expect_no_error(test_regress_table(
    100,
    4,
    c(2, 3, 5, 2, 2),
    3,
    4,
    FALSE,
    4,
    1000,
    1e-10,
    0.1,
    50,
    0.01
  ))
})
