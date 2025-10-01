test_reproduce_posterior <- function(n_data, n_outcomes, n_cluster, n_rep,
                                     na_rm, n_thread, maxiter, tol, prob_na) {
  responses <- as.data.frame(random_response(n_data, n_outcomes, prob_na, NaN))
  formula <- formula(
    paste0("cbind(", paste(colnames(responses), collapse = ","), ")~1")
  )
  polca <- poLCAParallel::poLCA(formula, responses, n_cluster,
    maxiter = maxiter, tol = tol, na.rm = na_rm, nrep = n_rep,
    verbose = FALSE, n.thread = n_thread
  )

  results_og <- poLCA::poLCA.posterior(polca, polca$y)
  results_current <- poLCAParallel::poLCA.posterior(polca, polca$y)

  expect_equal(results_og, results_current)
}


test_that("non-regression-full-data", {
  # test using na_rm = TRUE and FALSE
  # with no missing data, they both should work in the same way
  set.seed(-1012646258)
  expect_no_error(test_reproduce_posterior(
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
})
