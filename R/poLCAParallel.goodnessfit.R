#' Add predcell, Gsq and Chisq to a fitted poLCA object
#'
#' Finds unique responses and put them in a dataframe along with the observed
#' and expected frequencies. Also calculates the log likelihood ratio and chi
#' squared statistics.
#'
#' Wrapper function around the C++ function `GoodnessFitRcpp()`.
#'
#' @param results A model object estimated using the `poLCA` function
#'
#' @return The model object with three added items:
#' * `predcell`: dataframe of unique responses with their observed and expected
#'   frequencies
#' * `Gsq`: log likelihood ratio
#' * `Chisq`: chi squared statistic
#' @export
poLCAParallel.goodnessfit <- function(results) {
  y <- results$y
  formula <- formula(
    paste0("cbind(", paste(colnames(y), collapse = ","), ")~1")
  )
  mframe <- stats::model.frame(formula, y, na.action = NULL)
  y <- stats::model.response(mframe)
  y[is.na(y)] <- 0
  prob_vec <- poLCAParallel.vectorize(results$probs)
  goodness_fit_results <- GoodnessFitRcpp(
    t(y),
    results$P,
    prob_vec$vecprobs,
    results$N,
    prob_vec$numChoices,
    prob_vec$classes
  )

  results$predcell <- data.frame(goodness_fit_results[[1]])
  colnames(results$predcell) <- c(colnames(y), "observed", "expected")
  results$Gsq <- goodness_fit_results[[2]]
  results$Chisq <- goodness_fit_results[[3]]
  return(results)
}
