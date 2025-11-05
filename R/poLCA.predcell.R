#' Predicted cell percentages in a latent class model
#'
#' Calculates the predicted cell percentages from a latent class model, for
#' specified values of the manifest variables.
#'
#' The parameters estimated by a latent class model can be used to produce a
#' density estimate of the underlying probability mass function across the cells
#' in the multi-way table of manifest variables.  This function calculates cell
#' percentages for that density estimate, corresponding to selected sets of
#' responses on the manifest variables, `y`.
#'
#' @keywords methods
#' @seealso `poLCA`
#'
#' @param lc A model object estimated using the `poLCA` function.
#' @param y A vector or matrix containing series of responses on the manifest
#' variables in `lc`.
#' @returns A vector containing cell percentages corresponding to the specified
#' sets of responses `y`, based on the estimated latent class model `lc`.
#'
#' @examples
#' data(carcinoma)
#' f <- cbind(A, B, C, D, E, F, G) ~ 1
#' lca3 <- poLCA(f, carcinoma, nclass = 3) # log-likelihood: -293.705
#'
#' # Only 20 out of 32 possible response patterns are observed
#' lca3$predcell
#'
#' # Produce cell probabilities for one sequence of responses
#' poLCA.predcell(lc = lca3, y = c(1, 1, 1, 1, 1, 1, 1))
#'
#' # Estimated probabilities for a cell with zero observations
#' poLCA.predcell(lc = lca3, y = c(1, 1, 1, 1, 1, 1, 2))
#'
#' # Cell probabilities for both cells at once; y entered as a matrix
#' poLCA.predcell(lc = lca3, y = rbind(
#'   c(1, 1, 1, 1, 1, 1, 1),
#'   c(1, 1, 1, 1, 1, 1, 2)
#' ))
#'
#' @export
poLCA.predcell <- function(lc, y) {
  K.j <- sapply(lc$probs, ncol)
  trap <- FALSE
  if (is.vector(y) | any(dim(y) == 1)) {
    if ((length(y) != length(K.j)) | (any(y[1:length(K.j)] > K.j)) |
      (any(y <= 0)) | (any(y != round(y)))) {
      cat("Error: invalid vector (y) of manifest variable values. \n")
      trap <- TRUE
    } else {
      y <- matrix(y, nrow = 1)
    }
  } else {
    if ((ncol(y) != length(K.j)) | (any(apply(y, 2, max) > K.j)) |
      (any(y <= 0)) | (any(y != round(y)))) {
      cat("Error: invalid matrix (y) of manifest variable values. \n")
      trap <- TRUE
    }
  }
  if (trap) {
    invisible(NULL)
  } else {
    ret <- likelihood(poLCAParallel.vectorize(lc$probs), y) %*% lc$P
    return(ret)
  }
}
