#' Posterior probabilities from a latent class model
#'
#' Calculates the posterior probability that cases belong to each latent class.
#'
#' From the parameters estimated by the latent class model, this function
#' calculates the "posterior" probability that a specified case -- characterized
#' by values of the manifest variables `y`, and, if a latent class regression
#' model, concomitant variables `x` -- "belongs to" each latent class in `lc`.
#' For observed cases, this information is also contained in the `lc` model
#' object as `lc$posterior`. The added benefit of this function is that it can
#' calculate posterior class membership probabilities for arbitrary values of
#' `x` and `y`, whether or observed or not.
#'
#' @keywords methods
#' @seealso `poLCA`
#'
#' @param lc A model object estimated using the `poLCA` function.
#' @param y A vector or matrix containing series of responses on the manifest
#'   variables in `lc`.
#' @param x An optional vector or matrix of covariate values, if `lc` was
#'   specified as a latent class regression model.
#' @returns A matrix containing posterior probabilities corresponding to the
#'   specified sets of responses `y`, based on the estimated latent class model
#'   `lc`. For each row (one case), the first column gives the posterior
#'   probability of being in class 1, the second column gives the posterior
#'   probability of being in class 2, and so forth. Across rows, these
#'   probabilities sum to one.
#'
#' @examples
#' data(election)
#'
#' ## Basic latent class model with three classes
#' f1 <- cbind(
#'   MORALG, CARESG, KNOWG, LEADG, DISHONG, INTELG,
#'   MORALB, CARESB, KNOWB, LEADB, DISHONB, INTELB
#' ) ~ 1
#' lc1 <- poLCA(f1, election, nclass = 3) # log-likelihood: -16714.66
#'
#' # The first observed case
#' lc1$y[1, ]
#' lc1$posterior[1, ]
#' poLCA.posterior(lc = lc1, y = as.numeric(lc1$y[1, ]))
#'
#' # A hypothetical case
#' poLCA.posterior(lc = lc1, y = rep(2, 12))
#'
#' # Entering y as a matrix
#' lc1$posterior[1:10, ]
#' poLCA.posterior(lc = lc1, y = mapply(as.numeric, lc1$y[1:10, ]))
#'
#'
#' ## Latent class regression model with three classes
#' f2 <- cbind(
#'   MORALG, CARESG, KNOWG, LEADG, DISHONG, INTELG, MORALB, CARESB,
#'   KNOWB, LEADB, DISHONB, INTELB
#' ) ~ AGE + EDUC + GENDER
#' lc2 <- poLCA(f2, election, nclass = 3) # log-likelihood: -16598.38
#'
#' # Posteriors for case number 97 (poorly classified)
#' lc2$y[97, ]
#' lc2$x[97, ]
#' lc2$posterior[97, ]
#' poLCA.posterior(lc = lc2, y = as.numeric(lc2$y[97, ]), x = c(41, 6, 1))
#'
#' # If x is not specified, the posterior is calculated using the population
#' # average
#' poLCA.posterior(lc = lc2, y = as.numeric(lc2$y[97, ]))
#'
#' # Entering y and x as matrices
#' round(lc2$posterior[95:100, ], 2)
#' round(poLCA.posterior(
#'   lc = lc2, y = mapply(as.numeric, lc2$y[95:100, ]),
#'   x = as.matrix(lc2$x[95:100, -1])
#' ), 2)
#'
#' @export
poLCA.posterior <- function(lc, y, x = NULL) {
  if (is.vector(y) | any(dim(y) == 1)) {
    y <- matrix(y, nrow = 1)
  }
  y[is.na(y)] <- 0

  if (!is.null(x)) {
    if (is.vector(x)) x <- matrix(x, nrow = 1)
    x <- cbind(1, x)
  }

  if ((ncol(lc$x) > 1) & (!is.null(x))) {
    prior <- poLCA.updatePrior(lc$coeff, x, length(lc$P))
  } else {
    prior <- matrix(lc$P, nrow = nrow(y), ncol = length(lc$P), byrow = T)
  }
  ret <- poLCA.postClass.C(prior, poLCA.vectorize(lc$probs), y)
  return(ret)
}
