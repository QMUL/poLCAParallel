#' Create simulated cross-classification data
#'
#' Uses the latent class model's assumed data-generating process to create a
#' simulated dataset that can be used to test the properties of the poLCA latent
#' class and latent class regression estimator.
#'
#' Note that entering `probs` overrides `nclass`, `ndv`, and `nresp`. It also
#' overrides `P` if the length of the `P` vector is not equal to the length of
#' the `probs` list.  Likewise, if `probs=NULL`, then `length(nresp)` overrides
#' `ndv` and `length(P)` overrides `nclass`.  Setting `niv>1` causes any
#' user-entered value of `P` to be disregarded.
#'
#' @keywords methods
#' @seealso `poLCA`
#'
#' @param N number of observations.
#' @param probs a list of matrices of dimension  `nclass ` by `nresp ` with each
#'   matrix corresponding to one manifest variable, and each row containing the
#'   class-conditional outcome probabilities (which must sum to 1). If `probs`
#'   is `NULL` (default) then the outcome probabilities are generated randomly.
#' @param nclass number of latent classes. If `probs` is specified, then
#'   `nclass` is set equal to the number of rows in each matrix in that list. If
#'   `P` is specified, then `nclass` is set equal to the length of that vector.
#'   If `b` is specified, then `nclass` is set equal to one greater than the
#'   number of columns in `b`. Otherwise, the default is two.
#' @param ndv number of manifest variables. If `probs` is specified, then `ndv`
#'   is set equal to the number of matrices in that list. If `nresp` is
#'   specified, then `ndv` is set equal to the length of that vector. Otherwise,
#'   the default is four.
#' @param nresp number of possible outcomes for each manifest variable. If
#'   `probs` is specified, then `ndv` is set equal to the number of columns in
#'   each matrix in that list. If both `probs` and  `nresp` are `NULL`
#'   (default), then the manifest variables are assigned a random number of
#'   outcomes between two and five.
#' @param x a matrix of concomicant variables with `N` rows and `niv` columns.
#'   If `x=NULL` (default), but  `niv>0`, then `niv` concomitant variables will
#'   be generated as mutually independent random draws from a standard normal
#'   distribution.
#' @param niv number of concomitant variables (covariates). Setting `niv=0`
#'   (default) creates a data set assuming no covariates. If `nclass=1` then
#'   `niv` is automatically set equal to 0. If both `x` and `niv` are entered,
#'   then the number of columns in `x` overrides the value of `niv`. The number
#'   of rows in `b`, less one, also overrides `niv`.
#' @param b when using covariates, an  `niv+1` by `nclass-1` matrix of
#'   (multinomial) logit coefficients. If `b` is `NULL` (default), then
#'   coefficients are generated as random integers between -2 and 2.
#' @param P a vector of mixing proportions (class population shares) of length
#'   `nclass`. `P` must sum to 1. Disregarded if `b` is specified or `niv>1`
#'   because then `P` is, in part, a function of the concomitant variables. If
#'   `P` is `NULL` (default), then the mixing proportions are generated
#'   randomly.
#' @param missval logical. If `TRUE` then a fraction `pctmiss` of the manifest
#'   variables are randomly dropped as missing values. Default is `FALSE`.
#' @param pctmiss percentage of values to be dropped as missing, if
#'   `missval=TRUE`. If `pctmiss` is `NULL` (default), then a value between 5
#'   and 40 percent is chosen randomly.
#' @returns A list containing the following
#'   - `dat`: a data frame containing the simulated variables. Variable names
#'     for manifest variables are Y1, Y2, etc. Variable names for concomitant
#'     variables are X1, X2, etc.
#'   - `probs`: a list of matrices of dimension `nclass` by `nresp` containing
#'     the class-conditional response probabilities.
#'   - `nresp`: a vector containing the number of possible outcomes for each
#'      manifest variable.
#'   - `b`: coefficients on covariates, if used.
#'   - `P`: mixing proportions corresponding to each latent class.
#'   - `pctmiss`: percent of observations missing.
#'   - `trueclass`: `N` by 1 vector containing the "true" class
#'     membership for each individual.
#'
#' @examples
#' # Create a sample data set with 3 classes and no covariates
#' # and run poLCA to recover the specified parameters.
#' probs <- list(
#'   matrix(c(0.6, 0.1, 0.3, 0.6, 0.3, 0.1, 0.3, 0.1, 0.6),
#'     ncol = 3, byrow = TRUE
#'   ), # conditional resp prob to Y1
#'   matrix(c(0.2, 0.8, 0.7, 0.3, 0.3, 0.7),
#'     ncol = 2, byrow = TRUE
#'   ), # conditional resp prob to Y2
#'   matrix(c(0.3, 0.6, 0.1, 0.1, 0.3, 0.6, 0.3, 0.6, 0.1),
#'     ncol = 3, byrow = TRUE
#'   ), # conditional resp prob to Y3
#'   matrix(c(0.1, 0.1, 0.5, 0.3, 0.5, 0.3, 0.1, 0.1, 0.3, 0.1, 0.1, 0.5),
#'     ncol = 4, byrow = TRUE
#'   ), # conditional resp prob to Y4
#'   matrix(c(0.1, 0.1, 0.8, 0.1, 0.8, 0.1, 0.8, 0.1, 0.1),
#'     ncol = 3, byrow = TRUE
#'   )
#' ) # conditional resp prob to Y5
#' simdat <- poLCA.simdata(N = 1000, probs, P = c(0.2, 0.3, 0.5))
#' f1 <- cbind(Y1, Y2, Y3, Y4, Y5) ~ 1
#' lc1 <- poLCA(f1, simdat$dat, nclass = 3)
#' table(lc1$predclass, simdat$trueclass)
#'
#' # Create a sample dataset with 2 classes and three covariates.
#' # Then compare predicted class memberships when the model is
#' # estimated "correctly" with covariates to when it is estimated
#' # "incorrectly" without covariates.
#' simdat2 <- poLCA.simdata(
#'   N = 1000, ndv = 7, niv = 3, nclass = 2,
#'   b = matrix(c(1, -2, 1, -1))
#' )
#' f2a <- cbind(Y1, Y2, Y3, Y4, Y5, Y6, Y7) ~ X1 + X2 + X3
#' lc2a <- poLCA(f2a, simdat2$dat, nclass = 2)
#' f2b <- cbind(Y1, Y2, Y3, Y4, Y5, Y6, Y7) ~ 1
#' lc2b <- poLCA(f2b, simdat2$dat, nclass = 2)
#' table(lc2a$predclass, lc2b$predclass)
#'
#' @export
poLCA.simdata <- function(N = 5000, probs = NULL, nclass = 2, ndv = 4,
                          nresp = NULL, x = NULL, niv = 0, b = NULL, P = NULL,
                          missval = FALSE, pctmiss = NULL) {
  if (is.null(probs)) {
    if (is.null(nresp)) {
      nresp <- ceiling(stats::runif(ndv, min = 1, max = 5))
    }
    if (!is.null(P)) {
      nclass <- length(P)
    }
    if (!is.null(b)) {
      nclass <- ncol(b) + 1
    }
    ndv <- length(nresp)
    probs <- list()
    for (i in 1:ndv) {
      probs[[i]] <- matrix(stats::runif(nclass * nresp[i]),
        nrow = nclass,
        ncol = nresp[i]
      )
      probs[[i]] <- probs[[i]] / rowSums(probs[[i]])
    }
  } else {
    ndv <- length(probs)
    nclass <- nrow(probs[[1]])
    nresp <- sapply(probs, ncol)
  }
  if (nclass == 1) {
    niv <- 0
    b <- NULL
    P <- 1
    group <- matrix(1, nrow = N, ncol = 1)
  } else {
    if (!is.null(x)) {
      niv <- ncol(x)
      if (nrow(x) != N) {
        cat(paste(
          "ALERT: number of rows of x does not equal N;",
          "new covariates will be generated randomly. \n \n"
        ))
        x <- NULL
      }
      if (ncol(x) != (nrow(b) - 1)) {
        cat(paste(
          "ALERT:",
          "number of columns of x does not conform to number of rows in b;",
          "new covariates will be generated randomly. \n \n"
        ))
        x <- NULL
      }
    }
    if (!is.null(b)) {
      niv <- nrow(b) - 1
    }
    if (niv > 0) {
      if (is.null(x)) {
        x <- matrix(stats::rnorm(N * niv), nrow = N, ncol = niv)
      }
      colnames(x) <- paste("X", c(1:niv), sep = "")
      if (is.null(b)) {
        b <- matrix(
          round(stats::runif(((nclass - 1) * (niv + 1)),
            min = -2, max = 2
          )),
          nrow = (niv + 1)
        )
      }
      prior <- poLCA.updatePrior(c(b), cbind(1, x), nclass)
    } else {
      if (nrow(probs[[1]]) != length(P)) {
        P <- stats::runif(nclass)
        P <- P / sum(P)
      }
      prior <- matrix(P, byrow = TRUE, nrow = N, ncol = nclass)
    }
    group <- rmulti(prior)
  }
  y <- rmulti(probs[[1]][group, ])
  for (j in 2:ndv) {
    y <- cbind(y, rmulti(probs[[j]][group, ]))
  }
  colnames(y) <- paste("Y", c(1:ndv), sep = "")
  if (niv > 0) {
    probs_vectorized <- poLCAParallel.vectorize(probs)

    posterior <- PosteriorRcpp(
      t(y), probs_vectorized$vecprobs, probs_vectorized$numChoices, prior,
      dim(y)[1], probs_vectorized$classes
    )

    P <- colMeans(posterior)
  }
  if (missval) {
    if (is.null(pctmiss)) pctmiss <- stats::runif(1, min = 0.05, max = 0.4)
    make.na <- cbind(
      ceiling(stats::runif(round(pctmiss * N * ndv),
        min = 0, max = N
      )),
      ceiling(stats::runif(round(pctmiss * N * ndv), min = 0, max = ndv))
    )
    y[make.na] <- NA
  }
  ret <- list()
  if (is.null(x)) {
    ret$dat <- data.frame(y)
  } else {
    ret$dat <- data.frame(y, x)
  }
  ret$trueclass <- group
  ret$probs <- probs
  ret$nresp <- nresp
  ret$b <- b
  ret$P <- P
  ret$pctmiss <- pctmiss
  return(ret)
}
