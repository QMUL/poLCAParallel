#' Entropy of a fitted latent class model
#'
#' Calculates the entropy of a cross-classification table produced as a density
#' estimate using a latent class model.
#'
#' Entropy is a measure of dispersion (or concentration) in a probability mass
#' function. For multivariate categorical data it is calculated \deqn{H =
#' -\sum_c p_c log(p_c)} where \eqn{p_c} is the share of the probability in the
#' $c$th cell of the cross-classification table. A fitted latent class model
#' produces a smoothed density estimate of the underlying distribution of cell
#' percentages in the multi-way table of the manifest variables. This function
#' calculates the entropy of that estimated probability mass function.
#'
#' @keywords methods
#' @seealso `poLCA`
#'
#' @param lc A model object estimated using the `poLCA` function.
#' @returns A number taking a minumum value of 0 (representing complete
#' concentration of probability on one cell) and a maximum value equal to the
#' logarithm of the total number of cells in the fitted cross-classfication
#' table (representing complete dispersion, or equal probability for outcomes
#' across every cell).
#'
#' @examples
#' data(carcinoma)
#' f <- cbind(A, B, C, D, E, F, G) ~ 1
#' lca2 <- poLCA(f, carcinoma, nclass = 2)  # log-likelihood: -317.2568
#' lca3 <- poLCA(f, carcinoma, nclass = 3)  # log-likelihood: -293.705
#' # log-likelihood: -289.2858
#' lca4 <- poLCA(f, carcinoma, nclass = 4, nrep = 10, maxiter = 5000)
#'
#' # Maximum entropy (if all cases equally dispersed)
#' log(prod(sapply(lca2$probs, ncol)))
#'
#' # Sample entropy ("plug-in" estimator, or MLE)
#' p.hat <- lca2$predcell$observed/lca2$N
#' H.hat <- -sum(p.hat * log(p.hat))
#' H.hat   # 2.42
#'
#' # Entropy of fitted latent class models
#' poLCA.entropy(lca2)
#' poLCA.entropy(lca3)
#' poLCA.entropy(lca4)
#'
#' @export

poLCA.entropy <-
function(lc) {
    K.j <- sapply(lc$probs,ncol)
    fullcell <- expand.grid(lapply(K.j,seq,from=1))
    P.c <- poLCA.predcell(lc,fullcell)
    return(-sum(P.c * log(P.c),na.rm=TRUE))
}
