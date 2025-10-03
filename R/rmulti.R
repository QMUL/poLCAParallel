#' Random draws from a multinomial distributionintegers
#'
#' One random draw from a multinomial distribution or list of multinomial
#' distributions.
#'
#' @keywords methods
#'
#' @param p matrix of dimension `n` by `r` containing probabilities,
#' for each row, of drawing each of `r` outcomes. `p` may also be
#' entered as a vector, in which case `rmulti` treats it as a matrix of
#' dimension `n=1` by `r`. Each row of matrix `p` must sum to 1
#' or `rmulti` will not work properly.
#'
#' @returns A vector of length `n`. Each item represents one draw from the
#' multinomial distribution parameterized by the outcome probabilities in each
#' row of `p`.
#'
#' @examples
#' ##
#' ## One draw from a three-category multinomial distribution.
#' ##
#' p1 <- c(0.7,0.2,0.1)
#' rmulti(p1)
#'
#' ##
#' ## 10,000 draws from a three-category multinomial distribution.
#' ##
#' n <- 10000
#' p2 <- matrix(p1,nrow=n,ncol=length(p1),byrow=TRUE)
#' rmdraws <- rmulti(p2)
#' table(rmdraws)/n    # should be approximately 0.7, 0.2, 0.1
#'
#' ##
#' ## 10,000 draws from a mixture of three groups of a
#' ## four-category multinomial distribution.
#' ##
#' group.p <- matrix(c(0.5,0.3,0.2),nrow=n,ncol=3,byrow=TRUE)
#' group <- rmulti(group.p)
#' p3 <- t(matrix(NA,nrow=n,ncol=4))
#' p3[,group==1] <- c(0.7,0.1,0.1,0.1)
#' p3[,group==2] <- c(0.1,0.7,0.1,0.1)
#' p3[,group==3] <- c(0.1,0.1,0.1,0.7)
#' p3 <- t(p3)
#' rmdraws3 <- rmulti(p3)
#' table(group,rmdraws3)
#' table(group,rmdraws3)/rowSums(table(group,rmdraws3))
#'
#' @export
rmulti <-
function(p) {
    if (is.vector(p)) p <- t(p)
    n <- nrow(p)
    p <- matrix(p[,-ncol(p)],nrow=n)
    thresh <- matrix(apply(p,1,cumsum),nrow=n,ncol=ncol(p),byrow=TRUE)
    vals <- matrix(stats::runif(n),ncol=ncol(thresh),nrow=n)
    return(rowSums(vals>thresh)+1)
}
