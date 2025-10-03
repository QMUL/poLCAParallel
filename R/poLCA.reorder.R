#' Reorder latent classes in poLC
#'
#' A helper function to simplify the reordering of latent classes returned by
#' `poLCA`
#'
#' Because the latent classes outputted by `poLCA` are unordered categories, the
#' numerical order of the classes is arbitrary, and is determined solely by the
#' initial values of the EM algorithm. If `probs.start` is set to `NULL` (the
#' default) when calling `poLCA`, then the function generates the starting
#' values randomly in each run, typically rearranging the latent class labels.
#' The `poLCA.reorder` function is a convenient way to manually adjust the order
#' of the latent classes, by changing the order of the `probs.start`. Refitting
#' the latent class model using these reordered start values will produce a
#' model having the desired category labels.
#'
#' @keywords methods
#' @seealso `poLCA`
#'
#' @param probs a list of class-conditional response probabilities previously
#' used as start values to estimate a particular latent class model using
#' `poLCA`.
#' @param o.new a vector of length equal to the number of latent classes in
#' `probs`, giving the desired reordering of the latent classes.
#'
#' @returns A list of matrices containing the rearranged (by row)
#'     class-conditional response probabilities.
#'
#' @examples
#' ##
#' ## Using the "cheating" sample data set, make the larger
#' ## non-cheater class the first ("reference") class in a
#' ## latent class regression model.  The coefficient on GPA
#' ## now maintains a consistent interpretation.
#' ##
#' data(cheating)
#' f2 <- cbind(LIEEXAM, LIEPAPER, FRAUD, COPYEXAM) ~ GPA
#' lc.ch <- poLCA(f2, cheating, nclass = 2, verbose = FALSE)
#' probs.start.new <- poLCA.reorder(lc.ch$probs.start,
#'                                  order(lc.ch$P, decreasing = TRUE))
#' lc.ch <- poLCA(f2, cheating, nclass = 2, probs.start = probs.start.new)
#'
#' @export
poLCA.reorder <-
function(probs,o.new) {
    J <- length(probs)
    for (j in 1:J) probs[[j]] <- probs[[j]][o.new,]
    return(probs)
}
