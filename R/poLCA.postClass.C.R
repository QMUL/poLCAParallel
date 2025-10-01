poLCA.postClass.C <- function(prior, vp, y) {
    posterior <- double(dim(y)[1] * vp$classes)
    postclass(
        t(prior),
        vp$vecprobs,
        t(y),
        length(vp$numChoices),
        dim(y)[1],
        vp$numChoices,
        vp$classes,
        posterior
    )
    posterior <- matrix(posterior, ncol = vp$classes, byrow = TRUE)
    return(posterior)
}
