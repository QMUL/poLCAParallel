#' Frequency tables of predicted cell counts from latent class analysis
#'
#' Calculates predicted cell frequencies based on an estimated latent class
#' model.
#'
#' This function outputs predicted cell counts for user-specified combinations
#' of the manifest variables, based on a latent class model estimated by the
#' `poLCA` function.  The `predcell` table outputted automatically by `poLCA`
#' also contains predicted cell frequencies, but only for cells containing at
#' least one observation.  In contrast, `poLCA.table` will calculate
#' predicted cell counts for all cells, including those with zero observations.
#'
#' @keywords methods
#' @seealso `poLCA`
#'
#' @param formula A formula expression of the form `variable ~ 1` for a one-way
#'   frequency distribution, or `row ~ column` for two way-tables.
#' @param condition A list containing the values of the manifest variables to
#'   hold fixed when creating the table specified by the `formula` argument.
#'   Setting this to an empty list, `condition=list()`, conditions on none of
#'   the other manifest variables, producing the marginal frequencies.
#' @param lc A model object previously estimated using the `poLCA` function.
#' @return A vector or table containing the specified frequency distribution.
#'
#' @examples
#' data(gss82)
#' f <- cbind(PURPOSE, ACCURACY, UNDERSTA, COOPERAT) ~ 1
#' gss.lc2 <- poLCA(f, gss82, nclass = 2)
#' gss.lc2$predcell
#'
#' poLCA.table(
#'   formula = COOPERAT ~ 1,
#'   condition = list(PURPOSE = 3, ACCURACY = 1, UNDERSTA = 2),
#'   lc = gss.lc2
#' )
#' poLCA.table(
#'   formula = COOPERAT ~ UNDERSTA,
#'   condition = list(PURPOSE = 3, ACCURACY = 1),
#'   lc = gss.lc2
#' )
#' poLCA.table(
#'   formula = COOPERAT ~ UNDERSTA,
#'   condition = list(),
#'   lc = gss.lc2
#' )
#'
#' @export
poLCA.table <- function(formula, condition = NULL, lc) {
  y <- lc$y
  mf <- as.data.frame(mapply(
    as.numeric,
    stats::model.frame(formula, y, na.action = NULL)
  ))
  ret <- NULL
  trap <- FALSE
  if (any(condition <= 0) |
    any(condition > apply(y[names(condition)], 2, max, na.rm = T))) {
    cat("Error: Some 'condition' values are not observed in data set. \n")
    trap <- TRUE
  } else if (any(table(c(names(condition), names(mf))) > 1)) {
    cat(paste(
      "Error: Variables can only be specified once in",
      "'formula' or 'condition'. \n"
    ))
    trap <- TRUE
  } else if (ncol(mf) > 2) {
    cat("Error: 'formula' must be of form 'y~1' or 'y~x'. \n")
    trap <- TRUE
  }
  if (!trap) {
    grp <- F
    sel <- list()
    for (j in 1:ncol(y)) {
      if (names(y)[j] %in% names(mf)) {
        sel[[j]] <- c(1:max(mf[, which(names(mf) == names(y)[j])], na.rm = T))
      } else {
        if (sum(names(condition) == names(y)[j]) == 0) {
          sel[[j]] <- c(1:max(as.numeric(y[, j]), na.rm = T))
          grp <- TRUE
        } else {
          sel[[j]] <- condition[[which(names(condition) == names(y)[j])]]
        }
      }
    }
    names(sel) <- names(y)
    yc <- expand.grid(sel)
    predcell <- lc$N / .Machine$double.xmax * (
      poLCA.ylik.C(poLCA.vectorize(lc$probs), yc) %*% lc$P)
    if (ncol(mf) > 1) {
      ord <- 1 +
        (which(names(mf)[1] == names(y)) > which(names(mf)[2] == names(y)))
      if (grp) {
        pc.col <- NULL
        for (i1 in 1:max(mf[, 3 - ord], na.rm = T)) {
          for (i2 in 1:max(mf[, ord], na.rm = T)) {
            pc.col <- c(
              pc.col,
              sum(predcell[yc[, which(names(y) %in% names(mf)[3 - ord])] ==
                i1 &
                yc[, which(names(y) %in% names(mf)[ord])] == i2])
            )
          }
        }
        predcell <- pc.col
      }
      nr <- apply(mf, 2, max, na.rm = T)[ord]
      nc <- apply(mf, 2, max, na.rm = T)[3 - ord]
      ret <- matrix(predcell, nrow = nr, ncol = nc)
      if (is.factor(y[, match(names(mf)[ord], names(y))])) {
        rownames(ret) <- levels(y[, match(names(mf)[ord], names(y))])
      } else {
        rownames(ret) <- paste(names(mf)[ord], c(1:max(mf[, ord], na.rm = T)))
      }
      if (is.factor(y[, match(names(mf)[3 - ord], names(y))])) {
        colnames(ret) <- levels(y[, match(names(mf)[3 - ord], names(y))])
      } else {
        colnames(ret) <- paste(
          names(mf)[3 - ord],
          c(1:max(mf[, 3 - ord], na.rm = T))
        )
      }
      if (ord == 2) {
        ret <- t(ret)
      }
    } else {
      if (grp) {
        pc.col <- NULL
        for (i in 1:max(mf, na.rm = T)) {
          pc.col <- c(
            pc.col,
            sum(predcell[yc[, match(names(mf), names(y))] == i])
          )
        }
        predcell <- pc.col
      }
      ret <- matrix(predcell, nrow = 1, ncol = max(mf, na.rm = T))
      rownames(ret) <- ""
      if (is.factor(y[, match(names(mf)[1], names(y))])) {
        colnames(ret) <- levels(y[, match(names(mf)[1], names(y))])
      } else {
        colnames(ret) <- paste(names(mf)[1], c(1:max(mf, na.rm = T)))
      }
    }
    return(ret)
  } else {
    invisible(NULL)
  }
}
