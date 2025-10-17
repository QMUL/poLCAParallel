#' Test the function poLCAParallel::poLCA.simdata()
#'
#' Test the function poLCAParallel::poLCA.simdata() against the original
#' poLCA::poLCA.simdata()
#'
#' @param seed to set the rng before each function call
#' @param args list of arguments to pass to the function
test_simdata <- function(seed, args) {
  set.seed(seed)
  data_polca <- do.call(poLCA::poLCA.simdata, args)
  set.seed(seed)
  data_polcaparallel <- do.call(poLCAParallel::poLCA.simdata, args)
  expect_equal(data_polcaparallel, data_polca)
}

test_that("default", {
  # default value
  expect_no_error(test_simdata(1277260137, list()))
})

test_that("non-regression-all-random", {
  # only pass the size (or partially) of the model, eg number of classes, number
  # of categories, ... etc
  # the parameters, such as outcome and prior probabilities, are randomly
  # generated inside the function
  # vary which parameters to provide or not

  expect_no_error(test_simdata(-966151570, list(nclass = 6)))
  expect_no_error(test_simdata(-1578000385, list(ndv = 7)))
  expect_no_error(test_simdata(1815930918, list(nclass = 10, ndv = 5)))
  expect_no_error(test_simdata(1859213588, list(nresp = c(2, 3, 5, 2, 2))))
  expect_no_error(test_simdata(
    1988508340,
    list(nclass = 8, nresp = c(2, 3, 5, 2, 2))
  ))
  expect_no_error(test_simdata(-1571392400, list(N = 1000, nclass = 6)))
  expect_no_error(test_simdata(-2096859141, list(N = 1000, ndv = 7)))
  expect_no_error(test_simdata(
    -613257287,
    list(N = 1000, nclass = 10, ndv = 5)
  ))
  expect_no_error(test_simdata(
    -1526726605,
    list(N = 1000, nresp = c(2, 3, 5, 2, 2))
  ))
  expect_no_error(test_simdata(
    191263031,
    list(N = 1000, nclass = 8, nresp = c(2, 3, 5, 2, 2))
  ))
})

test_that("non-regression-pass-probs", {
  # pass (or partially) the parameters of the model
  # if partially passed, either the default value is used or randomly generated
  # inside the function
  # vary which parameters to provide or not

  set.seed(-688058586)
  probs <- random_unvectorized_probs(c(2, 3, 5, 2, 2), 4)
  expect_no_error(test_simdata(-203219460, list(probs = probs)))

  set.seed(-1654954026)
  prior <- random_cluster_probs(1, 8)
  expect_no_error(test_simdata(-350403489, list(P = prior)))

  set.seed(-533270882)
  prior <- random_cluster_probs(1, 8)
  expect_no_error(test_simdata(450234962, list(P = prior, ndv = 6)))

  set.seed(-1220730757)
  prior <- random_cluster_probs(1, 8)
  expect_no_error(test_simdata(
    385657457,
    list(P = prior, nresp = c(2, 3, 5, 4, 2))
  ))

  set.seed(2146774050)
  probs <- random_unvectorized_probs(c(2, 3, 5, 2, 2), 4)
  prior <- random_cluster_probs(1, 4)
  expect_no_error(test_simdata(1018688540, list(probs = probs, P = prior)))

  set.seed(-2007353213)
  probs <- random_unvectorized_probs(c(2, 3, 5, 2, 7), 4)
  expect_no_error(test_simdata(
    -1775380840,
    list(N = 1000, probs = probs)
  ))

  set.seed(1734551279)
  prior <- random_cluster_probs(1, 8)
  expect_no_error(test_simdata(
    927544077,
    list(N = 1000, P = prior)
  ))

  set.seed(-1652727725)
  prior <- random_cluster_probs(1, 8)
  expect_no_error(test_simdata(1293516212, list(N = 1000, P = prior, ndv = 6)))

  set.seed(11881185)
  prior <- random_cluster_probs(1, 8)
  expect_no_error(test_simdata(
    -1985641227,
    list(N = 1000, P = prior, nresp = c(2, 2, 4, 6, 2))
  ))

  set.seed(-716957191)
  probs <- random_unvectorized_probs(c(2, 3, 5, 5, 2), 4)
  prior <- random_cluster_probs(1, 4)
  expect_no_error(test_simdata(
    -1608480737,
    list(N = 1000, probs = probs, P = prior)
  ))
})

test_that("regression-no-param", {
  # pass (or partially) regression model specification
  # the gradient or features isn't passed in this section
  # vary which parameters to provide or not

  expect_no_error(test_simdata(-1458523973, list(nclass = 4, ndv = 5, niv = 4)))
  expect_no_error(test_simdata(
    -976592540,
    list(nclass = 3, nresp = c(2, 5, 6, 3, 2), niv = 5)
  ))

  expect_no_error(test_simdata(-843267700, list(nclass = 4, ndv = 5, niv = 4)))

  set.seed(1767497224)
  probs <- random_unvectorized_probs(c(2, 3, 5, 2, 2), 4)
  expect_no_error(test_simdata(-203219460, list(probs = probs, niv = 6)))

  expect_no_error(test_simdata(
    -2140551814,
    list(N = 1000, nclass = 4, ndv = 5, niv = 4)
  ))
  expect_no_error(test_simdata(
    -1073753404,
    list(N = 1000, nclass = 3, nresp = c(2, 5, 6, 3, 2), niv = 5)
  ))

  expect_no_error(test_simdata(-2013770106, list(nclass = 4, ndv = 5, niv = 4)))

  set.seed(109649579)
  probs <- random_unvectorized_probs(c(2, 3, 5, 2, 2), 4)
  expect_no_error(test_simdata(
    2073157051,
    list(N = 1000, probs = probs, niv = 6)
  ))
})

test_that("regression-random-features", {
  # design matrix of features (aka x) is random within the function by not
  # providing it
  # vary which parameters to provide or not

  expect_no_error(test_simdata(-758321419, list(niv = 5)))
  expect_no_error(test_simdata(
    1880746837,
    list(nresp = c(3, 5, 4, 3, 2), niv = 5)
  ))

  set.seed(-2098347430)
  nclass <- 10
  gradient <- matrix(stats::rnorm((nclass - 1) * 10), ncol = nclass - 1)
  expect_no_error(test_simdata(
    1813336882,
    list(b = gradient)
  ))

  set.seed(-1639136259)
  nclass <- 5
  gradient <- matrix(stats::rnorm((nclass - 1) * 10), ncol = nclass - 1)
  expect_no_error(test_simdata(
    1813336882,
    list(nresp = c(2, 5, 6, 3, 2), b = gradient)
  ))

  set.seed(-405236032)
  nclass <- 5
  probs <- random_unvectorized_probs(c(2, 3, 5, 2, 2), nclass)
  gradient <- matrix(stats::rnorm((nclass - 1) * 10), ncol = nclass - 1)
  expect_no_error(test_simdata(
    -1535599873,
    list(probs = probs, b = gradient)
  ))
})

test_that("regression-provide-features", {
  # provide the design matrix of features (aka x)
  # vary which parameters to provide or not

  set.seed(-2009190883)
  n_data <- 1000
  nclass <- 4
  n_feature <- 5
  features <- as.matrix(random_features(n_data, n_feature))
  gradient <- matrix(stats::rnorm((nclass - 1) * (n_feature + 1)),
    nrow = n_feature + 1
  )
  expect_no_error(test_simdata(
    -2126326028,
    list(N = n_data, x = features, b = gradient)
  ))

  set.seed(-1811143544)
  n_data <- 1000
  nclass <- 4
  n_feature <- 5
  features <- as.matrix(random_features(n_data, n_feature))
  gradient <- matrix(stats::rnorm((nclass - 1) * (n_feature + 1)),
    nrow = n_feature + 1
  )
  expect_no_error(test_simdata(
    -680553055,
    list(N = n_data, x = features, b = gradient, niv = 7)
  ))

  set.seed(-703709404)
  n_data <- 1000
  nclass <- 4
  n_feature <- 5
  features <- as.matrix(random_features(n_data, n_feature))
  gradient <- matrix(stats::rnorm((nclass - 1) * (n_feature + 1)),
    nrow = n_feature + 1
  )
  expect_no_error(test_simdata(
    -680553055,
    list(N = n_data, x = features, b = gradient, nresp = c(3, 5, 9, 2))
  ))

  set.seed(-1160742358)
  n_data <- 1000
  nclass <- 4
  n_feature <- 5
  features <- as.matrix(random_features(n_data, n_feature))
  gradient <- matrix(stats::rnorm((nclass - 1) * (n_feature + 1)),
    nrow = n_feature + 1
  )
  probs <- random_unvectorized_probs(c(2, 3, 5, 2, 2), nclass)
  expect_no_error(test_simdata(
    703852772,
    list(N = n_data, probs = probs, x = features, b = gradient)
  ))
})

test_that("missing-data", {
  expect_no_error(test_simdata(1886413857, list(missval = TRUE)))
  expect_no_error(test_simdata(
    -1347152099,
    list(missval = TRUE, pctmiss = 0.2)
  ))
})
