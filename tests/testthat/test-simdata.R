#' Test the function `poLCAParallel::poLCA.simdata()`
#'
#' Test the function `poLCAParallel::poLCA.simdata()` against the original
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
  set.seed(1277260137)
  seeds <- sample.int(.Machine$integer.max, N_REPEAT)
  for (i in seq_len(N_REPEAT)) {
    expect_no_error(test_simdata(seeds[i], list()))
  }
})

test_that("non-regression-all-random", {
  # only pass the size (or partially) of the model, eg number of classes, number
  # of categories, ... etc
  # the parameters, such as outcome and prior probabilities, are randomly
  # generated inside the function
  # vary which parameters to provide or not

  set.seed(-1571392400)
  seeds <- sample.int(.Machine$integer.max, N_REPEAT)
  for (i in seq_len(N_REPEAT)) {
    expect_no_error(test_simdata(
      seeds[i],
      list(N = 1000, nclass = 6)
    ))
    expect_no_error(test_simdata(
      seeds[i],
      list(N = 1000, ndv = 7)
    ))
    expect_no_error(test_simdata(
      seeds[i],
      list(N = 1000, nclass = 10, ndv = 5)
    ))
    expect_no_error(test_simdata(
      seeds[i],
      list(N = 1000, nresp = c(2, 3, 5, 2, 2))
    ))
    expect_no_error(test_simdata(
      seeds[i],
      list(N = 1000, nclass = 8, nresp = c(2, 3, 5, 2, 2))
    ))
  }
})

test_that("non-regression-pass-probs", {
  # pass (or partially) the parameters of the model
  # if partially passed, either the default value is used or randomly generated
  # inside the function
  # vary which parameters to provide or not

  set.seed(-2007353213)
  seeds <- sample.int(.Machine$integer.max, N_REPEAT)
  for (i in seq_len(N_REPEAT)) {
    set.seed(seeds[i])
    probs <- random_unvectorized_probs(c(2, 3, 5, 2, 7), 4)
    expect_no_error(test_simdata(
      sample.int(.Machine$integer.max, 1),
      list(N = 1000, probs = probs)
    ))

    set.seed(seeds[i])
    prior <- random_cluster_probs(1, 8)
    expect_no_error(test_simdata(
      sample.int(.Machine$integer.max, 1),
      list(N = 1000, P = prior)
    ))

    set.seed(seeds[i])
    prior <- random_cluster_probs(1, 8)
    expect_no_error(test_simdata(
      sample.int(.Machine$integer.max, 1),
      list(N = 1000, P = prior, ndv = 6)
    ))

    set.seed(seeds[i])
    prior <- random_cluster_probs(1, 8)
    expect_no_error(test_simdata(
      sample.int(.Machine$integer.max, 1),
      list(N = 1000, P = prior, nresp = c(2, 2, 4, 6, 2))
    ))

    set.seed(seeds[i])
    probs <- random_unvectorized_probs(c(2, 3, 5, 5, 2), 4)
    prior <- random_cluster_probs(1, 4)
    expect_no_error(test_simdata(
      sample.int(.Machine$integer.max, 1),
      list(N = 1000, probs = probs, P = prior)
    ))
  }
})

test_that("regression-no-param", {
  # pass (or partially) regression model specification
  # the gradient or features isn't passed in this section
  # vary which parameters to provide or not

  set.seed(-2140551814)
  seeds <- sample.int(.Machine$integer.max, N_REPEAT)
  for (i in seq_len(N_REPEAT)) {
    expect_no_error(test_simdata(
      seeds[i],
      list(N = 1000, nclass = 4, ndv = 5, niv = 4)
    ))

    expect_no_error(test_simdata(
      seeds[i],
      list(N = 1000, nclass = 3, nresp = c(2, 5, 6, 3, 2), niv = 5)
    ))

    expect_no_error(test_simdata(
      seeds[i],
      list(N = 1000, nclass = 4, ndv = 5, niv = 4)
    ))

    set.seed(seeds[i])
    probs <- random_unvectorized_probs(c(2, 3, 5, 2, 2), 4)
    expect_no_error(test_simdata(
      sample.int(.Machine$integer.max, 1),
      list(N = 1000, probs = probs, niv = 6)
    ))
  }
})

test_that("regression-random-features", {
  # design matrix of features (aka x) is random within the function by not
  # providing it
  # vary which parameters to provide or not

  set.seed(-758321419)
  seeds <- sample.int(.Machine$integer.max, N_REPEAT)
  for (i in seq_len(N_REPEAT)) {
    expect_no_error(test_simdata(
      seeds[i],
      list(N = 1000, niv = 5)
    ))
    expect_no_error(test_simdata(
      seeds[i],
      list(N = 1000, nresp = c(3, 5, 4, 3, 2), niv = 5)
    ))

    set.seed(seeds[i])
    nclass <- 10
    gradient <- matrix(stats::rnorm((nclass - 1) * 10), ncol = nclass - 1)
    expect_no_error(test_simdata(
      sample.int(.Machine$integer.max, 1),
      list(N = 1000, b = gradient)
    ))

    set.seed(seeds[i])
    nclass <- 5
    gradient <- matrix(stats::rnorm((nclass - 1) * 10), ncol = nclass - 1)
    expect_no_error(test_simdata(
      sample.int(.Machine$integer.max, 1),
      list(N = 1000, nresp = c(2, 5, 6, 3, 2), b = gradient)
    ))

    set.seed(seeds[i])
    nclass <- 5
    probs <- random_unvectorized_probs(c(2, 3, 5, 2, 2), nclass)
    gradient <- matrix(stats::rnorm((nclass - 1) * 10), ncol = nclass - 1)
    expect_no_error(test_simdata(
      sample.int(.Machine$integer.max, 1),
      list(N = 1000, probs = probs, b = gradient)
    ))
  }
})

test_that("regression-provide-features", {
  # provide the design matrix of features (aka x)
  # vary which parameters to provide or not

  set.seed(-2009190883)
  seeds <- sample.int(.Machine$integer.max, N_REPEAT)

  for (i in seq_len(N_REPEAT)) {
    set.seed(seeds[i])
    n_data <- 1000
    nclass <- 4
    n_feature <- 5
    features <- as.matrix(random_features(n_data, n_feature))
    gradient <- matrix(stats::rnorm((nclass - 1) * (n_feature + 1)),
      nrow = n_feature + 1
    )
    expect_no_error(test_simdata(
      sample.int(.Machine$integer.max, 1),
      list(N = n_data, x = features, b = gradient)
    ))

    set.seed(seeds[i])
    n_data <- 1000
    nclass <- 4
    n_feature <- 5
    features <- as.matrix(random_features(n_data, n_feature))
    gradient <- matrix(stats::rnorm((nclass - 1) * (n_feature + 1)),
      nrow = n_feature + 1
    )
    expect_no_error(test_simdata(
      sample.int(.Machine$integer.max, 1),
      list(N = n_data, x = features, b = gradient, niv = 7)
    ))

    set.seed(seeds[i])
    n_data <- 1000
    nclass <- 4
    n_feature <- 5
    features <- as.matrix(random_features(n_data, n_feature))
    gradient <- matrix(stats::rnorm((nclass - 1) * (n_feature + 1)),
      nrow = n_feature + 1
    )
    expect_no_error(test_simdata(
      sample.int(.Machine$integer.max, 1),
      list(N = n_data, x = features, b = gradient, nresp = c(3, 5, 9, 2))
    ))

    set.seed(seeds[i])
    n_data <- 1000
    nclass <- 4
    n_feature <- 5
    features <- as.matrix(random_features(n_data, n_feature))
    gradient <- matrix(stats::rnorm((nclass - 1) * (n_feature + 1)),
      nrow = n_feature + 1
    )
    probs <- random_unvectorized_probs(c(2, 3, 5, 2, 2), nclass)
    expect_no_error(test_simdata(
      sample.int(.Machine$integer.max, 1),
      list(N = n_data, probs = probs, x = features, b = gradient)
    ))
  }
})

test_that("missing-data", {
  set.seed(1886413857)
  seeds <- sample.int(.Machine$integer.max, N_REPEAT)
  for (i in seq_len(N_REPEAT)) {
    expect_no_error(test_simdata(
      seeds[i],
      list(N = 1000, missval = TRUE)
    ))

    expect_no_error(test_simdata(
      seeds[i],
      list(N = 1000, missval = TRUE, pctmiss = 0.2)
    ))
  }
})
