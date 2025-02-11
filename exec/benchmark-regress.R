benchmark <- function(n_data, n_category, max_n_outcomes, n_feature, n_class, n_rep,
                      prob_na, na_rm, seed) {
  set.seed(seed)
  coeff <- matrix(rnorm(n_feature * (n_class - 1)), n_feature, n_class - 1)
  features <- matrix(rnorm(n_data * n_feature), n_data, n_feature)

  prob_class <- cbind(1, exp(features %*% coeff))
  prob_class <- prob_class / replicate(n_class, rowSums(prob_class))

  n_outcomes <- rep(0, n_category)
  for (i in seq_len(n_category)) {
    n_outcomes[i] <- sample(seq_len(max_n_outcomes - 1), 1) + 1
  }

  probs <- list()
  for (i in seq_len(n_class)) {
    probs[[i]] <- list()
    for (j in seq_len(n_category)) {
      probs_i_j <- runif(n_outcomes[j])
      probs[[i]][[j]] <- probs_i_j / sum(probs_i_j)
    }
  }

  y <- matrix(nrow = n_data, ncol = n_category)
  for (i in seq_len(n_data)) {
    class_i <- sample(seq_len(n_class), 1, prob = prob_class[i, ])
    for (j in seq_len(n_category)) {
      if (runif(1) < prob_na) {
        y[i, j] <- NA
      } else {
        y[i, j] <- sample(seq_len(n_outcomes[j]),
          1,
          prob = probs[[class_i]][[j]]
        )
      }
    }
  }

  y <- as.data.frame(y)

  x <- as.data.frame(features)
  colnames(x) <- paste0(rep("U", n_feature), paste(seq_len(n_feature)))

  data <- merge(y, x, by = "row.names", all = TRUE)

  formula <- formula(
    paste0(
      "cbind(", paste(colnames(y), collapse = ","), ")~cbind(",
      paste0(colnames(x), collapse = "+"),
      ")"
    )
  )

  lca_parallel <- poLCAParallel::poLCA(
    formula, data,
    nclass = n_class, nrep = n_rep, verbose = FALSE, na.rm = na_rm
  )
  diff_time <- lca_parallel$time
  units(diff_time) <- "secs"
  # compare timings
  cat(paste("Time:", diff_time, "s\n"))
}

benchmark(1000, 5, 2, 3, 5, 96, 0, TRUE, 10293444)
benchmark(1000, 10, 2, 5, 10, 96, 0, TRUE, 10293444)
benchmark(1000, 30, 2, 10, 15, 96, 0, TRUE, 10293444)
benchmark(1000, 30, 2, 10, 20, 96, 0, TRUE, 10293444)
benchmark(2000, 30, 2, 10, 15, 96, 0, TRUE, 10293444)
benchmark(2000, 30, 2, 10, 20, 96, 0, TRUE, 10293444)

benchmark(1000, 5, 2, 3, 5, 96, 0.01, FALSE, 10293444)
benchmark(1000, 10, 2, 5, 10, 96, 0.01, FALSE, 10293444)
benchmark(1000, 30, 2, 10, 15, 96, 0.01, FALSE, 10293444)
benchmark(1000, 30, 2, 10, 20, 96, 0.01, FALSE, 10293444)
benchmark(2000, 30, 2, 10, 15, 96, 0.01, FALSE, 10293444)
benchmark(2000, 30, 2, 10, 20, 96, 0.01, FALSE, 10293444)
