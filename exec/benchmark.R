benchmark <- function(n_data, n_category, max_n_outcomes, n_class, n_rep,
                      prob_na, na_rm, seed) {
  set.seed(seed)
  prob_class <- runif(n_class)
  prob_class <- prob_class / sum(prob_class)

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
    class_i <- sample(seq_len(n_class), 1, prob = prob_class)
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
  formula <- formula(
    paste0("cbind(", paste(colnames(y), collapse = ","), ")~1")
  )

  lca_parallel <- poLCAParallel::poLCA(
    formula, y,
    nclass = n_class, nrep = n_rep, verbose = FALSE, na.rm = na_rm
  )
  diff_time <- lca_parallel$time
  units(diff_time) <- "secs"
  # compare timings
  cat(paste("Time:", diff_time, "s\n"))
}

benchmark(1000, 20, 2, 5, 96, 0, TRUE, 10293444)
benchmark(5000, 20, 2, 10, 96, 0, TRUE, 10293444)
benchmark(5000, 20, 2, 15, 96, 0, TRUE, 10293444)
benchmark(5000, 20, 2, 20, 96, 0, TRUE, 10293444)
benchmark(10000, 20, 2, 10, 96, 0, TRUE, 10293444)
benchmark(10000, 50, 2, 10, 96, 0, TRUE, 10293444)

benchmark(1000, 20, 2, 5, 96, 0.01, FALSE, 10293444)
benchmark(5000, 20, 2, 10, 96, 0.01, FALSE, 10293444)
benchmark(5000, 20, 2, 15, 96, 0.01, FALSE, 10293444)
benchmark(5000, 20, 2, 20, 96, 0.01, FALSE, 10293444)
benchmark(10000, 20, 2, 10, 96, 0.01, FALSE, 10293444)
benchmark(10000, 50, 2, 10, 96, 0.01, FALSE, 10293444)
