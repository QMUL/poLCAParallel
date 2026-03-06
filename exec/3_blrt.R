# Example script on doing bootstrap likelihood ratio test using poLCAParallel
#
# For $R$ classes, bootstrap likelihood ratio test (BLRT) fits the null model
# using $R-1$ classes and an alt model using $R$ classes, producing a likelihood
# ratio. Using the fitted null and alt models, a parametric bootstrap can be
# done to get an empirical distribution of the likelihood ratio.
#
# To do model selection, select the highest $R$ where the fitted log likelihood
# ratio is on the 95% percentile or higher of the empirical (bootstrap)
# distribution of log likelihood ratios, eg p-value less than 5%. Thus this
# script should be run for a different number of classes, eg using an array job.
#
# The code uses poLCAParallel::blrt() to do BLRT. It records all bootstrap
# samples log likelihood ratios, which are plotted. Figures are saved in the
# current directory

nrep <- 32 # number of different initial values
n_class_max <- 10 # maximum number of classes to investigate
n_bootstrap <- 100 # number of bootstrap samples
set.seed(1746091653)

# carcinoma is the sample data from poLCA
data(carcinoma, package = "poLCAParallel")
data_og <- carcinoma
data_column_names <- colnames(data_og)
formula <- cbind(A, B, C, D, E, F, G) ~ 1

# fit the model onto the data for different number of classes
# save the fitted model into model_array
model_array <- list()
for (nclass in 1:n_class_max) {
  model <- poLCAParallel::poLCA(
    formula, data_og,
    nclass = nclass, nrep = nrep, verbose = FALSE
  )
  model_array[[nclass]] <- model
}

# store p values for each nclass, 1 to n_class_max
# store 0 for 1 number of class, ie this says you cannot have zero number of
# classes
p_value_array <- c(0)
# for all number of classes investigated:
#   - store the log likelihood ratio
#   - store all bootstrap samples log likelihoods ratios
fitted_log_ratio_array <- rep(NaN, n_class_max)
bootstrap_log_ratio_array <- list()

# do the bootstrap likelihood ratio test for each number of classes
for (nclass in 2:n_class_max) {
  # get the null and alt models
  # these are models with one number of class differences
  null_model <- model_array[[nclass - 1]]
  alt_model <- model_array[[nclass]]

  # for each bootstrap sample, store the log likelihood ratio here
  bootstrap_results <- poLCAParallel::blrt(
    null_model, alt_model, n_bootstrap, nrep
  )

  # log likelihood ratio to compare the two models
  fitted_log_ratio_array[nclass] <- bootstrap_results[["fitted_log_ratio"]]
  # store the log likelihoods ratios for all bootstrap samples
  bootstrap_log_ratio_array[[nclass]] <-
    bootstrap_results[["bootstrap_log_ratio"]]
  # store the p value for this nclass
  p_value_array <- c(p_value_array, bootstrap_results[["p_value"]])
}

# plot the bootstrap distribution of the log likelihood ratios for each class
# the red line shows the log likelihood ratio using the real data
dev.new()
boxplot(bootstrap_log_ratio_array,
  xlab = "number of classses", ylab = "log likelihood ratio"
)
# also plot the log likelihood ratio when using the real data
lines(1:n_class_max, fitted_log_ratio_array,
  type = "b", col = "red", pch = 15
)

# Plot the p value for each number of class.
# The p value is the proportion of bootstrap samples with log likelihood ratios
# greater than using the real data.
# Looking at the plot, I would select 3 or 4 classes as this is the biggest
# class with a small p value.
# Additional notes on how to interpret this graph:
#   - a low p value for a number of classes k suggest k number of classes is
#     better than k-1, so you would expect to see low p values for low number of
#     classes until you reach the optimal number of classes
#   - when the data follows the null hypothesis, the p value would follow a
#     uniform distribution, so for a class number too high, it should fluctuate
#     randomly between 0 and 1
# the solid line is at 5%
dev.new()
barplot(p_value_array,
  xlab = "number of classes", ylab = "p-value",
  names.arg = 1:n_class_max
)
abline(h = 0.05)
