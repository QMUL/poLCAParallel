#include "RcppArmadillo.h"

#include "em_algorithm.h"

using namespace Rcpp;

// EM FIT
// Fit using the EM algorithm
// Args:
  // features: design matrix of features
  // responses: design matrix transpose of responses
  // initial_prob: vector of response probabilities for each cluster, flatten
    // list of matrices, from the return value of poLCA.vectorize.R
    // flatten list of matrices
      // dim 0: for each outcome
      // dim 1: for each category
      // dim 2: for each cluster
  // n_data: number of data points
  // n_feature: number of features
  // n_category: number of categories
  // n_outcomes: vector, number of possible responses for each category
  // n_cluster: number of clusters, or classes, to fit
  // max_iter: maximum number of iterations for EM algorithm
  // tolerance: stop fitting the log likelihood change less than this
// Return a list:
  // posterior: matrix of posterior probabilities, dim 0: for each data point,
    // dim 1: for each cluster
  // prior: matrix of prior probabilities, dim 0: for each data point,
    // dim 1: for each cluster
  // estimated_prob: vector of estimated response probabilities, in the same
    // format as initial_prob
  // ln_l: log likelihood
  // n_iter: number of iterations taken
// [[Rcpp::export]]
List emFit(
    NumericMatrix features,
    IntegerMatrix responses,
    NumericVector initial_prob,
    int n_data,
    int n_feature,
    int n_category,
    IntegerVector n_outcomes,
    int n_cluster,
    int max_iter,
    double tolerance) {

  int sum_outcomes = 0;
  int* n_outcomes_array = n_outcomes.begin();
  for (int i=0; i<n_category; i++) {
    sum_outcomes += n_outcomes_array[i];
  }

  NumericMatrix posterior(n_data, n_cluster);
  NumericMatrix prior(n_data, n_cluster);
  NumericVector estimated_prob(sum_outcomes*n_cluster);
  NumericVector regress_coeff(n_feature*(n_cluster-1));

  EmAlgorithm* fitter = new EmAlgorithm(
      features.begin(),
      responses.begin(),
      initial_prob.begin(),
      n_data,
      n_feature,
      n_category,
      n_outcomes.begin(),
      sum_outcomes,
      n_cluster,
      max_iter,
      tolerance,
      posterior.begin(),
      prior.begin(),
      estimated_prob.begin(),
      regress_coeff.begin()
  );

  fitter->Fit();

  double ln_l = fitter->get_ln_l();
  int n_iter = fitter->get_n_iter();

  delete fitter;

  List to_return;
  to_return.push_back(posterior);
  to_return.push_back(prior);
  to_return.push_back(estimated_prob);
  to_return.push_back(regress_coeff);
  to_return.push_back(ln_l);
  to_return.push_back(n_iter);
  return to_return;
}
