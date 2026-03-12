// Copyright (C) 2025 Sherman Lo

// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License along
// with this program; if not, write to the Free Software Foundation, Inc.,
// 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

/**
 * Rcpp functions and C++ code which do not belong in the polca_parallel
 * namespace and antipicated to be used through R only
 *
 * Functions here are Posterior() and Likelihood(). It could be possible they
 * can be integrated into the polca_parallel namespace. See EmAlgorithm::EStep()
 *
 */

#include <RcppArmadillo.h>

#include <cstddef>

#include "em_algorithm.h"
#include "util_rcpp.h"

/**
 * Calculate the posterior for every data point and cluster
 *
 * @param responses Design matrix <b>transposed</b> of responses, matrix
 * containing outcomes/responses for each category as integers 1, 2, 3, .... If
 * supported, 0 can be used to indicate a missing value. The matrix has
 * dimensions
 * <ul>
 *   <li>dim 0: for each category</li>
 *   <li>dim 1: for each data point</li>
 * </ul>
 * @param probs Matrix of response probabilities for each outcome, conditioned
 * on the category and cluster. A flattened list in the following order
 * <ul>
 *   <li>
 *     dim 0: for each outcome | category (inner), for each category (outer)
 *   </li>
 *   <li>dim 1: for each cluster</li>
 * </ul>
 * @param n_outcomes Number of possible outcomes for each category
 * @param prior The probability a data point is in cluster m <b>not</b> given
 * responses after calculations. The matrix has the following dimensions
 * <ul>
 *   <li>dim 0: for each data</li>
 *   <li>dim 1: for each cluster</li>
 * </ul>
 * @param n_data Number of data points
 * @param n_cluster Number of clusters
 * @param posterior <b>Modified</b> The resulting posterior for each data point
 * and cluster. the probability a data point is in cluster m given response. The
 * matrix has the following dimensions
 * <ul>
 *   <li>dim 0: for each data</li>
 *   <li>dim 1: for each cluster</li>
 * </ul>
 */
void Posterior(std::span<const int> responses, const arma::Mat<double>& probs,
               std::span<const size_t> n_outcomes,
               std::span<const double> prior, std::size_t n_data,
               std::size_t n_cluster, arma::Mat<double>& posterior) {
  auto posterior_ptr = posterior.begin();
  auto prior_ptr = prior.begin();
  for (std::size_t i_cluster = 0; i_cluster < n_cluster; ++i_cluster) {
    for (std::size_t i_data = 0; i_data < n_data; ++i_data) {
      std::span<const int> responses_i =
          responses.subspan(i_data * n_outcomes.size(), n_outcomes.size());
      // use <true> so it does zero checks
      // set to true as it is assumed this is not performance critical
      // ie only run once, unlike in the EM algorithm where the E step is called
      // iteratively
      *posterior_ptr =
          polca_parallel::Likelihood<true>(responses_i, n_outcomes,
                                           probs.unsafe_col(i_cluster)) *
          *prior_ptr;
      std::advance(posterior_ptr, 1);
      std::advance(prior_ptr, 1);
    }
  }
  auto normaliser = arma::sum(posterior, 1);  // row sum
  posterior.each_col() /= normaliser;         // normalise by the row sum
}

/**
 * Calculate the posterior for every data point and cluster
 *
 * @param responses Design matrix <b>transposed</b> of responses, matrix
 * containing outcomes/responses for each category as integers 1, 2, 3, .... If
 * supported, 0 can be used to indicate a missing value. The matrix has
 * dimensions
 * <ul>
 *   <li>dim 0: for each category</li>
 *   <li>dim 1: for each data point</li>
 * </ul>
 * @param probs Vector of response probabilities for each outcome, conditioned
 * on the category and cluster. Can be the return value of
 * <code>poLCAParallel.vectorize.R</code>. Flatten list in the following order
 * <ul>
 *   <li>dim 0: for each outcome</li>
 *   <li>dim 1: for each category</li>
 *   <li>dim 2: for each cluster</li>
 * </ul>
 * @param n_outcomes_int Number of possible outcomes for each category
 * @param prior The probability a data point is in cluster m <b>not</b> given
 * responses after calculations. The matrix has the following dimensions <ul>
 *   <li>dim 0: for each data</li>
 *   <li>dim 1: for each cluster</li>
 * </ul>
 * @param n_data Number of data points
 * @param n_cluster Number of clusters
 * @return The resulting posterior for each data point and cluster. the
 * probability a data point is in cluster m given response. The matrix has the
 * following dimensions
 * <ul>
 *   <li>dim 0: for each data</li>
 *   <li>dim 1: for each cluster</li>
 * </ul>
 */
// [[Rcpp::export]]
Rcpp::NumericMatrix PosteriorRcpp(Rcpp::IntegerVector responses,
                                  Rcpp::NumericVector probs,
                                  Rcpp::IntegerVector n_outcomes_int,
                                  Rcpp::NumericVector prior, std::size_t n_data,
                                  std::size_t n_cluster) {
  Rcpp::NumericMatrix posterior(n_data, n_cluster);

  std::vector<std::size_t> n_outcomes_size_t(n_outcomes_int.cbegin(),
                                             n_outcomes_int.cend());
  polca_parallel::NOutcomes n_outcomes(n_outcomes_size_t.data(),
                                       n_outcomes_size_t.size());

  arma::Mat<double> posterior_arma(posterior.begin(), n_data, n_cluster, false,
                                   true);

  const arma::Mat<double> probs_arma(probs.begin(), n_outcomes.sum(), n_cluster,
                                     false, true);

  Posterior(polca_parallel::VectorToConstSpan(responses), probs_arma,
            n_outcomes, polca_parallel::VectorToConstSpan(prior), n_data,
            n_cluster, posterior_arma);

  return posterior;
}

/**
 * Calculate the likelihood for every data point and cluster
 *
 * @param responses Design matrix <b>transposed</b> of responses, matrix
 * containing outcomes/responses for each category as integers 1, 2, 3, .... If
 * supported, 0 can be used to indicate a missing value. The matrix has
 * dimensions
 * <ul>
 *   <li>dim 0: for each category</li>
 *   <li>dim 1: for each data point</li>
 * </ul>
 * @param probs Matrix of response probabilities for each outcome, conditioned
 * on the category and cluster. A flattened list in the following order
 * <ul>
 *   <li>
 *     dim 0: for each outcome | category (inner), for each category (outer)
 *   </li>
 *   <li>dim 1: for each cluster</li>
 * </ul>
 * @param n_outcomes Number of possible outcomes for each category
 * @param n_data Number of data points
 * @param n_cluster Number of clusters
 * @param likelihood <b>Modified</b> The resulting likelihood for each data
 * point and cluster. the probability a data point is in cluster m given
 * response. The matrix has the following dimensions
 * <ul>
 *   <li>dim 0: for each data</li>
 *   <li>dim 1: for each cluster</li>
 * </ul>
 */
void Likelihood(std::span<const int> responses, const arma::Mat<double>& probs,
                std::span<const size_t> n_outcomes, std::size_t n_data,
                std::size_t n_cluster, std::span<double> likelihood) {
  auto likelihood_ptr = likelihood.begin();

  for (std::size_t i_cluster = 0; i_cluster < n_cluster; ++i_cluster) {
    for (std::size_t i_data = 0; i_data < n_data; ++i_data) {
      std::span<const int> responses_i =
          responses.subspan(i_data * n_outcomes.size(), n_outcomes.size());

      *likelihood_ptr = polca_parallel::Likelihood(responses_i, n_outcomes,
                                                   probs.unsafe_col(i_cluster));
      std::advance(likelihood_ptr, 1);
    }
  }
}

/**
 * Calculate the likelihood for every data point and cluster
 *
 * @param responses Design matrix <b>transposed</b> of responses, matrix
 * containing outcomes/responses for each category as integers 1, 2, 3, .... If
 * supported, 0 can be used to indicate a missing value. The matrix has
 * dimensions
 * <ul>
 *   <li>dim 0: for each category</li>
 *   <li>dim 1: for each data point</li>
 * </ul>
 * @param probs Vector of response probabilities for each outcome, conditioned
 * on the category and cluster. Can be the return value of
 * <code>poLCAParallel.vectorize.R</code>. Flatten list in the following order
 * <ul>
 *   <li>dim 0: for each outcome</li>
 *   <li>dim 1: for each category</li>
 *   <li>dim 2: for each cluster</li>
 * </ul>
 * @param n_outcomes_int Number of possible outcomes for each category
 * @param n_data Number of data points
 * @param n_cluster Number of clusters
 * @return The resulting likelihood for each data point and cluster. the
 * probability a data point is in cluster m given response. The matrix has the
 * following dimensions
 * <ul>
 *   <li>dim 0: for each data</li>
 *   <li>dim 1: for each cluster</li>
 * </ul>
 */
// [[Rcpp::export]]
Rcpp::NumericMatrix LikelihoodRcpp(Rcpp::IntegerVector responses,
                                   Rcpp::NumericVector probs,
                                   Rcpp::IntegerVector n_outcomes_int,
                                   std::size_t n_data, std::size_t n_cluster) {
  Rcpp::NumericMatrix likelihood(n_data, n_cluster);

  std::vector<std::size_t> n_outcomes_size_t(n_outcomes_int.cbegin(),
                                             n_outcomes_int.cend());
  polca_parallel::NOutcomes n_outcomes(n_outcomes_size_t.data(),
                                       n_outcomes_size_t.size());

  const arma::Mat<double> probs_arma(probs.begin(), n_outcomes.sum(), n_cluster,
                                     false, true);

  Likelihood(polca_parallel::VectorToConstSpan(responses), probs_arma,
             n_outcomes, n_data, n_cluster,
             polca_parallel::VectorToSpan(likelihood));

  return likelihood;
}
