// poLCAParallel
// Copyright (C) 2024 Sherman Lo

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

#include <RcppArmadillo.h>

#include <cstddef>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include "regularised_error.h"
#include "standard_error.h"
#include "standard_error_regress.h"
#include "util.h"

/**
 * Instantiate a polca_parallel::StandardError object
 *
 * Instantiate a polca_parallel::StandardError object (which may also be from a
 * subclass) according to the user provided <code>n_feature</code> and
 * <code>use_smooth</code>.
 *
 * It uses <code>n_feature</code> to determine if it is a regression problem
 * or not
 *
 * It uses <code>use_smooth</code> to determine whether to use the standard or
 * regularised error
 *
 * @tparam Args
 * @param n_feature Number of features
 * @param use_smooth Whether to smooth the outcome probabilities to produce more
 *  numerical stable results at the cost of bias
 * @param args All arguments to pass to the constructor of
 * polca_parallel::StandardError, including <code>n_feature</code>, in order.
 * See polca_parallel::StandardError::StandardError
 * @return std::unique_ptr<polca_parallel::StandardError>
 */
template <typename... Args>
std::unique_ptr<polca_parallel::StandardError> InitStandardError(
    std::size_t n_feature, bool use_smooth, Args... args) {
  if (n_feature == 1) {
    if (use_smooth) {
      return std::make_unique<polca_parallel::RegularisedError>(
          std::forward<Args>(args)...);
    } else {
      return std::make_unique<polca_parallel::StandardError>(
          std::forward<Args>(args)...);
    }
  } else {
    if (use_smooth) {
      return std::make_unique<polca_parallel::RegularisedErrorRegress>(
          std::forward<Args>(args)...);
    } else {
      return std::make_unique<polca_parallel::StandardErrorRegress>(
          std::forward<Args>(args)...);
    }
  }
}

/**
 * To be exported to R, calculate the standard error for a poLCA model
 *
 * @param features Design matrix of features, matrix with dimensions
 * <ul>
 *   <li>dim 0: for each data point</li>
 *   <li>dim 1: for each feature</li>
 * </ul>
 * @param responses Design matrix of responses, matrix containing
 * outcomes/responses for each category as integers 1, 2, 3, .... Missing values
 * may be encoded as 0. The matrix has dimensions
 * <ul>
 *   <li>dim 0: for each data point</li>
 *   <li>dim 1: for each category</li>
 * </ul>
 * @param probs Vector of response probabilities for each outcome, conditioned
 * on the category and cluster. Can be the return value of
 * <code>poLCAParallel.vectorize.R</code>. Flatten list in the following order
 * <ul>
 *   <li>dim 0: for each outcome</li>
 *   <li>dim 1: for each category</li>
 *   <li>dim 2: for each cluster</li>
 * </ul>
 * @param prior Design matrix of prior probabilities, probability data point
 * is in cluster m <b>not</b> given responses after calculations, it shall be in
 * matrix form with dimensions
 * <ul>
 *   <li>dim 0: for each data</li>
 *   <li>dim 1: for each cluster</li>
 * </ul>
 * @param posterior Design matrix of posterior probabilities (also called
 * responsibility), probability data point is in cluster m given responses
 * matrix
 * <ul>
 *   <li>dim 0: for each data</li>
 *   <li>dim 1: for each cluster</li>
 * </ul>
 * @param n_data Number of data points
 * @param n_feature Number of features
 * @param n_outcomes_int Array of number of outcomes, for each category
 * @param n_cluster Number of clusters fitted
 * @param use_smooth True to smooth the outcome probabilities
 * @return A list containing
 * <ul>
 *   <li>
 *     <code>[[1]]</code>:
 *     prior error, vector of length <code>n_cluster</code>
 *   </li>
 *   <li>
 *     <code>[[2]]</code>:
 *     outcome probabilities error, vector of length
 *     <code>n_outcomes.sum() * n_cluster</code>
 *   </li>
 *   <li>
 *     <code>[[3]]</code>:
 *     regress_coeff_error, covariance matrix of size
 *     <ul>
 *       <li>dim 0: <code>n_feature * (n_cluster - 1)</code></li>
 *       <li>dim 1: <code>n_feature * (n_cluster - 1)</code></li>
 *     </ul>
 *   </li>
 * </ul>
 */
// [[Rcpp::export]]
Rcpp::List StandardErrorRcpp(Rcpp::NumericVector features,
                             Rcpp::IntegerMatrix responses,
                             Rcpp::NumericVector probs,
                             Rcpp::NumericMatrix prior,
                             Rcpp::NumericMatrix posterior, std::size_t n_data,
                             std::size_t n_feature,
                             Rcpp::IntegerVector n_outcomes_int,
                             std::size_t n_cluster, bool use_smooth) {
  std::vector<std::size_t> n_outcomes_size_t(n_outcomes_int.cbegin(),
                                             n_outcomes_int.cend());
  polca_parallel::NOutcomes n_outcomes(n_outcomes_size_t.data(),
                                       n_outcomes_size_t.size());

  std::size_t len_regress_coeff = n_feature * (n_cluster - 1);

  // allocate matrices to pass pointers to C++ code
  Rcpp::NumericVector prior_error(n_cluster);
  Rcpp::NumericVector probs_error(n_outcomes.sum() * n_cluster);
  Rcpp::NumericMatrix regress_coeff_error(len_regress_coeff, len_regress_coeff);

  std::unique_ptr<polca_parallel::StandardError> error = InitStandardError(
      n_feature, use_smooth, std::span<const double>(features),
      std::span<const int>(responses), std::span<const double>(probs),
      std::span<const double>(prior), std::span<const double>(posterior),
      n_data, n_feature, n_outcomes, n_cluster, std::span<double>(prior_error),
      std::span<double>(probs_error), std::span<double>(regress_coeff_error));
  error->Calc();

  Rcpp::List to_return;
  to_return.push_back(prior_error);
  to_return.push_back(probs_error);
  to_return.push_back(regress_coeff_error);
  return to_return;
}
