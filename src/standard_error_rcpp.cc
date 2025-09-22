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
 * outcomes/responses for each category as integers 1, 2, 3, .... The matrix
 * has dimensions
 * <ul>
 *   <li>dim 0: for each data point</li>
 *   <li>dim 1: for each category</li>
 * </ul>
 * @param probs Vector of probabilities for each outcome, for each category,
 * for each cluster flatten list of matrices
 * <ul>
 *   <li>dim 0: for each outcome</li>
 *   <li>dim 1: for each category</li>
 *   <li>dim 2: for each cluster</li>
 * </ul>
 * @param prior Design matrix of prior probabilities, probability data point
 * is in cluster m NOT given responses after calculations, it shall be in
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
 * @return Rcpp::List
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
