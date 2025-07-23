
// poLCAParallel
// Copyright (C) 2022 Sherman Lo

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

#include <array>
#include <cstddef>
#include <iterator>
#include <map>
#include <span>
#include <vector>

#include "goodness_fit.h"
#include "util.h"

/**
 * Function to be exported to R, goodness of fit statistics
 *
 * Get goodness of fit statistics given fitted probabilities
 *
 * @param responses Design matrix TRANSPOSED of responses, matrix containing
 * outcomes/responses for each category as integers 1, 2, 3, .... The matrix has
 * dimensions
 * <ul>
 *   <li>dim 0: for each category</li>
 *   <li>dim 1: for each data point</li>
 * </ul>
 * @param prior: vector of prior probabilities, for each cluster
 * @param outcome_prob: vector of response probabilities for each cluster,
 * flatten list of matrices, from the return value of poLCA.vectorize.R, flatten
 * list of matrices
 * <ul>
 *   <li>dim 0: for each outcome</li>
 *   <li>dim 1: for each category</li>
 *   <li>dim 2: for each cluster</li>
 * </ul>
 * @param n_data number of data points
 * @param n_obs number of fully observed data points
 * @param n_outcomes_int vector, number of possible responses for each category
 * @param n_cluster number of clusters, or classes, to fit
 * @return a list containing:
 * <ul>
 *   <li>unique_freq_table: data frame of unique responses with their observed
 *   frequency and expected frequency</li>
 *   <li>ln_l_ratio</li>
 *   <li>chi_squared</li>
 * </ul>
 */
// [[Rcpp::export]]
Rcpp::List GoodnessFitRcpp(Rcpp::IntegerMatrix responses,
                           Rcpp::NumericVector prior,
                           Rcpp::NumericVector outcome_prob, std::size_t n_data,
                           std::size_t n_obs,
                           Rcpp::IntegerVector n_outcomes_int,
                           std ::size_t n_cluster) {
  std::vector<std::size_t> n_outcomes_size_t(n_outcomes_int.cbegin(),
                                             n_outcomes_int.cend());
  polca_parallel::NOutcomes n_outcomes(n_outcomes_size_t.data(),
                                       n_outcomes_size_t.size());
  std::size_t n_category = n_outcomes.size();

  polca_parallel::GoodnessOfFit goodness_of_fit;
  goodness_of_fit.Calc(
      std::span<const int>(responses.cbegin(), responses.size()),
      std::span<const double>(prior.cbegin(), prior.size()),
      std::span<const double>(outcome_prob.cbegin(), outcome_prob.size()),
      n_data, n_obs, n_outcomes, n_cluster);

  std::map<std::vector<int>, polca_parallel::Frequency>& frequency_map =
      goodness_of_fit.GetFrequencyMap();

  // get log likelihood ratio and chi squared statistics
  auto [ln_l_ratio, chi_squared] = goodness_of_fit.GetStatistics(n_data);

  // transfer results from frequency_map to a NumericMatrix
  // frequency_table
  // last two columns for observed and expected frequency
  std::size_t n_unique = frequency_map.size();
  Rcpp::NumericMatrix frequency_table(n_unique, n_category + 2);
  auto freq_table_ptr = frequency_table.begin();

  std::size_t data_index = 0;
  for (auto iter = frequency_map.cbegin(); iter != frequency_map.cend();
       ++iter) {
    const std::vector<int>& response_i = iter->first;
    polca_parallel::Frequency frequency = iter->second;

    // copy over response
    for (std::size_t j = 0; j < n_category; ++j) {
      *std::next(freq_table_ptr, j * n_unique + data_index) = response_i[j];
    }
    // copy over observed and expected frequency
    *std::next(freq_table_ptr, n_category * n_unique + data_index) =
        static_cast<double>(frequency.observed);
    *std::next(freq_table_ptr, (n_category + 1) * n_unique + data_index) =
        frequency.expected;
    ++data_index;
  }

  Rcpp::List to_return;
  to_return.push_back(frequency_table);
  to_return.push_back(ln_l_ratio);
  to_return.push_back(chi_squared);

  return to_return;
}
