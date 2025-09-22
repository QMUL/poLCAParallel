
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

#ifndef POLCAPARALLEL_INCLUDE_GOODNESS_FIT_H_
#define POLCAPARALLEL_INCLUDE_GOODNESS_FIT_H_

#include <cstddef>
#include <map>
#include <span>
#include <tuple>
#include <vector>

#include "util.h"

namespace polca_parallel {

/**
 * For storing the observed and expected frequency, used for chi-squared test
 */
struct Frequency {
  std::size_t observed;
  double expected;
};

/**
 * For calculating the goodness of fit
 *
 * For calculating the goodness of fit of the observed responses to the fitted
 * prior and outcome probabilities. Each uniquely observed responses are
 * recorded and with observed and expected frequencies tallied up. The log
 * likelihood ratio and chi squared statistics are calculated from that.
 *
 * How to use:
 * <ul>
 *   <li>Instantiate</li>
 *   <li>Call Calc()</li>
 *   <li>
 *     Call GetFrequencyMap() to get a map/table of unique responses and their
 *     respective frequencies
 *   </li>
 *   <li>
 *     Call GetStatistics() to calculate and get the log likelihood and
 *     chi-squared statistics
 *   </li>
 * </ul>
 *
 * Warning!!! It is known these statistics fall apart when the expected
 * frequency is less than about five. Reponses should be grouped together to
 * bring the expected frequency higher. Because this code tries to be as close
 * to the original, responses are not grouped together and low expected
 * frequencies can commonly occur. In these cases, the statistics will be
 * misleading.
 */
class GoodnessOfFit {
 private:
  /**
   * Map of all of uniquely observed responses with their observed and expected
   * frequency
   */
  std::map<std::vector<int>, Frequency> frequency_map_;
  /** Number of fully observed data, calculated after calling
   * CalcUniqueObserved()*/
  std::size_t n_obs_ = 0;

 public:
  /** Construct a GoodnessOfFit object */
  GoodnessOfFit();

  virtual ~GoodnessOfFit() = default;

  /**
   * Calculate the observed and expected frequencies
   *
   * Calculate the observed and expected frequencies for each unique response
   * in the parameter <code>responses</code>. The member variables
   * GoodnessOfFit::frequency_map_ and GoodnessOfFit::n_obs_ are modified
   *
   * This class supports missing data, encode missing data with 0 in
   * <code>responses</code>. Data points with missing values are ignored
   *
   * @param responses Design matrix <b>transposed</b> of responses, matrix
   * containing outcomes/responses for each category as integers 1, 2, 3, ....
   * Missing values may be encoded as 0. The matrix has dimensions
   * <ul>
   *   <li>dim 0: for each category</li>
   *   <li>dim 1: for each data point</li>
   * </ul>
   * @param prior Vector containing the prior probabilities for each cluster
   * <ul>
   *   <li>dim 0: for each cluster</li>
   * </ul>
   * @param outcome_prob The outcome probabilities, conditioned on the
   * cluster and category. A flatten list in the following order
   * <ul>
   *   <li>dim 0: for each outcome</li>
   *   <li>dim 1: for each category</li>
   *   <li>dim 2: for each cluster</li>
   * </ul>
   * @param n_data Number of data points
   * @param n_outcomes Vector of number of outcomes for each category and its
   * sum
   * @param n_cluster Number of clusters to fitted
   */
  void Calc(std::span<const int> responses, std::span<const double> prior,
            std::span<const double> outcome_prob, std::size_t n_data,
            NOutcomes n_outcomes, std::size_t n_cluster);

  /**
   * @return Map of uniquely observed responses with their observed and expected
   * frequency
   */
  [[nodiscard]] std::map<std::vector<int>, Frequency>& GetFrequencyMap();

  /**
   * Get chi-squared and log-likelihood ratio statistics
   *
   * Calculate and return the chi-squared statistics and log-likelihood ratio
   *
   * @return Log-likelihood ratio and chi-squared statistics
   */
  [[nodiscard]] std::tuple<double, double> GetStatistics() const;

 private:
  /**
   * Update the frequency map with unique responses and their count
   *
   * Find and count unique combinations of outcomes which are observed in the
   * dataset. Results are stored in the member variable
   * GoodnessOfFit::frequency_map_. Unique responses are saved as keys in the
   * map and the count the value of Frequency.observed
   *
   * @param responses Design matrix <b>transposed</b> of responses, matrix
   * containing outcomes/responses for each category as integers 1, 2, 3, ....
   * The matrix has dimensions
   * <ul>
   *   <li>dim 0: for each category</li>
   *   <li>dim 1: for each data point</li>
   * </ul>
   * @param n_data Number of data points
   * @param n_outcomes Array of integers, number of outcomes for each category
   */
  void CalcUniqueObserved(std::span<const int> responses, std::size_t n_data,
                          std::span<const std::size_t> n_outcomes);

  /**
   * Update the frequency map to contain expected frequencies
   *
   * For each key or unique response in the member variable
   * GoodnessOfFit::frequency_map_, modify the value of Frequency.expected with
   * the likelihood of that unique reponse multiplied by GoodnessOfFit::n_obs_
   *
   * @param prior Vector of prior probabilities (probability in a cluster),
   * length <code>n_cluster</code>
   * @param outcome_prob Vector of response probabilities, conditioned on
   * cluster and category. A flattened list in the following order
   * <ul>
   *   <li>dim 0: for each outcome</li>
   *   <li>dim 1: for each category</li>
   *   <li>dim 2: for each cluster</li>
   * </ul>
   * @param n_outcomes Array of integers, number of outcomes for each category
   * @param n_cluster Number of clusters
   */
  void CalcExpected(std::span<const double> prior,
                    std::span<const double> outcome_prob,
                    polca_parallel::NOutcomes n_outcomes,
                    std::size_t n_cluster);
};

}  // namespace polca_parallel

#endif  // POLCAPARALLEL_INCLUDE_GOODNESS_FIT_H_
