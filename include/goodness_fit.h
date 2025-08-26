
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
  GoodnessOfFit();

  virtual ~GoodnessOfFit() = default;

  /**
   * Construct a GoodnessOfFit object
   *
   * @param responses Design matrix TRANSPOSED of responses, matrix containing
   * outcomes/responses for each category as integers 1, 2, 3, .... The matrix
   * has dimensions
   * <ul>
   *   <li>dim 0: for each category</li>
   *   <li>dim 1: for each data point</li>
   * </ul>
   * @param prior Contains the resulting prior probabilities after calling
   * Fit(). Design matrix of prior probabilities. It's the probability a data
   * point is in cluster m NOT given responses after calculations. The matrix
   * has the following dimensions dimensions
   * <ul>
   *   <li>dim 0: for each data</li>
   *   <li>dim 1: for each cluster</li>
   * </ul>
   * @param outcome_prob Contains the resulting outcome probabilities after
   * calling Fit(). Vector of estimated response probabilities, conditioned on
   * cluster, for each category. A flatten list in the following order
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

  [[nodiscard]] std::map<std::vector<int>, Frequency>& GetFrequencyMap();

  /**
   * Get chi-squared and log-likelihood ratio statistics
   *
   * Calculate and return the chi-squared statistics and log-likelihood ratio
   *
   * @return std::array<double, 2> containing log-likelihood ratio and
   * chi-squared statistics
   */
  [[nodiscard]] std::tuple<double, double> GetStatistics() const;

 private:
  /**
   * Update the frequency map with unique observations and their count
   *
   * Iterate through all the responses, then find and count unique combinations
   * of outcomes which were observed in the dataset. Results are stored in a
   * map. Observations are presented as a std::vector<int> of length n_category,
   * each element contains an int which represents the resulting outcome for
   * each category.
   *
   * @param responses Design matrix TRANSPOSED of responses, matrix containing
   * outcomes/responses
   * for each category as integers 1, 2, 3, .... The matrix has dimensions
   * <ul>
   *   <li>dim 0: for each category</li>
   *   <li>dim 1: for each data point</li>
   * </ul>
   * @param n_data number of data points
   * @param n_outcomes array of integers, number of outcomes for each category,
   * array of length n_category
   */
  void CalcUniqueObserved(std::span<const int> responses, std::size_t n_data,
                          std::span<const std::size_t> n_outcomes);

  /**
   * Update the frequency map to contain expected frequencies
   *
   * Update the expected frequency in a map of <vector<int>, Frequency> by
   * modifying the value of Frequency.expected with the likelihood of that
   * unique reponse with multiplied by n_data
   *
   * @param prior vector of prior probabilities (probability in a cluster),
   * length n_cluster
   * @param outcome_prob Vector of estimated response probabilities, conditioned
   * on cluster, for each category. A flattened list in the following order
   * <ul>
   *   <li>dim 0: for each outcome</li>
   *   <li>dim 1: for each category</li>
   *   <li>dim 2: for each cluster</li>
   * </ul>
   * @param n_outcomes array of integers, number of outcomes for each category,
   * array of length n_category
   * @param n_cluster number of clusters (or classes)
   */
  void CalcExpected(std::span<const double> prior,
                    std::span<const double> outcome_prob,
                    polca_parallel::NOutcomes n_outcomes,
                    std::size_t n_cluster);
};

}  // namespace polca_parallel

#endif  // POLCAPARALLEL_INCLUDE_GOODNESS_FIT_H_
