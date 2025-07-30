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

#ifndef POLCAPARALLEL_INCLUDE_EM_ALGORITHM_ARRAY_SERIAL_H_
#define POLCAPARALLEL_INCLUDE_EM_ALGORITHM_ARRAY_SERIAL_H_

#include <cstddef>
#include <memory>
#include <random>
#include <span>

#include "em_algorithm_array.h"
#include "util.h"

namespace polca_parallel {

/**
 * Serial version of EmAlgorithmArray
 *
 * Only uses one thread (so the parameter n_thread is not provided) and each
 * repetition reuses one rng, rather than each repetition having a rng each.
 * Thus the member variable EmAlgorithmArraySerial::seed_array_ shall only
 * contain one seed. The rng is only used for creating new initial values should
 * a repetition fail.
 *
 * This is used by Blrt where each thread works on different bootstrap samples
 * in parallel. This ensures no additional threads are spawned.
 *
 */
class EmAlgorithmArraySerial : public polca_parallel::EmAlgorithmArray {
 private:
  /** The one and only random number generator to be used by all repetitions */
  std::unique_ptr<std::mt19937_64> rng_;

 public:
  /**
   * Construct a new EM Algorithm Array object
   *
   * Construct a new EM Algorithm Array object. This serial version only uses
   * one thread
   *
   * @param features Design matrix of features, matrix with dimensions
   * <ul>
   *   <li>dim 0: for each data point</li>
   *   <li>dim 1: for each feature</li>
   * </ul>
   * Can be empty and not used if using only for the non-regression problem
   * @param responses Design matrix <b>transposed</b> of responses, matrix
   * containing outcomes/responses for each category as integers 1, 2, 3, ....
   * The matrix has dimensions
   * <ul>
   *   <li>dim 0: for each category</li>
   *   <li>dim 1: for each data point</li>
   * </ul>
   * @param initial_prob Vector of initial probabilities for each outcome, for
   * each category, for each cluster and for each repetition, flatten list in
   * the following order
   * <ul>
   *   <li>dim 0: for each outcome</li>
   *   <li>dim 1: for each category</li>
   *   <li>dim 2: for each cluster</li>
   *   <li>dim 3: for each repetition</li>
   * </ul>
   * Use RandomInitialProb() in util.h to produce random initial probabilities
   * if required
   * @param n_data Number of data points
   * @param n_feature Number of features, set to 1 if this is a non-regression
   * problem
   * @param n_outcomes Array of the number of outcomes for each category and its
   * sum
   * @param n_cluster Number of clusters to fit
   * @param n_rep Number of repetitions to do, this defines dim 3 of
   * initial_prob
   * @param max_iter Maximum number of iterations for EM algorithm
   * @param tolerance Tolerance for the difference in log-likelihood, used for
   * stopping condition
   * @param posterior To store results, design matrix of posterior probabilities
   * (also called responsibility), the probability a data point is in cluster
   * m given responses, matrix with dimensions
   * <ul>
   *   <li>dim 0: for each data</li>
   *   <li>dim 1: for each cluster</li>
   * </ul>
   * @param prior To store results, design matrix of prior probabilities,
   * the probability a data point is in cluster m <b>not</b> given responses
   * <ul>
   *   <li>dim 0: for each data</li>
   *   <li>dim 1: for each cluster</li>
   * </ul>
   * @param estimated_prob To store results, vector of estimated response
   * probabilities for each category, flatten list in the following order
   * <ul>
   *   <li>dim 0: for each outcome</li>
   *   <li>dim 1: for each category</li>
   *   <li>dim 2: for each cluster</li>
   * </ul>
   * @param regress_coeff To store results, matrix with dimensions:
   * <ul>
   *   <li>dim 0: n_features</li>
   *   <li>dim 1: n_cluster - 1</li>
   * </ul>
   * This matrix is multiplied to the feature design matrix and then linked to
   * the prior using softmax. Not used in the non-regression problem
   */
  EmAlgorithmArraySerial(
      std::span<const double> features, std::span<const int> responses,
      std::span<const double> initial_prob, std::size_t n_data,
      std::size_t n_feature, NOutcomes n_outcomes, std::size_t n_cluster,
      std::size_t n_rep, unsigned int max_iter, double tolerance,
      std::span<double> posterior, std::span<double> prior,
      std::span<double> estimated_prob, std::span<double> regress_coeff);

  /**
   * Construct a new EM Algorithm Array object
   *
   * Construct a new EM Algorithm Array object for clustering (non-regression)
   * only. This serial version only uses one thread
   *
   * @param responses Design matrix <b>transposed</b> of responses, matrix
   * containing outcomes/responses for each category as integers 1, 2, 3, ....
   * The matrix has dimensions
   * <ul>
   *   <li>dim 0: for each category</li>
   *   <li>dim 1: for each data point</li>
   * </ul>
   * @param initial_prob Vector of initial probabilities for each outcome, for
   * each category, for each cluster and for each repetition, flatten list in
   * the following order
   * <ul>
   *   <li>dim 0: for each outcome</li>
   *   <li>dim 1: for each category</li>
   *   <li>dim 2: for each cluster</li>
   *   <li>dim 3: for each repetition</li>
   * </ul>
   * Use RandomInitialProb() in util.h to produce random initial probabilities
   * @param n_data Number of data points
   * @param n_outcomes Array of the number of outcomes for each category and its
   * sum
   * @param n_cluster Number of clusters to fit
   * @param n_rep Number of repetitions to do, this defines dim 3 of
   * initial_prob
   * @param max_iter Maximum number of iterations for EM algorithm
   * @param tolerance Tolerance for the difference in log-likelihood, used for
   * stopping condition
   * @param posterior To store results, design matrix of posterior probabilities
   * (also called responsibility), the probability a data point is in cluster
   * m given responses, matrix with dimensions
   * <ul>
   *   <li>dim 0: for each data</li>
   *   <li>dim 1: for each cluster</li>
   * </ul>
   * @param prior To store results, design matrix of prior probabilities,
   * the probability a data point is in cluster m <b>not</b> given responses
   * <ul>
   *   <li>dim 0: for each data</li>
   *   <li>dim 1: for each cluster</li>
   * </ul>
   * @param estimated_prob To store results, vector of estimated response
   * probabilities for each category, flatten list in the following order
   * <ul>
   *   <li>dim 0: for each outcome</li>
   *   <li>dim 1: for each category</li>
   *   <li>dim 2: for each cluster</li>
   * </ul>
   */
  EmAlgorithmArraySerial(std::span<const int> responses,
                         std::span<const double> initial_prob,
                         std::size_t n_data, NOutcomes n_outcomes,
                         std::size_t n_cluster, std::size_t n_rep,
                         unsigned int max_iter, double tolerance,
                         std::span<double> posterior, std::span<double> prior,
                         std::span<double> estimated_prob);

  ~EmAlgorithmArraySerial() override = default;

  /**
   * Set the seed_array_ to contain only one seed and instantiate the rng_
   *
   * The rng is only used if a repetition fails and tries again using new
   * initial values generated by the rng.
   */
  void SetSeed(std::seed_seq& seed) override;

  /**
   * Set the seed_array_ to contain only one seed and instantiate the rng_
   *
   * The rng is only used if a repetition fails and tries again using new
   * initial values generated by the rng.
   */
  void SetSeed(unsigned seed);

  /**
   * Transfer ownership of a rng to this object and set rng_
   *
   * Transfer ownership of a rng to this object and set rng_. This rng is only
   * used if a repetition fails and tries again using new initial values
   * generated by the rng.
   */
  void SetRng(std::unique_ptr<std::mt19937_64> rng);

  /**
   * Transfer ownership of the rng from this object as a return value
   */
  [[nodiscard]] std::unique_ptr<std::mt19937_64> MoveRng();

 protected:
  /**
   * Set the rng of an EmAlgorithm object
   *
   * This will transfer ownership of rng_ from this object to the fitter. Ensure
   * to call MoveRngBackFromFitter() to retrieve it back afterwards.
   *
   * Because each repetition reuses the same rng, the parameter rep_index is
   * ignored.
   */
  void SetFitterRng(std::size_t rep_index,
                    polca_parallel::EmAlgorithm& fitter) override;

  /**
   * Transfer ownership of an EmAlgorithm's rng back to this object's rng_
   */
  void MoveRngBackFromFitter(polca_parallel::EmAlgorithm& fitter) override;
};

}  // namespace polca_parallel

#endif  // POLCAPARALLEL_INCLUDE_EM_ALGORITHM_ARRAY_SERIAL_H_
