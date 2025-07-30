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

#ifndef POLCAPARALLEL_INCLUDE_EM_ALGORITHM_ARRAY_H_
#define POLCAPARALLEL_INCLUDE_EM_ALGORITHM_ARRAY_H_

#include <atomic>
#include <cstddef>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <span>
#include <vector>

#include "em_algorithm.h"
#include "util.h"

namespace polca_parallel {

/**
 * For using the EM algorithm with multiple initial probabilities
 *
 * Run multiple EM algorithms with different initial probabilities to try to
 * find the global maximum. Using a different initial probability is known as a
 * repetition. Multiple threads can be used to run these repetitions in
 * parallel.
 *
 * How to use:
 * <ul>
 *   <li>
 *     Pass the data (features, responses, initial probabilities) and memory to
 *     store the results via the constructor
 *   </li>
 *   <li>
 *     Call optional methods SetSeed(), set_best_initial_prob() and/or
 *     set_ln_l_array()
 *   </li>
 *   <li>
 *     Call the method Fit<EmAlgorithm>() to run multiple EM algorithms where
 *     the type provided is EmAlgorithm or one of its subclass:
 *       <ul>
 *         <li>EmAlgorithm</li>
 *         <li>EmAlgorithmNan</li>
 *         <li>EmAlgorithmRegress</li>
 *         <li>EmAlgorithmNanRegress</li>
 *       </ul>
 *     Choose a subclass which best describes the problem and the data
 *       <ul>
 *         <li>ie if it is a regression problem or not</li>
 *         <li>ie if the responses contains nan encoded as zeros</li>
 *       </ul>
 *     The resulting fit with the best log-likelihood is recorded.
 *   </li>
 *   <li>
 *     Call the methods get_best_rep_index(), get_n_iter() and/or
 *     get_has_restarted() to get optional information
 *   </li>
 * </ul>
 *
 * It is very possible this could have been implemented using OpenMP instead.
 */
class EmAlgorithmArray {
 private:
  /**
   * Design matrix of features to provide to each EmAlgorithm, matrix with
   * dimensions
   * <ul>
   *   <li>dim 0: for each data point</li>
   *   <li>dim 1: for each feature</li>
   * </ul>
   * Can be empty and not used if using only for the non-regression problem
   */
  std::span<const double> features_;
  /**
   * Design matrix <b>transposed</b> of responses to provide to each
   * EmAlgorithm, matrix containing outcomes/responses for each category as
   * integers 1, 2, 3, .... The matrix has dimensions
   * <ul>
   *   <li>dim 0: for each category</li>
   *   <li>dim 1: for each data point</li>
   * </ul>
   */
  std::span<const int> responses_;
  /** Number of data points */
  const std::size_t n_data_;
  /** Number of features */
  const std::size_t n_feature_;
  /** Vector of the number of outcomes for each category */
  NOutcomes n_outcomes_;
  /** Number of clusters (classes in literature) to fit */
  const std::size_t n_cluster_;
  /** Maximum number of iterations for EM algorithm */
  const unsigned int max_iter_;
  /** To provide to EmAlgorithm */
  const double tolerance_;
  /**
   * To store the posterior result from the best repetition. Accessing and
   * writing should be done with locking and unlocking
   * EmAlgorithmArray::results_lock_ when using multiple threads.
   *
   * Design matrix of posterior probabilities (also called responsibility), the
   * probability a data point is in cluster m given responses, matrix with
   * dimensions
   * <ul>
   *   <li>dim 0: for each data</li>
   *   <li>dim 1: for each cluster</li>
   * </ul>
   */
  std::span<double> posterior_;
  /**
   * To store the prior result from the best repetition. Accessing and writing
   * should be done with locking and unlocking EmAlgorithmArray::results_lock_
   * when using multiple threads.
   *
   * Design matrix of prior probabilities, the probability a data point is in
   * cluster m <b>not</b> given responses
   * <ul>
   *   <li>dim 0: for each data</li>
   *   <li>dim 1: for each cluster</li>
   * </ul>
   */
  std::span<double> prior_;
  /**
   * To store the estimated probabilities from the best repetition. Accessing
   * and writing should be done with locking and unlocking
   * EmAlgorithmArray::results_lock_ when using multiple threads.
   *
   * Vector of estimated response probabilities for each category, flatten list
   * in the following order
   * <ul>
   *   <li>dim 0: for each outcome</li>
   *   <li>dim 1: for each category</li>
   *   <li>dim 2: for each cluster</li>
   * </ul>
   */
  std::span<double> estimated_prob_;
  /**
   * To store the regression coefficients from the best repetition. Accessing
   * and writing should be done with locking and unlocking
   * EmAlgorithmArray::results_lock_ when using multiple threads.
   *
   * Matrix with dimensions:
   * <ul>
   *   <li>dim 0: EmAlgorithmArray::n_feature_</li>
   *   <li>dim 1: EmAlgorithmArray::n_cluster_ - 1</li>
   * </ul>
   * This matrix is multiplied to the feature design matrix and then linked to
   * the prior using softmax. Not used in the non-regression problem
   */
  std::span<double> regress_coeff_;

  /**
   * Optional, to store initial prob to obtain max likelihood or from the best
   * repetition. Accessing and writing should be done with locking and unlocking
   * EmAlgorithmArray::results_lock_ when using multiple threads. It shall be
   * the same format as the member variable with the same name in EmAlgorithm.
   */
  std::optional<std::span<double>> best_initial_prob_;

  /** Number of initial values to try */
  const std::size_t n_rep_;

  /** The best log-likelihood found so far */
  double optimal_ln_l_ = -INFINITY;
  /**
   * Number of iterations the optimal fitter has done. Accessing and writing
   * should be done with locking and unlocking EmAlgorithmArray::results_lock_
   * when using multiple threads.
   */
  std::size_t n_iter_;
  /** True if the EM algorithm has to ever restart */
  bool has_restarted_ = false;
  /**
   * An array of initial probabilities, each repetition uses
   * EmAlgorithmArray::n_outcomes_.sum() * EmAlgorithmArray::n_cluster_
   * probabilities
   */
  std::span<const double> initial_prob_;
  /**
   * The latest initial value is being worked on
   */
  std::atomic<std::size_t> n_rep_done_ = 0;
  /**
   * Optional, maximum log-likelihood for each repetition. Set using
   * set_ln_l_array()
   */
  std::optional<std::span<double>> ln_l_array_;
  /** Index of which initial value has the best log-likelihood */
  std::size_t best_rep_index_;
  /** Number of threads */
  const std::size_t n_thread_;

  /**
   * For locking EmAlgorithmArray::posterior_, EmAlgorithmArray::prior_,
   * EmAlgorithmArray::estimated_prob_, EmAlgorithmArray::regress_coeff_,
   * EmAlgorithmArray::best_initial_prob_,
   * EmAlgorithmArray::optimal_ln_l_, EmAlgorithmArray::n_iter_,
   * EmAlgorithmArray::best_rep_index_ and EmAlgorithmArray::has_restarted_ */
  std::mutex results_lock_;

 protected:
  /**
   * An array of seeds, one for each repetition. It is used to seed the rng for
   * each repetition, it is only used if a run fails and needs to generate new
   * initial values
   */
  std::unique_ptr<std::vector<unsigned>> seed_array_;

 public:
  /**
   * Construct a new EM Algorithm Array object
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
   * @param n_thread Number of threads to use
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
   *   <li>dim 0: EmAlgorithmArray::n_feature_</li>
   *   <li>dim 1: EmAlgorithmArray::n_cluster_ - 1</li>
   * </ul>
   * This matrix is multiplied to the feature design matrix and then linked to
   * the prior using softmax. Not used in the non-regression problem
   */
  EmAlgorithmArray(std::span<const double> features,
                   std::span<const int> responses,
                   std::span<const double> initial_prob, std::size_t n_data,
                   std::size_t n_feature, NOutcomes n_outcomes,
                   std::size_t n_cluster, std::size_t n_rep,
                   std::size_t n_thread, unsigned int max_iter,
                   double tolerance, std::span<double> posterior,
                   std::span<double> prior, std::span<double> estimated_prob,
                   std::span<double> regress_coeff);

  /**
   * Construct a new EM Algorithm Array object
   *
   * Construct a new EM Algorithm Array object for clustering (non-regression)
   * only.
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
   * @param n_thread Number of threads to use
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
  EmAlgorithmArray(std::span<const int> responses,
                   std::span<const double> initial_prob, std::size_t n_data,
                   NOutcomes n_outcomes, std::size_t n_cluster,
                   std::size_t n_rep, std::size_t n_thread,
                   unsigned int max_iter, double tolerance,
                   std::span<double> posterior, std::span<double> prior,
                   std::span<double> estimated_prob);

  virtual ~EmAlgorithmArray() = default;

  /**
   * Fit (in parallel) using the EM algorithm
   *
   * To be called right after construction or after setting optional settings.
   * Results with the best log-likelihood are recorded.
   *
   * Provide the appropriate EmAlgorithm class to use via the template, ie if
   * it is a regression problem or there are NaN numbers (encoded as zero) in
   * the responses. Examples:
   * <ul>
   *   <li>Fit<EmAlgorithm>()</li>
   *   <li>Fit<EmAlgorithmRegress>()</li>
   *   <li>Fit<EmAlgorithmNan>()</li>
   *   <li>Fit<EmAlgorithmNanRegress>()</li>
   * </ul>
   */
  template <typename EmAlgorithmType>
  void Fit();

  /**
   * Set the member variable EmAlgorithmArray::seed_array_
   *
   * Set the member variable EmAlgorithmArray::seed_array_ with a seed for each
   * repetition
   */
  virtual void SetSeed(std::seed_seq& seed);

  /**
   * Set where to store initial probabilities (optional)
   *
   * @param best_initial_prob best_initial_prob to provide to EmAlgorithm
   * objects
   */
  void set_best_initial_prob(std::span<double> best_initial_prob);

  /** Set where to store the log-likelihood for each iteration */
  void set_ln_l_array(std::span<double> ln_l_array);

  /**
   * Get the index of the repetition with the highest log-likelihood
   *
   * Only available after calling Fit()
   */
  [[nodiscard]] std::size_t get_best_rep_index() const;

  /**
   * Get the best log-likelihood from all repetitions
   *
   * Only available after calling Fit()
   */
  [[nodiscard]] double get_optimal_ln_l() const;

  /**
   * Get the number of EM iterations done for the repetition with the highest
   * log-likelihood
   *
   * Only available after calling Fit()
   */
  [[nodiscard]] unsigned int get_n_iter() const;

  /**
   * Return true if at least one repetition had to restart, eg due to a singular
   * matrix
   *
   * Only available after calling Fit()
   */
  [[nodiscard]] bool get_has_restarted() const;

 protected:
  /** Set the rng of a EmAlgorithm object given the rep_index it is working on*/
  virtual void SetFitterRng(std::size_t rep_index,
                            polca_parallel::EmAlgorithm& fitter);

  /**
   * Retrieve ownership of an rng back from a fitter
   */
  virtual void MoveRngBackFromFitter(polca_parallel::EmAlgorithm& fitter);

 private:
  /**
   * For each initial probability, fit using the EM algorithm
   *
   * Each thread calls this, each repeatedly instantiates EmAlgorithm and calls
   * Fit(). When a better log-likelihood is found after the fit, it copies the
   * results over, such as the posterior, prior, estimate probabilities,
   * regression coefficients and starting probabilities.
   *
   * Provide the appropriate EmAlgorithm class to use via the template, ie if
   * it is a regression problem or there are NaN numbers (encoded as zero) in
   * the responses. Examples:
   * <ul>
   *   <li>FitThread<EmAlgorithm>()</li>
   *   <li>FitThread<EmAlgorithmRegress>()</li>
   *   <li>FitThread<EmAlgorithmNan>()</li>
   *   <li>FitThread<EmAlgorithmNanRegress>()</li>
   * </ul>
   */
  template <typename EmAlgorithmType>
  void FitThread();
};

}  // namespace polca_parallel

#endif  // POLCAPARALLEL_INCLUDE_EM_ALGORITHM_ARRAY_H_
