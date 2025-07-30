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

#ifndef POLCAPARALLEL_INCLUDE_EM_ALGORITHM_REGRESS_H_
#define POLCAPARALLEL_INCLUDE_EM_ALGORITHM_REGRESS_H_

#include <cstddef>
#include <random>
#include <span>

#include "arma.h"
#include "em_algorithm.h"
#include "util.h"

namespace polca_parallel {

/**
 * For fitting the poLCA regression problem using the EM algorithm
 *
 * For fitting the poLCA regression problem using the EM algorithm. In the
 * regression problem, prior probabilities are obatined from the softmax
 * functions of the features.
 *
 * Provide initial outcome probabilities to initalise the EM algorithm
 *
 * How to use:
 * <ul>
 *   <li>
 *     Pass the data, initial probabilities and other parameters to the
 *     constructor. Also in the constructor, pass an array to store the
 *     posterior and prior probabilities (for each cluster), estimated
 *     outcome probabilities and the regression coefficients
 *   </li>
 *   <li>
 *     Call optional methods such as set_best_initial_prob(), set_seed()
 *     and/or set_rng()
 *   </li>
 *   <li>
 *      Call Fit() to fit using the EM algorithm, results are stored in the
 *      provided arrays. The EM algorithm restarts with random initial values
 *      should it fail for some reason (more commonly in the regression model)
 *   </li>
 *   <li>
 *     Extract optional results using the methods get_ln_l(), get_n_iter()
 *     and/or get_has_restarted()
 *   </li>
 * </ul>
 */
class EmAlgorithmRegress : public polca_parallel::EmAlgorithm {
 private:
  /**
   * If the log likelihood decreases by less than this, flag the log likelihood
   * as invalid to stop the EM algorithm. Negative value means the log
   * likelihood increases
   */
  static constexpr double kMinLogLikelihoodDifference = -1e-7;

 private:
  /**
   * Design matrix of features, matrix with dimensions
   * <ul>
   *   <li>dim 0: for each data point</li>
   *   <li>dim 1: for each feature</li>
   * </ul>
   */
  const arma::Mat<double> features_;
  /** Number of features */
  const std::size_t n_feature_;
  /**
   * Matrix of coefficients, to be multiplied to the features and linked to the
   * prior using softmax. The dimensions are of size
   * <ul>
   *   <li>dim 0: EmAlgorithmRegress::n_feature_</li>
   *   <li>dim 1: EmAlgorithmRegress::n_cluster_ - 1</li>
   * </ul>
   */
  arma::Mat<double> regress_coeff_;
  /** Number of parameters to estimate for the softmax */
  const std::size_t n_parameters_;
  /**
   * Vector of length EmAlgorithmRegress::n_parameters_
   *
   * Gradient of the log likelihood
   */
  arma::Col<double> gradient_;
  /**
   * Hessian of the log likelihood with dimensions
   * <ul>
   *   <li>dim 0: EmAlgorithmRegress::n_parameters_</li>
   *   <li>dim 1: EmAlgorithmRegress::n_parameters_</li>
   * </ul>
   */
  arma::Mat<double> hessian_;

 public:
  /**
   * Construct a new EM algorithm regression object
   *
   * The following content pointed to shall be modified:
   * <ul>
   *   <li><code>posterior</code></li>
   *   <li><code>prior</code></li>
   *   <li><code>estimated_prob</code></li>
   *   <li><code>regress_coeff</code></li>
   * </ul>
   *
   * @param features Design matrix of features, matrix with dimensions:
   * <ul>
   *   <li>dim 0: for each data point</li>
   *   <li>dim 1: for each feature</li>
   * </ul>
   * @param responses Design matrix <b>transposed</b> of responses, matrix
   * containing outcomes/responses for each category as integers 1, 2, 3, ....
   * The matrix has dimensions
   * <ul>
   *   <li>dim 0: for each category</li>
   *   <li>dim 1: for each data point</li>
   * </ul>
   * @param initial_prob Vector of initial probabilities for each category and
   * outcome, flatten list in the following order
   * <ul>
   *   <li>dim 0: for each outcome</li>
   *   <li>dim 1: for each category</li>
   *   <li>dim 2: for each cluster</li>
   * </ul>
   * @param n_data Number of data points
   * @param n_feature Number of features
   * @param n_outcomes Vector of number of outcomes for each category and its
   * sum
   * @param n_cluster Number of clusters to fit
   * @param max_iter Maximum number of iterations for EM algorithm
   * @param tolerance Tolerance for difference in log likelihood, used for
   * stopping condition
   * @param posterior Modified to contain the resulting posterior probabilities
   * after calling Fit(). Design matrix of posterior probabilities (also called
   * responsibility). It's the probability a data point is in cluster m given
   * responses. The matrix has the following dimensions
   * <ul>
   *   <li>dim 0: for each data</li>
   *   <li>dim 1: for each cluster</li>
   * </ul>
   * @param prior Modified to contain the resulting prior probabilities after
   * calling Fit(). Design matrix of prior probabilities. It's the probability a
   * data point is in cluster m <b>not</b> given responses after calculations.
   * The matrix has the following dimensions
   * <ul>
   *   <li>dim 0: for each data</li>
   *   <li>dim 1: for each cluster</li>
   * </ul>
   * @param estimated_prob Modified to contain the resulting outcome
   * probabilities after calling Fit(). Vector of estimated response
   * probabilities, conditioned on cluster, for each category. A flatten list in
   * the following order
   * <ul>
   *   <li>dim 0: for each outcome</li>
   *   <li>dim 1: for each category</li>
   *   <li>dim 2: for each cluster</li>
   * </ul>
   * @param regress_coeff Modified to contain the resulting matrix of
   * coefficients. To be multiplied to the features and linked to the prior
   * using softmax. The dimensions are of size
   * <ul>
   *   <li>dim 0: EmAlgorithmRegress::n_feature_</li>
   *   <li>dim 1: EmAlgorithmRegress::n_cluster_ - 1</li>
   * </ul>
   */
  EmAlgorithmRegress(std::span<const double> features,
                     std::span<const int> responses,
                     std::span<const double> initial_prob, std::size_t n_data,
                     std::size_t n_feature, NOutcomes n_outcomes,
                     std::size_t n_cluster, unsigned int max_iter,
                     double tolerance, std::span<double> posterior,
                     std::span<double> prior, std::span<double> estimated_prob,
                     std::span<double> regress_coeff);

  ~EmAlgorithmRegress() override = default;

 protected:
  /**
   * Reset parameters for a re-run
   *
   * Reset the parameters EmAlgorithm::estimated_prob_ with random starting
   * values and EmAlgorithmRegress::regress_coeff_ all to zero
   */
  void Reset() override;

  void InitPrior() override;

  /**
   * Adjust prior return value to matrix format
   *
   * Does nothing, the member variable EmAlgorithmRegress::prior_ is already
   * in the correct matrix format
   */
  void FinalPrior() override;

  [[nodiscard]] double GetPrior(std::size_t data_index,
                                std::size_t cluster_index) const override;

  [[nodiscard]] bool IsInvalidLikelihood(double ln_l_difference) const override;

  /**
   * Do M step
   *
   * Update the regression coefficient, prior probabilities and
   * estimated response probabilities given the posterior probabilities.
   * Modifies the member variables EmAlgorithmRegress::regress_coeff_,
   * EmAlgorithmRegress::gradient_, EmAlgorithmRegress::hessian_,
   * EmAlgorithmRegress::prior_ and EmAlgorithm::estimated_prob_
   *
   * @return <code>true</code> if the solver cannot find a solution,
   * <code>false</code> if successful
   */
  bool MStep() override;

  void NormalWeightedSumProb(const std::size_t cluster_index) override;

 private:
  /** Initalise EmAlgorithmRegress::regress_coeff_ to all zeros */
  void init_regress_coeff();

  /**
   * Calculate gradient of the log likelihood
   *
   * Updates the member variable EmAlgorithmRegress::gradient_
   */
  void CalcGrad();

  /**
   * Calculate hessian of the log likelihood
   *
   * Updates the member variable EmAlgorithmRegress::hessian_
   */
  void CalcHess();

  /**
   * Calculate one of the blocks of the Hessian
   *
   * Updates the member variable EmAlgorithmRegress::hessian_ with one of the
   * blocks. The hessian consists of (EmAlgorithmRegress::n_cluster_ - 1) by
   * (EmAlgorithmRegress::n_cluster_ - 1) blocks, each corresponding to cluster
   * 1, 2, 3, ..., EmAlgorithmRegress::n_cluster_ - 1
   *
   * @param cluster_index_0 row index of which block to work on
   * can take values of 0, 1, 2, ..., EmAlgorithmRegress::n_cluster_ - 2
   * @param cluster_index_1 column index of which block to work on
   * can take values of 0, 1, 2, ..., EmAlgorithmRegress::n_cluster_ - 2
   */
  void CalcHessSubBlock(std::size_t cluster_index_0,
                        std::size_t cluster_index_1);

  /**
   * Calculate element of a block from the Hessian
   *
   * @param feature_index_0 column index
   * @param feature_index_1 column index
   * @param prior_post_inter Vector of length EmAlgorithm::n_data_, dependent on
   * pair of clusters. For cluster \f$m\f$, let \f$r_m\f$ be a vector of
   * posteriors and \f$\pi_m\f$ be a vector of priors for cluster \f$m\f$. For
   * a cluster pair \f$u\f$ and \f$v\f$, the argument should be:
   * <ul>
   *   <li>For \f$u=v\f$: \f$r_u(1-r_u) - \pi_u(1-\pi_u)\f$</li>
   *   <li>Otherwise: \f$\pi_u \pi_v - r_u r_v\f$</li>
   * </ul>
   * @return double value of an element of the Hessian
   */
  [[nodiscard]] double CalcHessElement(
      std::size_t feature_index_0, std::size_t feature_index_1,
      const arma::Col<double>& prior_post_inter);

  /**
   * Assign an entry of the Hessian at specificed indexes
   *
   * Hessian is a block matrix, each rows/columns of block matrices correspond
   * to a cluster, and then each row/column of the block matrix correspond
   * to a feature. Use this method to assign a specific element of the hessian
   * matrix
   *
   * @param hess_element value of an entry of the Hessian
   * @param cluster_index_0 row index of block matrices
   * @param cluster_index_1 column index of block matrices
   * @param feature_index_0 row index within block matrix
   * @param feature_index_1 column index within block matrix
   */
  void AssignHessianAt(double hess_element, std::size_t cluster_index_0,
                       std::size_t cluster_index_1, std::size_t feature_index_0,
                       std::size_t feature_index_1);
};

}  // namespace polca_parallel

#endif  // POLCAPARALLEL_INCLUDE_EM_ALGORITHM_REGRESS_H_
