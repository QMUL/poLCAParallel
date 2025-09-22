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

#ifndef POLCAPARALLEL_INCLUDE_EM_ALGORITHM_H_
#define POLCAPARALLEL_INCLUDE_EM_ALGORITHM_H_

#include <cstddef>
#include <memory>
#include <optional>
#include <random>
#include <span>

#include "arma.h"
#include "util.h"

namespace polca_parallel {

/**
 * For fitting poLCA using the EM algorithm
 *
 * Provide initial outcome probabilities to initalise the EM algorithm
 *
 * How to use:
 * <ul>
 *   <li>
 *     Pass the data, initial probabilities and other parameters to the
 *     constructor. Also in the constructor, pass an array to store the
 *     posterior and prior probabilities (for each cluster), estimated
 *     outcome probabilities
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
class EmAlgorithm {
 protected:
  /**
   * Design matrix <b>transposed</b> of responses, matrix containing
   * outcomes/responses for each category as integers 1, 2, 3, .... The matrix
   * has dimensions
   * <ul>
   *   <li>dim 0: for each category</li>
   *   <li>dim 1: for each data point</li>
   * </ul>
   */
  std::span<const int> responses_;
  /**
   * Vector of initial probabilities for each category and responses, flatten
   * list in the following order
   * <ul>
   *   <li>dim 0: for each outcome</li>
   *   <li>dim 1: for each category</li>
   *   <li>dim 2: for each cluster</li>
   * </ul>
   */
  std::span<const double> initial_prob_;
  /** Number of data points */
  const std::size_t n_data_;
  /** Vector of the number of outcomes for each category */
  NOutcomes n_outcomes_;
  /** Number of clusters to fit */
  const std::size_t n_cluster_;
  /** Maximum number of iterations for EM algorithm */
  const unsigned int max_iter_;
  /** Tolerance for difference in log-likelihood, used for stopping condition */
  const double tolerance_;
  /**
   * Design matrix of posterior probabilities (also called responsibility). It's
   * the probability a data point is in cluster m given responses. The matrix
   * has the following dimensions
   * <ul>
   *   <li>dim 0: for each data</li>
   *   <li>dim 1: for each cluster</li>
   * </ul>
   */
  arma::Mat<double> posterior_;
  /**
   * Design matrix of prior probabilities
   *
   * The probability a data point is in cluster m <b>not</b> given responses
   * after calculations. The matrix has the following dimensions
   * <ul>
   *   <li>dim 0: for each data</li>
   *   <li>dim 1: for each cluster</li>
   * </ul>
   *
   * During the start and calculations, it may take on a different form
   * depending on the class implementation. For example, in EmAlgorithm, the
   * prior probability is the same for all data points. In EmAlgorithmRegress,
   * the prior probability is different for all data points. Thus in
   * EmAlgorithm, only the first m elements are used.
   *
   * Use the method GetPrior() to get the prior for a data point and cluster
   * rather than accessing the member variable direction during Fit()
   */
  arma::Mat<double> prior_;
  /**
   * Matrix of estimated response probabilities, conditioned on cluster, for
   * each category. A flattened list in the following order
   * <ul>
   *   <li>
   *     dim 0: for each outcome | category (inner), for each category (outer)
   *   </li>
   *   <li>dim 1: for each cluster</li>
   * </ul>
   */
  arma::Mat<double> estimated_prob_;
  /**
   * Optional, vector of <b>initial</b> response probabilities used to get the
   * maximum log-likelihood, this member variable is optional. Set using
   * set_best_initial_prob()
   *
   * A flattened list in the following order
   * <ul>
   *   <li>dim 0: for each outcome</li>
   *   <li>dim 1: for each category</li>
   *   <li>dim 2: for each cluster</li>
   * </ul>
   */
  std::optional<std::span<double>> best_initial_prob_;

  /** Log likelihood, updated at each iteration of EM */
  double ln_l_ = -INFINITY;
  /**
   * Vector, for each data point, the log-likelihood for each data point, the
   * total log-likelihood is the sum
   */
  arma::Col<double> ln_l_array_;
  /** Number of iterations done right now */
  unsigned int n_iter_ = 0;
  /**
   * Indicate if it needed to use new initial values during a fit, which can
   * happen if a matrix is singular for example
   */
  bool has_restarted_ = false;
  /** Random number generator for generating new initial values if fail */
  std::unique_ptr<std::mt19937_64> rng_;

 public:
  /**
   * Construct a new EM algorithm object
   *
   * The following content pointed to shall be modified:
   * <ul>
   *   <li><code>posterior</code></li>
   *   <li><code>prior</code></li>
   *   <li><code>estimated_prob</code></li>
   * </ul>
   *
   * Some parameters are ignored. This is designed so that the signature is the
   * same as the subclasses such as polca_parallel::EmAlgorithmRegress
   *
   * @param features Not used and ignored
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
   * @param n_feature Number of features, set this to 1
   * @param n_outcomes Vector of number of outcomes for each category and its
   * sum
   * @param n_cluster Number of clusters to fit
   * @param max_iter Maximum number of iterations for EM algorithm
   * @param tolerance Tolerance for difference in log-likelihood, used for
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
   * data point is in cluster <code>m</code> <b>not</b> given responses after
   * calculations. The matrix has the following dimensions
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
   * @param regress_coeff Not used and ignored
   */
  EmAlgorithm(std::span<const double> features, std::span<const int> responses,
              std::span<const double> initial_prob, std::size_t n_data,
              std::size_t n_feature, NOutcomes n_outcomes,
              std::size_t n_cluster, unsigned int max_iter, double tolerance,
              std::span<double> posterior, std::span<double> prior,
              std::span<double> estimated_prob,
              std::span<double> regress_coeff);

  /**
   * Construct a new EM algorithm object
   *
   * Construct a new EM algorithm object for the non-regression problem only
   *
   * The following content pointed to shall be modified:
   * <ul>
   *   <li><code>posterior</code></li>
   *   <li><code>prior</code></li>
   *   <li><code>estimated_prob</code></li>
   * </ul>
   *
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
   * @param n_outcomes Vector of number of outcomes for each category and its
   * sum
   * @param n_cluster Number of clusters to fit
   * @param max_iter Maximum number of iterations for EM algorithm
   * @param tolerance Tolerance for difference in log-likelihood, used for
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
   * data point is in cluster <code>m</code> <b>not</b> given responses after
   * calculations. The matrix has the following dimensions
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
   */
  EmAlgorithm(std::span<const int> responses,
              std::span<const double> initial_prob, std::size_t n_data,
              NOutcomes n_outcomes, std::size_t n_cluster,
              unsigned int max_iter, double tolerance,
              std::span<double> posterior, std::span<double> prior,
              std::span<double> estimated_prob);

  virtual ~EmAlgorithm() = default;

  /**
   * Fit model to data using EM algorithm
   *
   * Data is provided through the constructor, important results are stored in
   * the member variables:
   * <ul>
   *   <li>EmAlgorithm::posterior_</li>
   *   <li>EmAlgorithm::prior_</li>
   *   <li>EmAlgorithm::estimated_prob_</li>
   *   <li>EmAlgorithm::ln_l_</li>
   *   <li>EmAlgorithm::ln_l_array_</li>
   *   <li>EmAlgorithm::n_iter_</li>
   *   <li>EmAlgorithm::has_restarted_</li>
   *   <li>optionally, EmAlgorithm::best_initial_prob_</li>
   * </ul>
   */
  void Fit();

  /**
   * Set where to store initial probabilities (optional)
   *
   * @param best_initial_prob Vector to store the <b>initial</b> response
   * probabilities used to get the maximum log-likelihood, flatten list in the
   * following order
   * <ul>
   *   <li>dim 0: for each outcome</li>
   *   <li>dim 1: for each category</li>
   *   <li>dim 2: for each cluster</li>
   * </ul>
   */
  void set_best_initial_prob(std::span<double> best_initial_prob);

  /** Get the log-likelihood */
  [[nodiscard]] double get_ln_l() const;

  /** Get the number of iterations of EM done */
  [[nodiscard]] unsigned int get_n_iter() const;

  /**
   * Indicate if it was needed to use new initial values during a fit, it can
   * happen if a matrix is singular in the regression problem for example
   */
  [[nodiscard]] bool get_has_restarted() const;

  /**
   * Set rng using a seed, for generating new random initial values
   *
   * Set EmAlgorithm::rng_ using a seed, for generating new random initial
   * values
   */
  void set_seed(unsigned seed);

  /**
   * Set rng by transferring ownership of an rng to this object
   *
   * Set EmAlgorithm::rng_ by transferring ownership of an rng to this object
   *
   * Use this method if you want to use your own rng instead of the default
   * rng
   */
  void set_rng(std::unique_ptr<std::mt19937_64> rng);

  /**
   * Transfer ownership of EmAlgorithm::rng_ from this object
   *
   * Retrive the rng passed in set_rng()
   * */
  [[nodiscard]] std::unique_ptr<std::mt19937_64> move_rng();

 protected:
  /**
   * Reset parameters for a re-run
   *
   * Reset the parameters EmAlgorithm::estimated_prob_ with random starting
   * values
   */
  virtual void Reset();

  /**
   * Initialise prior probabilities
   *
   * Initialise the content of EmAlgorithm::prior_ which contains prior
   * probabilities for each cluster, ready for the EM algorithm
   */
  virtual void InitPrior();

  /**
   * Adjust prior return value to matrix format
   *
   * Adjust the member variable EmAlgorithm::prior_ so that it is in matrix
   * format. During Fit(), EmAlgorithm::prior_ is a vector of length
   * EmAlgorithm::n_cluster_ as it assumes each data point has the same prior.
   * Use this method to duplicate the prior for each data point so that the
   * member variable EmAlgorithm::prior_ becomes a matrix with the following
   * dimensions:
   *
   * <ul>
   *   <li>dim 0: for each data point
   *   <li>dim 1: for each cluster
   * </ul>
   */
  virtual void FinalPrior();

  /**
   * Get prior, for a specified data point and cluster, during the EM algorithm
   *
   * @param data_index
   * @param cluster_index
   * @return double prior
   */
  [[nodiscard]] virtual double GetPrior(const std::size_t data_index,
                                        const std::size_t cluster_index) const;

  /**
   * Do E step
   *
   * Update the posterior probabilities and log likelihood given the prior
   * probabilities and estimated response probabilities. Modifies the member
   * variables EmAlgorithm::posterior_ and EmAlgorithm::ln_l_array_
   */
  void EStep();

  /**
   * Calculates the unnormalize posterior and set it to EmAlgorithm::posterior_
   *
   * Calculates the unnormalize posterior for a given cluster and assign it to
   * EmAlgorithm::posterior_. See polca_parallel::PosteriorUnnormalize() for
   * further information
   *
   * @param responses_i vector of responses for a given cluster
   * @param prior prior for a given cluster
   * @param estimated_prob the corresponding cluster's column view of
   * EmAlgorithm::estimated_prob_. A flattened list in the following order
   * <ul>
   *   <li>dim 0: for each outcome</li>
   *   <li>dim 1: for each category</li>
   * </ul>
   */
  [[nodiscard]] virtual double PosteriorUnnormalize(
      std::span<const int> responses_i, double prior,
      const arma::Col<double>& estimated_prob) const;

  /**
   * Check if the likelihood is invalid
   *
   * @param ln_l_difference the change in log-likelihood after an iteration of
   * EM
   * @return <code>true</code> if the likelihood is invalid, <code>false</code>
   * if the likelihood is okay
   */
  [[nodiscard]] virtual bool IsInvalidLikelihood(double ln_l_difference) const;

  /**
   * Do M step
   *
   * Update the prior probabilities and estimated response probabilities given
   * the posterior probabilities. Modifies the member variables
   * EmAlgorithm::prior_ and EmAlgorithm::estimated_prob_
   *
   * @return <code>false</code>
   */
  virtual bool MStep();

  /**
   * Estimate outcome probability
   *
   * Updates and modifies the member variable EmAlgorithm::estimated_prob_ using
   * the posterior
   */
  void EstimateProbability();

  /**
   * Weighted sum of posteriors for a given cluster
   *
   * Update a column of the member variable EmAlgorithm::estimated_prob_ with a
   * weighted sum. For a given cluster, category and outcome, this weighted sum
   * is of posteriors over all data points, where the weight is a boolean
   * whether a response has been observed or not.
   *
   * The weighted sum can also be viewed as a selective sum of posteriors.
   *
   * The weighted sum is as follows, for a given cluster, category and outcome:
   *
   * Sum over data (<code>i = 0</code> to <code>n-1</code>) the posterior(for
   * data <code>i</code> and a given cluster) multiplied by boolean (has the
   * <code>i</code>th response for this category and outcome been observed)
   *
   * \f[
   * \sum_{i=0}^{n-1} \theta^{(i)}_m \delta[y^{(i)}_{j}, k_j]
   * \f]
   *
   * where \f$n\f$ is the number of data points, \f$j\f$ is the category,
   * \f$k_j\f$ is an outcome in category \f$j\f$, \f$\theta^{(i)}_m\f$ is the
   * posterior for data point \f$i\f$ and cluster \f$m\f$, \f$y^{(i)}_{j}\f$ is
   * the response for the  \f$i\f$ th data point and \f$j\f$th category,
   * \f$\delta\f$ is the Kronecker delta.
   *
   * This is done for all categories and outcomes for a given cluster
   *
   * This is used to estimate outcome probabilities later on
   *
   * @param cluster_index which cluster to consider
   */
  virtual void WeightedSumProb(const std::size_t cluster_index);

  /**
   * Normalise the weighted sum following WeightedSumProb()
   *
   * After calling WeightedSumProb(), call this to normalise the weighted sum so
   * that the member variable EmAlgorithm::estimated_prob_ contains estimated
   * probabilities for each outcome for a given cluster.
   *
   * Can be overridden as the sum of weights can be calculated differently.
   *
   * @param cluster_index which cluster to consider
   */
  virtual void NormalWeightedSumProb(const std::size_t cluster_index);

  /**
   * Normalise the weighted sum following WeightedSumProb() given the normaliser
   *
   * After calling WeightedSumProb(), call this to normalise the weighted sum in
   * EmAlgorithm::estimated_prob_ by a provided normaliser. Each probability for
   * a given cluster is divided by this normaliser.
   *
   * @param cluster_index which cluster to consider
   * @param normaliser the scale to divide the weighted sum by, should be the
   * sum of posteriors
   */
  void NormalWeightedSumProb(const std::size_t cluster_index,
                             double normaliser);
};

/**
 * Calculates the unnormalize posterior, that is likelihood multiplied by
 * prior
 *
 * Calculates the unnormalize posterior, that is likelihood multiplied by
 * prior for a given data point and cluster. This corresponds to the
 * probability that this data point belongs to a given cluster given the
 * responses and outcome probabilities, up to a constant.
 *
 * The likelihood is the product of outcome probabilities (or estimated in the
 * EM algorithm) which corresponds to the outcome responses.
 *
 * The prior (of the cluster) is given.
 *
 * It should be noted in the likelihood calculations, probabilities are
 * iteratively multiplied. To avoid underflow errors, a sum of log
 * probabilities should be done instead, however it was noted that the sum of
 * logs is slower.
 *
 * @tparam is_check_zero to check if the responses are zero or not, for
 * performance reason, use false when the responses do not contain zero values
 * @param responses_i the responses for a given data point, length n_catgeory
 * @param n_outcomes number of outcomes for each category
 * @param estimated_prob A column view of estimated_prob_. A flattened
 * list in the following order
 * <ul>
 *   <li>dim 0: for each outcome</li>
 *   <li>dim 1: for each category</li>
 * </ul>
 * @param prior the prior for this data point and cluster
 * @return the unnormalised posterior for this data point and cluster
 */
template <bool is_check_zero = false>
[[nodiscard]] double PosteriorUnnormalize(
    std::span<const int> responses_i, std::span<const std::size_t> n_outcomes,
    const arma::Col<double>& estimated_prob, double prior);

}  // namespace polca_parallel

#endif  // POLCAPARALLEL_INCLUDE_EM_ALGORITHM_H_
