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

#ifndef POLCAPARALLEL_SRC_EM_ALGORITHM_H_
#define POLCAPARALLEL_SRC_EM_ALGORITHM_H_

#include <memory>
#include <random>
#include <vector>

namespace polca_parallel {

/**
 * When multiplying probabilities together in PosteriorUnnormalize(), use the
 * sum of logs instead when the resulting probability is less than this
 * threshold
 **/
extern const int kUnderflowThreshold;

/**
 * For fitting poLCA using the EM algorithm for a given initial value
 *
 * How to use:
 * <ul>
 *   <li>
 *     Pass the data, initial probabilities and other parameters to the
 *     constructor. Also in the constructor, pass an array to store the
 *     posterior and prior probabilities (for each cluster) and the estimated
 *     response probabilities
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
   * Design matrix of features, matrix with dimensions
   * <ul>
   *   <li>dim 0: for each data point</li>
   *   <li>dim 1: for each feature</li>
   * </ul>
   */
  double* features_;
  /**
   * Design matrix TRANSPOSED of responses, matrix containing outcomes/responses
   * for each category as integers 1, 2, 3, .... The matrix has dimensions
   * <ul>
   *   <li>dim 0: for each category</li>
   *   <li>dim 1: for each data point</li>
   * </ul>
   */
  int* responses_;
  /**
   * Vector of initial probabilities for each category and responses, flatten
   * list in the following order
   * <ul>
   *   <li>dim 0: for each outcome</li>
   *   <li>dim 1: for each category</li>
   *   <li>dim 2: for each cluster</li>
   * </ul>
   */
  double* initial_prob_;
  /** Number of data points */
  int n_data_;
  /** Number of features */
  int n_feature_;
  /** Number of categories */
  int n_category_;
  /** Vector of the number of outcomes for each category */
  int* n_outcomes_;
  /** Sum of n_outcomes */
  int sum_outcomes_;
  /** Number of clusters to fit */
  int n_cluster_;
  /** Maximum number of iterations for EM algorithm */
  int max_iter_;
  /** Tolerance for difference in log-likelihood, used for stopping condition */
  double tolerance_;
  /**
   * Design matrix of posterior probabilities (also called responsibility). It's
   * the probability a data point is in cluster m given responses. The matrix
   * has the following dimensions
   * <ul>
   *   <li>dim 0: for each data</li>
   *   <li>dim 1: for each cluster</li>
   * </ul>
   */
  double* posterior_;
  /**
   * Design matrix of prior probabilities. It's the probability a data point is
   * in cluster m NOT given responses after calculations. The matrix has the
   * following dimensions
   * <ul>
   *   <li>dim 0: for each data</li>
   *   <li>dim 1: for each cluster</li>
   * </ul>
   * During the start and calculations, it may take on a different form. Use the
   * method GetPrior() to get the prior for a data point and cluster rather than
   * accessing the member variable direction
   */
  double* prior_;
  /**
   * Vector of estimated response probabilities, conditioned on cluster, for
   * each category. A flattened list in the following order
   * <ul>
   *   <li>dim 0: for each outcome</li>
   *   <li>dim 1: for each category</li>
   *   <li>dim 2: for each cluster</li>
   * </ul>
   */
  double* estimated_prob_;
  /**
   * Vector length n_features_*(n_cluster-1), linear regression coefficient
   * in matrix form, to be multiplied to the features and linked to the
   * prior using softmax
   */
  double* regress_coeff_;
  /**
   * Optional, vector of INITIAL response probabilities used to get the maximum
   * log-likelihood, this member variable is optional, set to NULL if not used.
   * Set using set_best_initial_prob()
   * A flattened list in the following order
   * <ul>
   *   <li>dim 0: for each outcome</li>
   *   <li>dim 1: for each category</li>
   *   <li>dim 2: for each cluster</li>
   * </ul>
   */
  double* best_initial_prob_ = nullptr;

  /** Log likelihood, updated at each iteration of EM */
  double ln_l_ = -INFINITY;
  /**
   * Vector, for each data point, the log-likelihood for each data point, the
   * total log-likelihood is the sum
   */
  std::vector<double> ln_l_array_;
  /** Number of iterations done right now */
  int n_iter_ = 0;
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
   * Please see the description of the member variables for further information.
   * The following content pointed to shall be modified:
   * <ul>
   *   <li>posterior</li>
   *   <li>prior</li>
   *   <li>estimated_prob</li>
   * </ul>
   *
   * @param features Not used and ignored
   * @param responses Design matrix TRANSPOSED of responses, matrix containing
   * outcomes/responses for each category as integers 1, 2, 3, .... The matrix
   * has dimensions
   * <ul>
   *   <li>dim 0: for each category</li>
   *   <li>dim 1: for each data point</li>
   * </ul>
   * @param initial_prob Vector of initial probabilities for each category and
   * outcome. This can be nullptr and set later using the method NewRun().
   * Flatten list in the following order
   * <ul>
   *   <li>dim 0: for each outcome</li>
   *   <li>dim 1: for each category</li>
   *   <li>dim 2: for each cluster</li>
   * </ul>
   * @param n_data Number of data points
   * @param n_feature Number of features
   * @param n_category Number of categories
   * @param n_outcomes Vector of number of outcomes for each category
   * @param sum_outcomes Sum of n_outcomes
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
   * data point is in cluster m NOT given responses after calculations. The
   * matrix has the following dimensions dimensions
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
  EmAlgorithm(double* features, int* responses, double* initial_prob,
              int n_data, int n_feature, int n_category, int* n_outcomes,
              int sum_outcomes, int n_cluster, int max_iter, double tolerance,
              double* posterior, double* prior, double* estimated_prob,
              double* regress_coeff);

  /**
   * Fit data to model using EM algorithm
   *
   * Data is provided through the constructor, important results are stored in
   * the member variables:
   * <ul>
   *   <li>posterior_</li>
   *   <li>prior_</li>
   *   <li>estimated_prob_</li>
   *   <li>ln_l_array_</li>
   *   <li>ln_l_</li>
   *   <li>n_iter_</li>
   *   <li>optionally, best_initial_prob_</li>
   * </ul>
   */
  void Fit();

  /**
   * Reset this object so that it can be re-used for another run with new
   * initial probabilities
   *
   * @param initial_prob Vector of initial probabilities for each category and
   * outcome, flatten list in the following order
   * <ul>
   *   <li>dim 0: for each outcome</li>
   *   <li>dim 1: for each category</li>
   *   <li>dim 2: for each cluster</li>
   * </ul>
   */
  virtual void NewRun(double* initial_prob);

  /**
   * Set where to store initial probabilities (optional)
   *
   * @param best_initial_prob Vector of INITIAL response probabilities used to
   * get the maximum log-likelihood, flatten list in the following order
   * <ul>
   *   <li>dim 0: for each outcome</li>
   *   <li>dim 1: for each category</li>
   *   <li>dim 2: for each cluster</li>
   * </ul>
   */
  void set_best_initial_prob(double* best_initial_prob);

  /** Get the log-likelihood */
  double get_ln_l();

  /** Get the number of iterations of EM done */
  int get_n_iter();

  /**
   * Indicate if it needed to use new initial values during a fit, it can happen
   * if a matrix is singular for example
   */
  bool get_has_restarted();

  /** Set rng using a seed, for generating new random initial values */
  void set_seed(unsigned seed);

  /**
   * Set rng by transferring ownership of an rng to this object
   *
   * Use this method if you want to use your own rng instead of the default
   * rng
   */
  void set_rng(std::unique_ptr<std::mt19937_64>* rng);

  /**
   * Transfer ownership of rng from this object
   *
   * Use this method if you want to ensure the rng you pass in set_rng() lives
   * when this object goes out of scope
   * */
  std::unique_ptr<std::mt19937_64> move_rng();

 protected:
  /**
   * Reset parameters for a re-run
   *
   * Reset the parameters estimated_prob_ with random starting values
   * @param uniform required to generate random probabilities
   */
  virtual void Reset(std::uniform_real_distribution<double>* uniform);

  /**
   * Initialise prior probabilities
   *
   * Initialise the content of prior_ which contains prior probabilities for
   * each cluster, ready for the EM algorithm
   */
  virtual void InitPrior();

  /** Adjust prior return value to matrix format */
  virtual void FinalPrior();

  /**
   * Get prior, for a specified data point and cluster, during the EM algorithm
   *
   * @param data_index
   * @param cluster_index
   * @return double prior
   */
  virtual double GetPrior(int data_index, int cluster_index);

  /**
   * Do E step
   *
   * Update the posterior probabilities given the prior probabilities and
   * estimated response probabilities. Modifies the member variables posterior_
   * and ln_l_array_. Calculations from the E step also provide the elements
   * for ln_l_array_.
   */
  void EStep();

  /**
   * Calculates the unnormalize posterior and assign it to posterior_
   *
   * Calculates the unnormalize posterior and asign it to posterior_. See
   * PosteriorUnnormalize() for further information.
   *
   * @param data_index data point index 0, 1, 2, ..., n_data - 1
   * @param cluster_index cluster index 0, 1, 2, ..., n_cluster - 1
   * @param estimated_prob pointer to estimated probabilities for the
   * corresponding cluster. This is modified to point to the probabilities for
   * the next cluster. It points to a flattened list in the following order
   * <ul>
   *   <li>dim 0: for each outcome</li>
   *   <li>dim 1: for each category</li>
   * </ul>
   */
  void PosteriorUnnormalize(int data_index, int cluster_index,
                            double** estimated_prob);

  /**
   * Check if the likelihood is invalid
   *
   * @param ln_l_difference the change in log-likelihood after an iteration of
   * EM
   * @return true if the likelihood is invalid
   * @return false if the likelihood is okay
   */
  virtual bool IsInvalidLikelihood(double ln_l_difference);

  /**
   * Do M step
   *
   * Update the prior probabilities and estimated response probabilities given
   * the posterior probabilities. Modifies the member variables prior_ and
   * estimated_prob_
   *
   * @return false
   */
  virtual bool MStep();

  /**
   * Estimate probability
   *
   * Updates and modify the member variable estimated_prob_ using the
   * posterior
   */
  void EstimateProbability();

  /**
   * Weighted sum of outcomes for a given cluster
   *
   * Calculates the sum of outcomes weighted by the posterior for a given
   * cluster (or vice versa) where the outcomes are either zeros (the outcome
   * has not been observed) or ones (the outcome has been observed). This is
   * done for all categories. The member variable estimated_prob_ is updated
   * with the results for the corresponding cluster.
   *
   * This is used to estimate outcome probabilities. Because the responses or
   * observed outcomes are binary, the weighted sum can also be viewed as a
   * selective sum of posteriors.
   *
   * @param cluster_index which cluster to consider
   */
  virtual void WeightedSumProb(int cluster_index);

  /**
   * Normalise the weighted sum following WeightedSumProb()
   *
   * After calling WeightedSumProb(), call this to normalise the weighted sum so
   * that the member variable estimated_prob_ contains estimated probabilities
   * for each outcome for a given cluster.
   *
   * Can be overridden as the sum of weights can be calculated differently.
   *
   * @param cluster_index which cluster to consider
   */
  virtual void NormalWeightedSumProb(int cluster_index);

  /**
   * Normalise the weighted sum following WeightedSumProb() given the normaliser
   *
   * After calling WeightedSumProb(), call this to normalise the weighted sum by
   * a provided normaliser. Each probability for a given cluster is divided by
   * this normaliser.
   *
   * @param cluster_index which cluster to consider
   * @param normaliser the scale to divide the weighted sum by, should be the
   * sum of posteriors
   */
  void NormalWeightedSumProb(int cluster_index, double normaliser);
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
 * iteratively multiplied. However, to avoid underflow errors, a sum of log
 * probabilities is done instead if an underflow is detected. It should be
 * noted the sum of logs is slower.
 *
 * @param responses_i the responses for a given data point, length n_catgeory
 * @param n_catgeory number of categories
 * @param n_outcomes number of outcomes for each category
 * @param estimated_prob pointer to estimated probabilities for the
 * corresponding cluster. This is modified to point to the probabilities for
 * the next cluster. It points to a flattened list in the following order
 * <ul>
 *   <li>dim 0: for each outcome</li>
 *   <li>dim 1: for each category</li>
 * </ul>
 * @param prior the prior for this data point and cluster
 * @return the unnormalised posterior for this data point and cluster
 */
double PosteriorUnnormalize(int* responses_i, int n_category, int* n_outcomes,
                            double** estimated_prob, double prior);

/**
 * Generate random response probabilities
 *
 * @param rng random number generator
 * @param uniform uniform (0, 1)
 * @param n_outcomes vector length n_category, number of outcomes for each
 * category
 * @param sum_outcomes sum of n_outcomes
 * @param n_category number of categories
 * @param n_cluster number of clusters
 * @param prob output, vector of random response probabilities, conditioned on
 * cluster, for each outcome, category and cluster, flattened list in the
 * following order
 * <ul>
 *   <li>dim 0: for each outcome</li>
 *   <li>dim 1: for each category</li>
 *   <li>dim 2: for each cluster</li>
 * </ul>
 */
void GenerateNewProb(std::mt19937_64* rng,
                     std::uniform_real_distribution<double>* uniform,
                     int* n_outcomes, int sum_outcomes, int n_category,
                     int n_cluster, double* prob);

}  // namespace polca_parallel

#endif  // POLCAPARALLEL_SRC_EM_ALGORITHM_H_
