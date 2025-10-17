// poLCAParallel
// Copyright (C) 2025 Sherman Lo

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

#ifndef POLCAPARALLEL_TESTS_UTIL_TEST_H_
#define POLCAPARALLEL_TESTS_UTIL_TEST_H_

#include <catch2/catch_all.hpp>
#include <cstddef>
#include <random>
#include <span>
#include <vector>

#include "em_algorithm.h"
#include "em_algorithm_array.h"
#include "util.h"

namespace polca_parallel_test {

inline constexpr double TOLERANCE = 1e-12;

/**
 * Calculate the number of fully observed responses
 *
 * Calculate (or count) the number of fully observed responses. Unobserved
 * responses are coded as zero
 *
 * @param responses Design matrix TRANSPOSED of responses, matrix containing
 * outcomes/responses for each category as integers 1, 2, 3, .... The matrix
 * has dimensions
 * <ul>
 *   <li>dim 0: for each category</li>
 *   <li>dim 1: for each data point</li>
 * </ul>
 * @param n_data Number of data points
 * @param n_category Number of categories in each response
 * @return std::size_t Number of fully observed responses
 */
[[nodiscard]] std::size_t CalcNObs(std::span<const int> responses,
                                   std::size_t n_data, std::size_t n_category);

/**
 * Generate random responses
 *
 * Generate random responses using random priors and random outcome
 * probabilities. Provide a rng and the resulting random responses are returned
 *
 * @param n_data number of data points
 * @param n_outcomes number of outcomes for each category
 * @param rng random number generator
 * @return std::vector<int> The generated responses in matrix form, design
 * matrix TRANSPOSED of responses, matrix containing outcomes/responses for each
 * category as integers 1, 2, 3, .... The matrix has dimensions
 * <ul>
 *   <li>dim 0: for each category</li>
 *   <li>dim 1: for each data point</li>
 * </ul>
 */
std::vector<int> RandomMarginal(std::size_t n_data,
                                polca_parallel::NOutcomes n_outcomes,
                                std::mt19937_64& rng);

/**
 * Set missing data at random to the responses
 *
 * Set missing data at random to the responses by setting them to zero
 *
 * @param missing_prob the probability a data point is set to zero or missing
 * @param rng random number generator
 * @param responses matrix of responses
 */
void SetMissingAtRandom(double missing_prob, std::mt19937_64& rng,
                        std::span<int> responses);

/**
 * Instantiate a rng from an array of numbers
 *
 * @param seed_array a vector of seeds to init a rng
 * @return std::mt19937_64 random number generator
 */
std::mt19937_64 InitRng(std::vector<unsigned>& seed_array);

/**
 * Instantiate a rng from a seed_seq
 *
 * @param seed_seq seed sequence to init a rng
 * @return std::mt19937_64 random number generator
 */
std::mt19937_64 InitRng(std::seed_seq& seed_seq);

/**
 * Create a random NOutcomes
 *
 * How to use: init a std::vector<std::size_t> of length n_category, then call
 * the function. The vector is modified with the random n_outcomes and the
 * corresponding NOutcomes object is returned
 *
 * @param max_n_outcome maximum number of outcome
 * @param n_outcomes_vec Modified, memory to store
 * @param rng random number generator
 * @return polca_parallel::NOutcomes
 */
polca_parallel::NOutcomes RandomNOutcomes(
    std::size_t max_n_outcome, std::vector<std::size_t>& n_outcomes_vec,
    std::mt19937_64& rng);

/**
 * Create random probabilities for each cluster
 *
 * Create random probabilities for each cluster which can be used for the prior
 * and/or posterior
 *
 * @param n_data number of data points
 * @param n_cluster number of clusters
 * @param rng random number generator
 * @return arma::Mat<double> matrix with size n_data x n_cluster, each row has
 * normalised probabilites for each cluster
 */
arma::Mat<double> RandomClusterProbs(std::size_t n_data, std::size_t n_cluster,
                                     std::mt19937_64& rng);

/**
 * Allocate memory for storing the outputs or results
 *
 * Allocate memory for storing the resulting posterior, prior, estiamted_prob
 * and regress_coeff
 *
 * @param n_data number of data points
 * @param n_feature number of features
 * @param n_outcomes number of outcomes for each category
 * @param n_cluster number of clusters
 * @return std::tuple<std::vector<double>, std::vector<double>,
 * std::vector<double>, std::vector<double>> Allocated memory for the posterior,
 * prior, estimated_prob and regress_coeff respectively
 */
std::tuple<std::vector<double>, std::vector<double>, std::vector<double>,
           std::vector<double>>
InitOutputs(std::size_t n_data, std::size_t n_feature,
            polca_parallel::NOutcomes n_outcomes, std::size_t n_cluster);

/**
 * Test the outcome probabilities
 *
 * Test the outcome probabilities are in [0.0, 1.0] and the outcome
 * probabilities, for a given category and cluster, sums to 1.0
 *
 * @param n_outcomes number of outcomes for each category
 * @param n_cluster number of clusters
 * @param probs vector of outcome probabilities for each outcome, category and
 * cluster, flatten list in the following order
 * <ul>
 *   <li>dim 0: for each outcome</li>
 *   <li>dim 1: for each category</li>
 *   <li>dim 2: for each cluster</li>
 * </ul>
 */
void TestOutcomeProbs(polca_parallel::NOutcomes n_outcomes,
                      std::size_t n_cluster, std::span<const double> probs);

/**
 * Test the cluster (prior/posterior) probabilities
 *
 * Test the cluster (prior/posterior) probabilities are in [0.0, 1.0] and the
 * cluster probabilities, for each data point or row, sums to 1.0
 *
 * @param cluster_probs design matrix of probabilities, the matrix has the
 * following dimensions
 * <ul>
 *   <li>dim 0: for each data</li>
 *   <li>dim 1: for each cluster</li>
 * </ul>
 * @param n_data number of data points, ie number of rows in cluster_probs
 * @param n_cluster number of clusters, ie number of columns in cluster_probs
 */
void TestClusterProbs(std::span<const double> cluster_probs, std::size_t n_data,
                      std::size_t n_cluster);

/**
 * Test the output of the posterior, prior, estimated_prob and regress_coeff
 *
 * EmAlgorithmType is used to determine to test regress_coeff or not. The output
 * regress_coeff is only tested for regression problems
 *
 * Test the probabilities in posterior, prior and estimated_prob are in
 * [0.0, 1.0] and they are correctly normalised. Also checks if regress_coeff
 * is a number if applicable
 *
 * @tparam EmAlgorithmType
 * @param n_data number of data points
 * @param n_outcomes number of outcomes for each category
 * @param n_cluster number of clusters
 * @param posterior design matrix of posterior probabilities. The matrix has the
 * following dimensions
 * <ul>
 *   <li>dim 0: for each data</li>
 *   <li>dim 1: for each cluster</li>
 * </ul>
 * @param prior design matrix of prior probabilities. The matrix has the
 * following dimensions
 * <ul>
 *   <li>dim 0: for each data</li>
 *   <li>dim 1: for each cluster</li>
 * </ul>
 * @param estimated_prob vector of outcome probabilities for each outcome,
 * category and cluster, flatten list in the following order
 * <ul>
 *   <li>dim 0: for each outcome</li>
 *   <li>dim 1: for each category</li>
 *   <li>dim 2: for each cluster</li>
 * </ul>
 * @param regress_coeff matrix of regression coefficients with the following
 * dimensionsisfinite
 * <ul>
 *   <li>dim 0: n_features</li>
 *   <li>dim 1: n_cluster - 1</li>
 * </ul>
 */
template <typename EmAlgorithmType>
void TestDefaultOutputs(std::size_t n_data,
                        polca_parallel::NOutcomes n_outcomes,
                        std::size_t n_cluster,
                        std::span<const double> posterior,
                        std::span<const double> prior,
                        std::span<const double> estimated_prob,
                        std::span<const double> regress_coeff);

/**
 * Test the optional outputs from EmAlgorithm
 *
 * Test the outputs of EmAlgorithm::get_best_rep_index() and
 * EmAlgorithm::get_n_iter()
 *
 * @param fitter EmAlgorithm object to test
 * @param max_iter max_iter passed to fitter
 */
void TestEmAlgorithmOptionalOutputs(polca_parallel::EmAlgorithm& fitter,
                                    std::size_t max_iter);

/**
 * Black box test for EmAlgorithm and their subclasses
 *
 * Black box test for EmAlgorithm and their subclasses. Provided simulated data
 * and the EmAlgorithm is initalised within the function for testing
 *
 * Sections:
 *
 * <ul>
 *   <li>
 *     Test the outputs: posterior, prior, estimated_prob, regress_coeff,
 *     get_ln_l() and get_n_iter()
 *   </li>
 *   <li>
 *     Same as above but also calls set_best_initial_prob() and test it
 *   </li>
 *   <li>
 *     Tests if the results can be reproduced again when given the equivalent
 *     rng
 *   </li>
 *   <li>
 *     Tests if the resulting state from move_rng() can be reproduced
 *   </li>
 * </ul>
 *
 * @tparam EmAlgorithmType the type of EmAlgorithm to test, this determines what
 * to test, eg regress_coeff only in regression problems
 * @param features design matrix of features, matrix with dimensions
 * <ul>
 *   <li>dim 0: for each data point</li>
 *   <li>dim 1: for each feature</li>
 * </ul>
 * can be empty for the non-regression problem
 * @param responses design matrix TRANSPOSED of responses, matrix containing
 * outcomes/responses for each category as integers 1, 2, 3, .... The matrix
 * has dimensions
 * <ul>
 *   <li>dim 0: for each category</li>
 *   <li>dim 1: for each data point</li>
 * </ul>
 * @param initial_prob vector of estimated response probabilities, conditioned
 * on cluster, for each category. A flatten list in the following order
 * <ul>
 *   <li>dim 0: for each outcome</li>
 *   <li>dim 1: for each category</li>
 *   <li>dim 2: for each cluster</li>
 * </ul>
 * @param n_data number of data points
 * @param n_feature number of features, set to 1 for the non-regression problem
 * @param n_outcomes number of outcomes for each category
 * @param n_cluster number of clusters
 * @param max_iter maximum number of iterations for EM algorithm
 * @param tolerance tolerance for difference in log-likelihood, used for
 * stopping condition
 * @param seed for seeding the EmAlgorithm
 * @param is_full_constructor true if to the constructor which requires all
 * parameters, false to use the overloaded connstructor which has fewer
 * parameters
 */
template <typename EmAlgorithmType>
void BlackBoxTestEmAlgorithm(std::span<const double> features,
                             std::span<const int> responses,
                             std::span<const double> initial_prob,
                             std::size_t n_data, std::size_t n_feature,
                             polca_parallel::NOutcomes n_outcomes,
                             std::size_t n_cluster, unsigned int max_iter,
                             double tolerance, unsigned int seed,
                             bool is_full_constructor);

/**
 * Test the optional outputs from EmAlgorithmArray
 *
 * Test the outputs of EmAlgorithmArray::get_best_rep_index() and
 * EmAlgorithmArray::get_n_iter()
 *
 * @param fitter EmAlgorithmArray to test
 * @param n_rep n_rep passed passed to fitter
 * @param max_iter max_iter passed to fitter
 */
void TestEmAlgorithmArrayOptionalOutputs(
    std::unique_ptr<polca_parallel::EmAlgorithmArray>& fitter,
    std::size_t n_rep, std::size_t max_iter);

/**
 * Black box test for EmAlgorithmArray and their subclasses
 *
 * Black box test for EmAlgorithmArray and their subclasses. Provided simulated
 * data and the EmAlgorithmArray is initalised within the function for testing
 *
 * Sections:
 *
 * <ul>
 *   <li>
 *     Test the outputs: posterior, prior, estimated_prob, regress_coeff,
 *     get_best_rep_index() and get_n_iter()
 *   </li>
 *   <li>
 *     Same as above but also calls set_best_initial_prob() and
 *     set_ln_l_array() before fitting. The resulting best_initial_prob and
 *     ln_l_array are tested
 *  </li>
 *  <li>
 *     Test if results can be reproduced again when given the same seed_seq and
 *     using one thread
 *   </li>
 * </ul>
 *
 * @tparam EmAlgorithmArrayType either EmAlgorithmArray or EmAlgorithmArray to
 * test
 * @tparam EmAlgorithmType the type to pass to Fit<>(), this specifies if the
 * problem is a regression problem or not, and if missing data is in the data
 * or not
 * @param features design matrix of features, matrix with dimensions
 * <ul>
 *   <li>dim 0: for each data point</li>
 *   <li>dim 1: for each feature</li>
 * </ul>
 * can be empty for the non-regression problem
 * @param responses design matrix TRANSPOSED of responses, matrix containing
 * outcomes/responses for each category as integers 1, 2, 3, .... The matrix
 * has dimensions
 * <ul>
 *   <li>dim 0: for each category</li>
 *   <li>dim 1: for each data point</li>
 * </ul>
 * @param initial_prob vector of estimated response probabilities, conditioned
 * on cluster, for each category. A flatten list in the following order
 * <ul>
 *   <li>dim 0: for each outcome</li>
 *   <li>dim 1: for each category</li>
 *   <li>dim 2: for each cluster</li>
 * </ul>
 * @param n_data number of data points
 * @param n_feature number of features, set to 1 for the non-regression problem
 * @param n_outcomes number of outcomes for each category
 * @param n_cluster number of clusters
 * @param n_rep number of initial values to try out
 * @param n_thread number of threads to use
 * @param max_iter maximum number of iterations for EM algorithm
 * @param tolerance tolerance for difference in log-likelihood, used for
 * stopping condition
 * @param seed_seq for seeding EmAlgorithmArray
 * @param is_full_constructor true if to the constructor which requires all
 * parameters, false to use the overloaded connstructor which has fewer
 * parameters
 */
template <typename EmAlgorithmArrayType, typename EmAlgorithmType>
void BlackBoxTestEmAlgorithmArray(std::span<const double> features,
                                  std::span<const int> responses,
                                  std::span<const double> initial_prob,
                                  std::size_t n_data, std::size_t n_feature,
                                  polca_parallel::NOutcomes n_outcomes,
                                  std::size_t n_cluster, std::size_t n_rep,
                                  std::size_t n_thread, unsigned int max_iter,
                                  double tolerance, std::seed_seq& seed_seq,
                                  bool is_full_constructor);
/**
 * Black box test for StandardError and their subclasses
 *
 * Black box test for StandardError and their subclasses. Provided simulated
 * data and the StandardError is initalised within the function for testing
 *
 * @tparam StandardErrorType the type to test, StandardError or their subclass
 * @param features design matrix of features, matrix with dimensions
 * <ul>
 *   <li>dim 0: for each data point</li>
 *   <li>dim 1: for each feature</li>
 * </ul>
 * can be empty for the non-regression problem
 * @param responses design matrix TRANSPOSED of responses, matrix containing
 * outcomes/responses for each category as integers 1, 2, 3, .... The matrix
 * has dimensions
 * <ul>
 *   <li>dim 0: for each category</li>
 *   <li>dim 1: for each data point</li>
 * </ul>
 * @param probs vector of estimated response probabilities, conditioned
 * on cluster, for each category. A flatten list in the following order
 * <ul>
 *   <li>dim 0: for each outcome</li>
 *   <li>dim 1: for each category</li>
 *   <li>dim 2: for each cluster</li>
 * </ul>
 * @param posterior design matrix of posterior probabilities. The matrix has the
 * following dimensions
 * <ul>
 *   <li>dim 0: for each data</li>
 *   <li>dim 1: for each cluster</li>
 * </ul>
 * @param prior design matrix of prior probabilities. The matrix has the
 * following dimensions
 * <ul>
 *   <li>dim 0: for each data</li>
 *   <li>dim 1: for each cluster</li>
 * </ul>
 * @param n_data number of data points
 * @param n_feature number of features, set to 1 for the non-regression problem
 * @param n_outcomes number of outcomes for each category
 * @param n_cluster number of clusters
 * @param is_full_constructor true if to the constructor which requires all
 * parameters, false to use the overloaded connstructor which has fewer
 * parameters
 */
template <typename StandardErrorType>
void BlackBoxTestStandardError(std::span<const double> features,
                               std::span<const int> responses,
                               std::span<const double> probs,
                               const arma::Mat<double>& posterior,
                               const arma::Mat<double>& prior,
                               std::size_t n_data, std::size_t n_feature,
                               polca_parallel::NOutcomes n_outcomes,
                               std::size_t n_cluster, bool is_full_constructor);

}  // namespace polca_parallel_test

#endif  // POLCAPARALLEL_TESTS_UTIL_TEST_H_
