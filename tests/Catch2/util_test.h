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

/** Tolerance for equality of probabilities */
inline constexpr double kTolerance = 1e-12;
/** Tolerance for <code>arma::Mat::is_symmetric()</code> */
inline constexpr double kSymmetricTolerance = 1e-15;

/**
 * Calculate the number of fully observed responses
 *
 * Calculate (or count) the number of fully observed responses. Unobserved
 * responses are coded as zero
 *
 * @param responses Design matrix <b>transposed</b> of responses, matrix
 * containing outcomes/responses for each category as integers 1, 2, 3, ....
 * Missing values may be encoded as 0. The matrix has dimensions
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
 * @param n_data Number of data points
 * @param n_outcomes Number of outcomes for each category
 * @param rng Random number generator
 * @return std::vector<int> The generated responses in matrix form, design
 * matrix <b>transposed</b> of responses, matrix containing outcomes/responses
 * for each category as integers 1, 2, 3, .... The matrix has dimensions
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
 * @param missing_prob Probability a data point is set to zero or missing
 * @param rng Random number generator
 * @param responses Matrix of responses to modify
 */
void SetMissingAtRandom(double missing_prob, std::mt19937_64& rng,
                        std::span<int> responses);

/**
 * Instantiate a rng from an array of numbers
 *
 * @param seed_array Vector of seeds to init a rng
 * @return std::mt19937_64 random number generator
 */
std::mt19937_64 InitRng(std::vector<unsigned>& seed_array);

/**
 * Instantiate a rng from a <code>seed_seq</code>
 *
 * @param seed_seq Seed sequence to init a rng
 * @return std::mt19937_64 random number generator
 */
std::mt19937_64 InitRng(std::seed_seq& seed_seq);

/**
 * Create a random polca_parallel::NOutcomes
 *
 * Create a random polca_parallel::NOutcomes. This is used to set a random
 * number for the number of outcomes for each category
 *
 * To use this function, init a <code>std::vector<std::size_t></code> of length
 * <code>n_category</code>, then pass it as an argument of
 * <code>n_outcomes_vec</code>. The vector is modified with the random
 * <code>n_outcomes</code> and the corresponding polca_parallel::NOutcomes
 * object is returned
 *
 * @param max_n_outcome Maximum number of outcome for every outcome
 * @param n_outcomes_vec <b>Modified</b> To store the number of outcomes for
 * each category
 * @param rng Random number generator
 * @return polca_parallel::NOutcomes The number of outcomes for each category
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
 * @param n_data Number of data points
 * @param n_cluster Number of clusters
 * @param rng Random number generator
 * @return arma::Mat<double> matrix with size <code>n_data</code> x
 * <code>n_cluster</code>, each row has normalised probabilites for each cluster
 */
arma::Mat<double> RandomClusterProbs(std::size_t n_data, std::size_t n_cluster,
                                     std::mt19937_64& rng);

/**
 * Allocate memory for storing the outputs or results
 *
 * Allocate memory for storing the resulting <code>posterior</code>,
 * <code>prior</code>, <code>estiamted_prob</code> and
 * <code>regress_coeff</code>
 *
 * @param n_data Number of data points
 * @param n_feature Number of features
 * @param n_outcomes Number of outcomes for each category
 * @param n_cluster Number of clusters
 * @return std::tuple<std::vector<double>, std::vector<double>,
 * std::vector<double>, std::vector<double>> Allocated memory for the
 * <code>posterior</code>, <code>prior</code>, <code>estimated_prob</code> and
 * <code>regress_coeff</code> respectively
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
 * @param n_outcomes Number of outcomes for each category
 * @param n_cluster Number of clusters
 * @param probs Vector of outcome probabilities for each outcome, category and
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
 * @param cluster_probs Design matrix of probabilities, the matrix has the
 * following dimensions
 * <ul>
 *   <li>dim 0: for each data</li>
 *   <li>dim 1: for each cluster</li>
 * </ul>
 * @param n_data Number of data points, ie number of rows in
 * <code>cluster_probs</code>
 * @param n_cluster Number of clusters, ie number of columns in
 * <code>cluster_probs</code>
 */
void TestClusterProbs(std::span<const double> cluster_probs, std::size_t n_data,
                      std::size_t n_cluster);

/**
 * Test the default outputs of polca_parallel::EmAlgorithm
 *
 * Test the default outputs of polca_parallel::EmAlgorithm. They are the
 * <code>posterior</code>, <code>prior</code> and <code>estimated_prob</code>.
 * For the regression problem, the output <code>regress_coeff</code> is also
 * tested
 *
 * Test the probabilities in <code>posterior</code>, <code>prior</code> and
 * <code>estimated_prob</code> are in
 * [0.0, 1.0] and they are correctly normalised. Also checks if
 * <code>regress_coeff</code> is a number if applicable
 *
 * <code>EmAlgorithmType</code> is used to determine to test
 * <code>regress_coeff</code> or not. The output <code>regress_coeff</code> is
 * only tested for regression problems
 *
 * @tparam EmAlgorithmType The type of polca_parallel::EmAlgorithm to test, this
 * determines what to test, eg <code>regress_coeff</code> is tested only in
 * regression problems, polca_parallel::EmAlgorithmNan supports missing values
 * @param n_data Number of data points
 * @param n_outcomes Number of outcomes for each category
 * @param n_cluster Number of clusters
 * @param posterior Design matrix of posterior probabilities. The matrix has the
 * following dimensions
 * <ul>
 *   <li>dim 0: for each data</li>
 *   <li>dim 1: for each cluster</li>
 * </ul>
 * @param prior Design matrix of prior probabilities. The matrix has the
 * following dimensions
 * <ul>
 *   <li>dim 0: for each data</li>
 *   <li>dim 1: for each cluster</li>
 * </ul>
 * @param estimated_prob Vector of outcome probabilities for each outcome,
 * category and cluster, flatten list in the following order
 * <ul>
 *   <li>dim 0: for each outcome</li>
 *   <li>dim 1: for each category</li>
 *   <li>dim 2: for each cluster</li>
 * </ul>
 * @param regress_coeff Matrix of regression coefficients with the following
 * dimensions
 * <ul>
 *   <li>dim 0: <code>n_feature</code></li>
 *   <li>dim 1: <code>n_cluster - 1</code></li>
 * </ul>
 */
template <typename EmAlgorithmType>
void TestEmAlgorithmDefaultOutputs(std::size_t n_data,
                                   polca_parallel::NOutcomes n_outcomes,
                                   std::size_t n_cluster,
                                   std::span<const double> posterior,
                                   std::span<const double> prior,
                                   std::span<const double> estimated_prob,
                                   std::span<const double> regress_coeff);

/**
 * Test the optional outputs from polca_parallel::EmAlgorithm
 *
 * Test the outputs of polca_parallel::EmAlgorithm::get_ln_l() and
 * polca_parallel::EmAlgorithm::get_n_iter()
 *
 * @param fitter polca_parallel::EmAlgorithm object to test
 * @param max_iter <code>max_iter</code> argument passed to <code>fitter</code>
 */
void TestEmAlgorithmOptionalOutputs(polca_parallel::EmAlgorithm& fitter,
                                    std::size_t max_iter);

/**
 * Black box test for polca_parallel::EmAlgorithm and their subclasses
 *
 * Black box test for polca_parallel::EmAlgorithm and their subclasses. Provided
 * simulated data and the polca_parallel::EmAlgorithm are initalised within the
 * function for testing
 *
 * Sections:
 *
 * <ul>
 *   <li>
 *     Test the outputs: <code>posterior</code>, <code>prior</code>,
 *     <code>estimated_prob</code>, <code>regress_coeff</code>,
 *     polca_parallel::EmAlgorithm::get_ln_l() and
 *     polca_parallel::EmAlgorithm::get_n_iter()
 *   </li>
 *   <li>
 *     Same as above but also calls
 *     polca_parallel::EmAlgorithm::set_best_initial_prob() and test it
 *   </li>
 *   <li>
 *     Tests if the results can be reproduced again when given the equivalent
 *     rng
 *   </li>
 *   <li>
 *     Tests if the resulting state from polca_parallel::EmAlgorithm::move_rng()
 *     can be reproduced
 *   </li>
 * </ul>
 *
 * @tparam EmAlgorithmType The type of polca_parallel::EmAlgorithm to test, this
 * determines what to test, eg <code>regress_coeff</code> is tested only in
 * regression problems, polca_parallel::EmAlgorithmNan supports missing values
 * @param features Design matrix of features, matrix with dimensions
 * <ul>
 *   <li>dim 0: for each data point</li>
 *   <li>dim 1: for each feature</li>
 * </ul>
 * Can be empty for the non-regression problem
 * @param responses Design matrix <b>transposed</b> of responses, matrix
 * containing outcomes/responses for each category as integers 1, 2, 3, .... If
 * supported, 0 can be used to indicate a missing value. The matrix has
 * dimensions
 * <ul>
 *   <li>dim 0: for each category</li>
 *   <li>dim 1: for each data point</li>
 * </ul>
 * @param initial_prob Vector of initial response probabilities for each
 * outcome, conditioned on the category and cluster. A flatten list in the
 * following order
 * <ul>
 *   <li>dim 0: for each outcome</li>
 *   <li>dim 1: for each category</li>
 *   <li>dim 2: for each cluster</li>
 * </ul>
 * @param n_data Number of data points
 * @param n_feature Number of features, set to 1 for the non-regression problem
 * @param n_outcomes Number of outcomes for each category
 * @param n_cluster Number of clusters
 * @param max_iter Maximum number of iterations for EM algorithm
 * @param tolerance Tolerance for difference in log-likelihood, used for
 * stopping condition
 * @param seed For seeding the polca_parallel::EmAlgorithm
 * @param is_full_constructor <code>true</code> if to use the constructor which
 * requires all parameters, <code>false</code> to use the overloaded
 * constructor which has fewer parameters
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
 * Test the optional outputs from polca_parallel::EmAlgorithmArray
 *
 * Test the outputs of polca_parallel::EmAlgorithmArray::get_best_rep_index()
 * and polca_parallel::EmAlgorithmArray::get_n_iter()
 *
 * @param fitter polca_parallel::EmAlgorithmArray to test
 * @param n_rep <code>n_rep</code> argument to pass to <code>fitter</code>
 * @param max_iter <code>max_iter</code> argument to pass <code>fitter</code>
 */
void TestEmAlgorithmArrayOptionalOutputs(
    std::unique_ptr<polca_parallel::EmAlgorithmArray>& fitter,
    std::size_t n_rep, std::size_t max_iter);

/**
 * Black box test for polca_parallel::EmAlgorithmArray and their subclasses
 *
 * Black box test for polca_parallel::EmAlgorithmArray and their subclasses.
 * Provided simulated data and the polca_parallel::EmAlgorithmArray are
 * initalised within the function for testing
 *
 * Sections:
 *
 * <ul>
 *   <li>
 *     Test the outputs: <code>posterior</code>, <code>prior</code>,
 *     <code>estimated_prob</code>, <code>regress_coeff</code>,
 *     polca_parallel::EmAlgorithmArray::get_best_rep_index() and
 *     polca_parallel::EmAlgorithmArray::get_n_iter()
 *   </li>
 *   <li>
 *     Same as above but also calls
 *     polca_parallel::EmAlgorithmArray::set_best_initial_prob() and
 *     polca_parallel::EmAlgorithmArray::set_ln_l_array() before fitting. The
 *     resulting <code>best_initial_prob</code> and <code>ln_l_array</code> are
 *     tested
 *  </li>
 *  <li>
 *     Test if results can be reproduced again when given the same
 *     <code>seed_seq</code> and using one thread
 *   </li>
 * </ul>
 *
 * @tparam EmAlgorithmArrayType Either polca_parallel::EmAlgorithmArray or
 * polca_parallel::EmAlgorithmArray to test
 * @tparam EmAlgorithmType The type to pass to
 * polca_parallel::EmAlgorithmArray::Fit<>(), this specifies if the problem is a
 * regression problem or not, and if missing data is in the data or not
 * @param features Design matrix of features, matrix with dimensions
 * <ul>
 *   <li>dim 0: for each data point</li>
 *   <li>dim 1: for each feature</li>
 * </ul>
 * Can be empty for the non-regression problem
 * @param responses Design matrix <b>transposed</b> of responses, matrix
 * containing outcomes/responses for each category as integers 1, 2, 3, .... If
 * supported, 0 can be used to indicate a missing value. The matrix has
 * dimensions
 * <ul>
 *   <li>dim 0: for each category</li>
 *   <li>dim 1: for each data point</li>
 * </ul>
 * @param initial_prob Vector of initial response probabilities for each
 * outcome, conditioned on the category and cluster. A flatten list in the
 * following order
 * <ul>
 *   <li>dim 0: for each outcome</li>
 *   <li>dim 1: for each category</li>
 *   <li>dim 2: for each cluster</li>
 * </ul>
 * @param n_data Number of data points
 * @param n_feature Number of features, set to 1 for the non-regression problem
 * @param n_outcomes Number of outcomes for each category
 * @param n_cluster Number of clusters
 * @param n_rep Number of initial values to try out
 * @param n_thread Number of threads to use
 * @param max_iter Maximum number of iterations for EM algorithm
 * @param tolerance Tolerance for difference in log-likelihood, used for
 * stopping condition
 * @param seed_seq For seeding polca_parallel::EmAlgorithmArray
 * @param is_full_constructor <code>true</code> if to use the constructor which
 * requires all parameters, <code>false</code> to use the overloaded
 * constructor which has fewer parameters
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
 * Black box test for polca_parallel::StandardError and their subclasses
 *
 * Black box test for polca_parallel::StandardError and their subclasses.
 * Provided simulated data and the polca_parallel::StandardError are initalised
 * within the function for testing.
 *
 * Test if the errors are positive. For the regression problem, test if the
 * regression coefficient covariance matrix is symmetric
 *
 * @tparam StandardErrorType The type to test, polca_parallel::StandardError or
 * their subclass
 * @param features Design matrix of features, matrix with dimensions
 * <ul>
 *   <li>dim 0: for each data point</li>
 *   <li>dim 1: for each feature</li>
 * </ul>
 * Can be empty for the non-regression problem
 * @param responses Design matrix <b>transposed</b> of responses, matrix
 * containing outcomes/responses for each category as integers 1, 2, 3, .... If
 * supported, 0 can be used to indicate a missing value. The matrix has
 * dimensions
 * <ul>
 *   <li>dim 0: for each category</li>
 *   <li>dim 1: for each data point</li>
 * </ul>
 * @param probs Vector of response probabilities for each outcome, conditioned
 * on the category and cluster. A flatten list in the following order
 * <ul>
 *   <li>dim 0: for each outcome</li>
 *   <li>dim 1: for each category</li>
 *   <li>dim 2: for each cluster</li>
 * </ul>
 * @param posterior Design matrix of posterior probabilities. The matrix has the
 * following dimensions
 * <ul>
 *   <li>dim 0: for each data</li>
 *   <li>dim 1: for each cluster</li>
 * </ul>
 * @param prior Design matrix of prior probabilities. The matrix has the
 * following dimensions
 * <ul>
 *   <li>dim 0: for each data</li>
 *   <li>dim 1: for each cluster</li>
 * </ul>
 * @param n_data Number of data points
 * @param n_feature Number of features, set to 1 for the non-regression problem
 * @param n_outcomes Number of outcomes for each category
 * @param n_cluster Number of clusters
 * @param is_full_constructor <code>true</code> if to use the constructor which
 * requires all parameters, <code>false</code> to use the overloaded
 * constructor which has fewer parameters
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
