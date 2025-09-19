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

/**
 * Test the classes for NaN handling
 *
 * Test the classes polca_parallel::EmAlgorithmNan and
 * polca_parallel::EmAlgorithmNanRegress. They fit the model onto the data
 * whilst handling missing data. See the function
 * polca_parallel_test::BlackBoxTestEmAlgorithm() for further details
 *
 * The test cases are the non-regression and regression problem
 */

#define CATCH_CONFIG_MAIN

#include <catch2/catch_all.hpp>
#include <memory>
#include <random>
#include <vector>

#include "em_algorithm_nan.h"
#include "util.h"
#include "util_test.h"

TEST_CASE("em-nan-non-regression", "[em][non_regression]") {
  std::size_t n_data = GENERATE(100);
  std::size_t n_category = GENERATE(2, 5);
  std::size_t max_n_outcome = GENERATE(2, 5);
  std::size_t n_cluster = GENERATE(2, 5);
  double missing_prob = GENERATE(0, 0.1);
  bool is_full_constructor = GENERATE(false, true);
  unsigned int max_iter = 80;
  double tolerance = 1e-10;

  std::vector<unsigned> rng_seed_array = {1445071836, 2641392836, 3558200140};
  std::mt19937_64 rng = polca_parallel_test::InitRng(rng_seed_array);

  std::vector<unsigned> seed_array = {716340698, 1684557984, 908767965};
  std::seed_seq seed_seq(seed_array.cbegin(), seed_array.cend());
  unsigned int seed;
  seed_seq.generate(&seed, &seed + 1);

  std::vector<std::size_t> n_outcomes_vec(n_category);
  polca_parallel::NOutcomes n_outcomes =
      polca_parallel_test::RandomNOutcomes(max_n_outcome, n_outcomes_vec, rng);

  std::vector<double> initial_prob =
      polca_parallel::RandomInitialProb(n_outcomes, n_cluster, 1, rng);

  std::vector<int> responses =
      polca_parallel_test::RandomMarginal(n_data, n_outcomes, rng);

  polca_parallel_test::SetMissingAtRandom(
      missing_prob, rng, std::span<int>(responses.begin(), responses.size()));
  polca_parallel_test::BlackBoxTestEmAlgorithm<polca_parallel::EmAlgorithmNan>(
      std::span<const double>(),
      std::span<const int>(responses.begin(), responses.size()),
      std::span<const double>(initial_prob.begin(), initial_prob.size()),
      n_data, 1, n_outcomes, n_cluster, max_iter, tolerance, seed,
      is_full_constructor);
}

TEST_CASE("em-nan-regression-missing-data", "[em][regression]") {
  std::size_t n_data = GENERATE(100);
  std::size_t n_feature = GENERATE(2, 4);
  std::size_t n_category = GENERATE(2, 5);
  std::size_t max_n_outcome = GENERATE(2, 5);
  std::size_t n_cluster = GENERATE(2, 5);
  double missing_prob = GENERATE(0, 0.1);
  bool is_full_constructor = GENERATE(false, true);
  unsigned int max_iter = 80;
  double tolerance = 1e-10;

  std::vector<unsigned> rng_seed_array = {1471244869, 1468103316, 2764256346};
  std::mt19937_64 rng = polca_parallel_test::InitRng(rng_seed_array);

  std::vector<unsigned> seed_array = {3615481394, 2883953266, 3110633602};
  std::seed_seq seed_seq(seed_array.cbegin(), seed_array.cend());
  unsigned int seed;
  seed_seq.generate(&seed, &seed + 1);

  std::vector<std::size_t> n_outcomes_vec(n_category);
  polca_parallel::NOutcomes n_outcomes =
      polca_parallel_test::RandomNOutcomes(max_n_outcome, n_outcomes_vec, rng);

  std::vector<double> initial_prob =
      polca_parallel::RandomInitialProb(n_outcomes, n_cluster, 1, rng);

  std::vector<int> responses =
      polca_parallel_test::RandomMarginal(n_data, n_outcomes, rng);

  std::vector<double> features(n_data * n_feature);
  for (double& feature : features) {
    std::normal_distribution<double> dist(0, 1);
    feature = dist(rng);
  }

  polca_parallel_test::SetMissingAtRandom(
      missing_prob, rng, std::span<int>(responses.begin(), responses.size()));
  polca_parallel_test::BlackBoxTestEmAlgorithm<
      polca_parallel::EmAlgorithmNanRegress>(
      features, std::span<const int>(responses.begin(), responses.size()),
      std::span<const double>(initial_prob.begin(), initial_prob.size()),
      n_data, n_feature, n_outcomes, n_cluster, max_iter, tolerance, seed,
      is_full_constructor);
}
