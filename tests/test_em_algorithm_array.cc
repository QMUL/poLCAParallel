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
 * @file
 * Test the class polca_parallel::EmAlgorithmArray
 *
 * Test the class polca_parallel::EmAlgorithmArray, which is responsible for
 * running multiple polca_parallel::EmAlgorithm in parallel. See the function
 * polca_parallel_test::BlackBoxTestEmAlgorithmArray() for further details
 *
 * The test cases are:
 *
 * <ul>
 *   <li>full-data: non-regression problem with all data observed</li>
 *   <li>full-data-regress: regression problem with all data observed</li>
 *   <li>missing-data: non-regression problem with some missing data</li>
 *   <li>missing-data-regress: regression problem with some missing data</li>
 * </ul>
 *
 * For the data to be valid:
 *
 * <ul>
 *   <li>
 *     For missing data, ensure each category has at least one observation
 *   </li>
 *   <li>
 *     For the regression problem, ensure there is more data than parameters, ie
 *     n_data >> n_feature * (n_cluster - 1)
 *   </li>
 * </ul>
 *
 * Other notes:
 *
 * <ul>
 *   <li>
 *     The use of bool is_full_constructor is to test the different overloaded
 *     constructors where in the non-regression problem, parameters such as
 *     features and n_feature are optional
 *   </li>
 *   <li>
 *     For the full data, it can be tested on both polca_parallel::EmAlgorithm
 *     and polca_parallel::EmAlgorithmRegress
 *   </li>
 * </ul>
 */

#define CATCH_CONFIG_MAIN

#include <catch2/catch_all.hpp>
#include <cmath>
#include <memory>
#include <numeric>
#include <random>
#include <tuple>
#include <vector>

#include "arma.h"
#include "em_algorithm.h"
#include "em_algorithm_array.h"
#include "em_algorithm_nan.h"
#include "util.h"
#include "util_test.h"

TEST_CASE("em-array-non-regression-full-data",
          "[em_array][full_data][non_regression]") {
  std::size_t n_data = GENERATE(10);
  std::size_t n_category = GENERATE(2);
  std::size_t max_n_outcome = GENERATE(2);
  std::size_t n_cluster = GENERATE(2);
  std::size_t n_rep = GENERATE(1, 2, 8);
  std::size_t n_thread = GENERATE(1, 4);
  bool is_full_constructor = GENERATE(false, true);
  unsigned int max_iter = 80;
  double tolerance = 1e-10;

  std::vector<unsigned> rng_seed_array = {1445071836, 2641392836, 3558200140};
  std::mt19937_64 rng = polca_parallel_test::InitRng(rng_seed_array);

  std::vector<unsigned> seed_array = {716340698, 1684557984, 908767965};
  std::seed_seq seed_seq(seed_array.cbegin(), seed_array.cend());

  std::vector<std::size_t> n_outcomes_vec(n_category);
  polca_parallel::NOutcomes n_outcomes =
      polca_parallel_test::RandomNOutcomes(max_n_outcome, n_outcomes_vec, rng);

  std::vector<double> initial_prob =
      polca_parallel::RandomInitialProb(n_outcomes, n_cluster, n_rep, rng);

  std::vector<int> responses =
      polca_parallel_test::RandomMarginal(n_data, n_outcomes, rng);

  SECTION("EmAlgorithm") {
    polca_parallel_test::BlackBoxTestEmAlgorithmArray<
        polca_parallel::EmAlgorithmArray, polca_parallel::EmAlgorithm>(
        std::span<const double>(),
        std::span<const int>(responses.begin(), responses.size()),
        std::span<const double>(initial_prob.begin(), initial_prob.size()),
        n_data, 1, n_outcomes, n_cluster, n_rep, n_thread, max_iter, tolerance,
        seed_seq, is_full_constructor);
  }
  SECTION("EmAlgorithmNan") {
    polca_parallel_test::BlackBoxTestEmAlgorithmArray<
        polca_parallel::EmAlgorithmArray, polca_parallel::EmAlgorithmNan>(
        std::span<const double>(),
        std::span<const int>(responses.begin(), responses.size()),
        std::span<const double>(initial_prob.begin(), initial_prob.size()),
        n_data, 1, n_outcomes, n_cluster, n_rep, n_thread, max_iter, tolerance,
        seed_seq, is_full_constructor);
  }
}

TEST_CASE("em-array-regression-full-data",
          "[em_array][full_data][regression]") {
  std::size_t n_data = GENERATE(500);
  std::size_t n_feature = GENERATE(2);
  std::size_t n_category = GENERATE(5);
  std::size_t max_n_outcome = GENERATE(2);
  std::size_t n_cluster = GENERATE(2);
  std::size_t n_rep = GENERATE(1, 2);
  std::size_t n_thread = GENERATE(1, 4);
  unsigned int max_iter = 80;
  double tolerance = 1e-10;

  std::vector<unsigned> rng_seed_array = {189605617, 2396276792, 219596477};
  std::mt19937_64 rng = polca_parallel_test::InitRng(rng_seed_array);

  std::vector<unsigned> seed_array = {742576796, 846367579, 456258734};
  std::seed_seq seed_seq(seed_array.cbegin(), seed_array.cend());

  std::vector<std::size_t> n_outcomes_vec(n_category);
  polca_parallel::NOutcomes n_outcomes =
      polca_parallel_test::RandomNOutcomes(max_n_outcome, n_outcomes_vec, rng);

  std::vector<double> initial_prob =
      polca_parallel::RandomInitialProb(n_outcomes, n_cluster, n_rep, rng);

  std::vector<int> responses =
      polca_parallel_test::RandomMarginal(n_data, n_outcomes, rng);

  std::vector<double> features(n_data * n_feature);
  for (double& feature : features) {
    std::normal_distribution<double> dist(0, 1);
    feature = dist(rng);
  }

  SECTION("EmAlgorithmRegress") {
    polca_parallel_test::BlackBoxTestEmAlgorithmArray<
        polca_parallel::EmAlgorithmArray, polca_parallel::EmAlgorithmRegress>(
        std::span<const double>(features.begin(), features.size()),
        std::span<const int>(responses.begin(), responses.size()),
        std::span<const double>(initial_prob.begin(), initial_prob.size()),
        n_data, n_feature, n_outcomes, n_cluster, n_rep, n_thread, max_iter,
        tolerance, seed_seq, true);
  }
  SECTION("EmAlgorithmNanRegress") {
    polca_parallel_test::BlackBoxTestEmAlgorithmArray<
        polca_parallel::EmAlgorithmArray,
        polca_parallel::EmAlgorithmNanRegress>(
        std::span<const double>(features.begin(), features.size()),
        std::span<const int>(responses.begin(), responses.size()),
        std::span<const double>(initial_prob.begin(), initial_prob.size()),
        n_data, n_feature, n_outcomes, n_cluster, n_rep, n_thread, max_iter,
        tolerance, seed_seq, true);
  }
}

TEST_CASE("em-array-non-regression-missing-data",
          "[em_array][missing_data][non_regression]") {
  std::size_t n_data = GENERATE(100);
  std::size_t n_category = GENERATE(2);
  std::size_t max_n_outcome = GENERATE(2);
  std::size_t n_cluster = GENERATE(2);
  std::size_t n_rep = GENERATE(1, 2, 8);
  std::size_t n_thread = GENERATE(1, 4);
  double missing_prob = GENERATE(0.1);
  bool is_full_constructor = GENERATE(false, true);
  unsigned int max_iter = 80;
  double tolerance = 1e-10;

  std::vector<unsigned> rng_seed_array = {3675480203, 1679317556, 1691702062};
  std::mt19937_64 rng = polca_parallel_test::InitRng(rng_seed_array);

  std::vector<unsigned> seed_array = {2305425190, 3912547098, 511195517};
  std::seed_seq seed_seq(seed_array.cbegin(), seed_array.cend());

  std::vector<std::size_t> n_outcomes_vec(n_category);
  polca_parallel::NOutcomes n_outcomes =
      polca_parallel_test::RandomNOutcomes(max_n_outcome, n_outcomes_vec, rng);

  std::vector<double> initial_prob =
      polca_parallel::RandomInitialProb(n_outcomes, n_cluster, n_rep, rng);

  std::vector<int> responses =
      polca_parallel_test::RandomMarginal(n_data, n_outcomes, rng);
  polca_parallel_test::SetMissingAtRandom(
      missing_prob, rng, std::span<int>(responses.begin(), responses.size()));

  polca_parallel_test::BlackBoxTestEmAlgorithmArray<
      polca_parallel::EmAlgorithmArray, polca_parallel::EmAlgorithmNan>(
      std::span<const double>(),
      std::span<const int>(responses.begin(), responses.size()),
      std::span<const double>(initial_prob.begin(), initial_prob.size()),
      n_data, 1, n_outcomes, n_cluster, n_rep, n_thread, max_iter, tolerance,
      seed_seq, is_full_constructor);
}

TEST_CASE("em-array-regression-missing-data",
          "[em_array][missing_data][regression]") {
  std::size_t n_data = GENERATE(500);
  std::size_t n_feature = GENERATE(2);
  std::size_t n_category = GENERATE(5);
  std::size_t max_n_outcome = GENERATE(2);
  std::size_t n_cluster = GENERATE(2);
  std::size_t n_rep = GENERATE(1, 2);
  std::size_t n_thread = GENERATE(1, 4);
  double missing_prob = GENERATE(0.1);
  unsigned int max_iter = 80;
  double tolerance = 1e-10;

  std::vector<unsigned> rng_seed_array = {412860385, 3071480017, 3823320066};
  std::mt19937_64 rng = polca_parallel_test::InitRng(rng_seed_array);

  std::vector<unsigned> seed_array = {2657464522, 1006284125, 3793063147};
  std::seed_seq seed_seq(seed_array.cbegin(), seed_array.cend());

  std::vector<std::size_t> n_outcomes_vec(n_category);
  polca_parallel::NOutcomes n_outcomes =
      polca_parallel_test::RandomNOutcomes(max_n_outcome, n_outcomes_vec, rng);

  std::vector<double> initial_prob =
      polca_parallel::RandomInitialProb(n_outcomes, n_cluster, n_rep, rng);

  std::vector<int> responses =
      polca_parallel_test::RandomMarginal(n_data, n_outcomes, rng);
  polca_parallel_test::SetMissingAtRandom(
      missing_prob, rng, std::span<int>(responses.begin(), responses.size()));

  std::vector<double> features(n_data * n_feature);
  for (double& feature : features) {
    std::normal_distribution<double> dist(0, 1);
    feature = dist(rng);
  }

  polca_parallel_test::BlackBoxTestEmAlgorithmArray<
      polca_parallel::EmAlgorithmArray, polca_parallel::EmAlgorithmNanRegress>(
      std::span<const double>(features.begin(), features.size()),
      std::span<const int>(responses.begin(), responses.size()),
      std::span<const double>(initial_prob.begin(), initial_prob.size()),
      n_data, n_feature, n_outcomes, n_cluster, n_rep, n_thread, max_iter,
      tolerance, seed_seq, true);
}
