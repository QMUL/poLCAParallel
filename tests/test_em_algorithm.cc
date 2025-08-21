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
 * Test the class polca_parallel::EmAlgorithm
 *
 * Test the class polca_parallel::EmAlgorithm which fits the model onto the
 * data. See the function polca_parallel_test::BlackBoxTestEmAlgorithm() for
 * further details
 *
 * The test case here is the non-regression problem with no missing data
 *
 * The use of bool is_full_constructor is to test the different overloaded
 * constructors where in the non-regression problem, parameters such as features
 * and n_feature are optional
 */

#define CATCH_CONFIG_MAIN

#include <catch2/catch_all.hpp>
#include <memory>
#include <random>
#include <vector>

#include "em_algorithm.h"
#include "util.h"
#include "util_test.h"

TEST_CASE("full-data") {
  std::size_t n_data = GENERATE(10, 100);
  std::size_t n_category = GENERATE(2, 5);
  std::size_t max_n_outcome = GENERATE(2, 5);
  std::size_t n_cluster = GENERATE(2, 5);
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

  polca_parallel_test::BlackBoxTestEmAlgorithm<polca_parallel::EmAlgorithm>(
      std::span<const double>(),
      std::span<const int>(responses.begin(), responses.size()),
      std::span<const double>(initial_prob.begin(), initial_prob.size()),
      n_data, 1, n_outcomes, n_cluster, max_iter, tolerance, seed,
      is_full_constructor);
}
