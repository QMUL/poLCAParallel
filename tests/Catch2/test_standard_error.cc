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
 * Test the class polca_parallel::StandardError
 *
 * Test the class polca_parallel::StandardError. The class calculates the
 * standard error for the non-regression problem. The data is randomly generated
 * as well as the fitted prior and posterior. See
 * polca_parallel_test::BlackBoxTestStandardError for further details on the
 * tests
 */

#define CATCH_CONFIG_MAIN

#include <catch2/catch_all.hpp>
#include <random>
#include <vector>

#include "arma.h"
#include "standard_error.h"
#include "util.h"
#include "util_test.h"

TEST_CASE("std-non-regress", "[std][non_regression]") {
  std::size_t n_data = GENERATE(10, 100, 1000);
  std::size_t n_category = GENERATE(2, 5, 10);
  std::size_t max_n_outcome = GENERATE(2, 5, 10);
  std::size_t n_cluster = GENERATE(2, 5, 10);
  bool is_full_constructor = GENERATE(false, true);
  double missing_prob = GENERATE(0, 0.1);

  std::vector<unsigned> rng_seed_array = {1391161309, 876420009, 1748017172};
  std::mt19937_64 rng = polca_parallel_test::InitRng(rng_seed_array);

  std::vector<std::size_t> n_outcomes_vec(n_category);
  polca_parallel::NOutcomes n_outcomes =
      polca_parallel_test::RandomNOutcomes(max_n_outcome, n_outcomes_vec, rng);

  std::vector<double> probs =
      polca_parallel::RandomInitialProb(n_outcomes, n_cluster, 1, rng);

  std::vector<int> responses =
      polca_parallel_test::RandomMarginal(n_data, n_outcomes, rng);
  arma::Mat<int> responses_arma(responses.data(), n_outcomes.size(), n_data);
  arma::inplace_trans(responses_arma);

  polca_parallel_test::SetMissingAtRandom(missing_prob, rng, responses);

  arma::Mat<double> prior = arma::repmat(
      polca_parallel_test::RandomClusterProbs(1, n_cluster, rng), n_data, 1);
  arma::Mat<double> posterior =
      polca_parallel_test::RandomClusterProbs(n_data, n_cluster, rng);

  polca_parallel_test::BlackBoxTestStandardError<polca_parallel::StandardError>(
      std::span<const double>(), responses_arma, probs, prior, posterior,
      n_data, 1, n_outcomes, n_cluster, is_full_constructor);
}
