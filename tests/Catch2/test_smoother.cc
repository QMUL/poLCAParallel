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
 * Test the class polca_parallel::Smoother
 *
 * Test the class polca_parallel::Smoother which smooth probabilities for use
 * in the regularised error. The probabilities are randomly generated and then
 * smoothed for testing
 */

#define CATCH_CONFIG_MAIN

#include <catch2/catch_all.hpp>
#include <random>
#include <vector>

#include "smoother.h"
#include "util.h"
#include "util_test.h"

TEST_CASE("smoother") {
  std::size_t n_data = GENERATE(10, 100);
  std::size_t n_category = GENERATE(2, 5);
  std::size_t max_n_outcome = GENERATE(2, 5);
  std::size_t n_cluster = GENERATE(2, 5);

  std::vector<unsigned> rng_seed_array = {1535937996, 1674817532, 809757126};
  std::mt19937_64 rng = polca_parallel_test::InitRng(rng_seed_array);

  std::vector<std::size_t> n_outcomes_vec(n_category);
  polca_parallel::NOutcomes n_outcomes =
      polca_parallel_test::RandomNOutcomes(max_n_outcome, n_outcomes_vec, rng);

  std::vector<double> probs =
      polca_parallel::RandomInitialProb(n_outcomes, n_cluster, 1, rng);

  arma::Mat<double> prior =
      polca_parallel_test::RandomClusterProbs(n_data, n_cluster, rng);
  arma::Mat<double> posterior =
      polca_parallel_test::RandomClusterProbs(n_data, n_cluster, rng);

  polca_parallel::Smoother smoother(probs, prior, posterior, n_data, n_outcomes,
                                    n_cluster);
  smoother.Smooth();

  std::span<const double> probs_smooth = smoother.get_probs();
  std::span<const double> prior_smooth = smoother.get_prior();
  std::span<const double> posterior_smooth = smoother.get_posterior();

  polca_parallel_test::TestClusterProbs(posterior_smooth, n_data, n_cluster);
  polca_parallel_test::TestClusterProbs(prior_smooth, n_data, n_cluster);
  polca_parallel_test::TestOutcomeProbs(n_outcomes, n_cluster, probs_smooth);
}
