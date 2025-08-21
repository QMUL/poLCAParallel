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
 * Test the class polca_parallel::Blrt
 *
 * Test the class polca_parallel::Blrt and if the results can be reproduced
 */

#define CATCH_CONFIG_MAIN

#include <catch2/catch_all.hpp>
#include <vector>

#include "arma.h"
#include "blrt.h"
#include "util.h"
#include "util_test.h"

TEST_CASE("test-blrt") {
  std::size_t n_data = GENERATE(10);
  std::size_t n_category = GENERATE(2, 5);
  std::size_t max_n_outcome = GENERATE(2, 5);
  std::size_t n_bootstrap = GENERATE(10);
  std::size_t n_rep = GENERATE(5);
  std::size_t n_thread = GENERATE(5);
  std::size_t n_cluster_null = GENERATE(2, 5, 10);
  std::size_t n_cluster_alt_more = GENERATE(0, 1, 2, 5);
  unsigned int max_iter = 80;
  double tolerance = 1e-10;

  std::size_t n_cluster_alt = n_cluster_null + n_cluster_alt_more;

  std::vector<unsigned> rng_seed_array = {4221229022, 273683600, 450073693};
  std::mt19937_64 rng = polca_parallel_test::InitRng(rng_seed_array);

  std::vector<unsigned> seed_array = {2326066219, 56235768, 3769351879};
  std::seed_seq seed_seq(seed_array.cbegin(), seed_array.cend());

  std::vector<std::size_t> n_outcomes_vec(n_category);
  polca_parallel::NOutcomes n_outcomes =
      polca_parallel_test::RandomNOutcomes(max_n_outcome, n_outcomes_vec, rng);

  arma::Mat<double> prior_null =
      polca_parallel_test::RandomClusterProbs(1, n_cluster_null, rng);
  std::vector<double> probs_null =
      polca_parallel::RandomInitialProb(n_outcomes, n_cluster_null, 1, rng);

  arma::Mat<double> prior_alt =
      polca_parallel_test::RandomClusterProbs(1, n_cluster_alt, rng);
  std::vector<double> probs_alt =
      polca_parallel::RandomInitialProb(n_outcomes, n_cluster_alt, 1, rng);

  std::vector<double> ratio_array(n_bootstrap, std::nan(""));

  polca_parallel::Blrt blrt(
      std::span<const double>(prior_null.begin(), prior_null.size()),
      std::span<const double>(probs_null.begin(), probs_null.size()),
      std::span<const double>(prior_alt.begin(), prior_alt.size()),
      std::span<const double>(probs_alt.begin(), probs_alt.size()), n_data,
      n_outcomes, n_bootstrap, n_rep, n_thread, max_iter, tolerance,
      std::span<double>(ratio_array.begin(), ratio_array.size()));

  blrt.SetSeed(seed_seq);
  blrt.Run();

  for (auto ratio : ratio_array) {
    CHECK(!std::isnan(ratio));
  }

  SECTION("test-reproducible") {
    std::vector<double> ratio_array_2(n_bootstrap);

    polca_parallel::Blrt blrt_2(
        std::span<const double>(prior_null.begin(), prior_null.size()),
        std::span<const double>(probs_null.begin(), probs_null.size()),
        std::span<const double>(prior_alt.begin(), prior_alt.size()),
        std::span<const double>(probs_alt.begin(), probs_alt.size()), n_data,
        n_outcomes, n_bootstrap, n_rep, 1, max_iter, tolerance,
        std::span<double>(ratio_array_2.begin(), ratio_array_2.size()));
    blrt_2.SetSeed(seed_seq);
    blrt_2.Run();

    for (std::size_t i = 0; i < n_bootstrap; ++i) {
      CHECK(ratio_array[i] == ratio_array_2[i]);
    }
  }
}
