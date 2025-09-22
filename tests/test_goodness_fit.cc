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
 * Test the class polca_parallel::GoodnessOfFit
 *
 * Test the class polca_parallel::GoodnessOfFit. Test the output frequency map
 * and the statistics. For the frequency map, it tests if the sum of observed
 * frequency is the same as the number of observed data
 */

#define CATCH_CONFIG_MAIN

#include <catch2/catch_all.hpp>
#include <cmath>
#include <map>
#include <random>
#include <vector>

#include "goodness_fit.h"
#include "util.h"
#include "util_test.h"

TEST_CASE("goodness-of-fit", "[gof]") {
  std::size_t n_data = GENERATE(10, 100);
  std::size_t n_category = GENERATE(2, 5);
  std::size_t max_n_outcome = GENERATE(2, 5);
  std::size_t n_cluster = GENERATE(2, 5);
  double missing_prob = GENERATE(0, 0.1);

  std::vector<unsigned> rng_seed_array = {2416726670, 1062543125, 977709725};
  std::mt19937_64 rng = polca_parallel_test::InitRng(rng_seed_array);

  std::vector<std::size_t> n_outcomes_vec(n_category);
  polca_parallel::NOutcomes n_outcomes =
      polca_parallel_test::RandomNOutcomes(max_n_outcome, n_outcomes_vec, rng);

  std::vector<double> probs =
      polca_parallel::RandomInitialProb(n_outcomes, n_cluster, 1, rng);

  std::vector<int> responses =
      polca_parallel_test::RandomMarginal(n_data, n_outcomes, rng);

  polca_parallel_test::SetMissingAtRandom(missing_prob, rng, responses);

  arma::Mat<double> prior =
      polca_parallel_test::RandomClusterProbs(1, n_cluster, rng);

  polca_parallel::GoodnessOfFit goodness_of_fit;
  goodness_of_fit.Calc(responses, prior, probs, n_data, n_outcomes, n_cluster);

  std::map<std::vector<int>, polca_parallel::Frequency>& frequency_map =
      goodness_of_fit.GetFrequencyMap();

  std::size_t n_obs_true =
      polca_parallel_test::CalcNObs(responses, n_data, n_category);

  // test if the sum of observed frequency from the frequency map is the same as
  // the number of observed data
  //
  // test the sum of expected frequency too but this need not necessarily sum
  // to the number of observed data. This is because the frequency table does
  // not include unobserved data, where the probability of observing them can be
  // non-zero

  std::size_t n_obs = 0;
  double n_exp = 0.0;
  for (auto iter = frequency_map.cbegin(); iter != frequency_map.cend();
       ++iter) {
    const std::vector<int>& response_i = iter->first;

    for (std::size_t i = 0; i < response_i.size(); ++i) {
      CHECK(response_i[i] > 0);
      CHECK(response_i[i] <= n_outcomes[i]);
    }

    polca_parallel::Frequency frequency = iter->second;
    CHECK(frequency.observed >= 0);
    CHECK(frequency.expected >= 0.0);
    CHECK(frequency.observed <= n_obs_true);
    CHECK(frequency.expected <= static_cast<double>(n_obs_true));
    n_obs += frequency.observed;
    n_exp += frequency.expected;
  }

  REQUIRE(n_obs == n_obs_true);
  REQUIRE(static_cast<double>(n_obs_true) - n_exp >
          -polca_parallel_test::TOLERANCE);

  auto [ln_l_ratio, chi_squared] = goodness_of_fit.GetStatistics();
  REQUIRE(std::isfinite(ln_l_ratio));
  REQUIRE(std::isfinite(chi_squared));
}
