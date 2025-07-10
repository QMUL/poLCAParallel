// poLCAParallel
// Copyright (C) 2024 Sherman Lo

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

#include "util.h"

#include <numeric>

polca_parallel::NOutcomes::NOutcomes(const std::size_t* data, std::size_t size)
    : std::span<const std::size_t>(data, size),
      sum_(std::accumulate(data, data + size, 0)) {}

std::size_t polca_parallel::NOutcomes::sum() const { return this->sum_; }

void polca_parallel::Random(std::span<const double> prior,
                            std::span<const double> prob, std::size_t n_data,
                            NOutcomes n_outcomes, std::mt19937_64& rng,
                            std::span<int> response) {
  std::discrete_distribution<std::size_t> prior_dist(prior.begin(),
                                                     prior.end());

  auto response_iter = response.begin();
  for (std::size_t i_data = 0; i_data < n_data; ++i_data) {
    std::size_t i_cluster = prior_dist(rng);  // select a random cluster
    // point to the corresponding probabilites for this random cluster
    auto prob_i = prob.begin();
    std::advance(prob_i, i_cluster * n_outcomes.sum());

    for (std::size_t n_outcome : n_outcomes) {
      std::discrete_distribution<int> outcome_dist(
          prob_i, std::next(prob_i, n_outcome));
      *response_iter = outcome_dist(rng) + 1;  // response is one-based index
      // increment for the next category
      std::advance(prob_i, n_outcome);
      std::advance(response_iter, 1);
    }
  }
}
