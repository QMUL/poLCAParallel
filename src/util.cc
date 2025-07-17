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

#include <cassert>
#include <numeric>

#include "arma.h"

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
      assert(std::next(prob_i, n_outcome) <= prob.end());
      assert(response_iter < response.end());

      std::discrete_distribution<int> outcome_dist(
          prob_i, std::next(prob_i, n_outcome));
      *response_iter = outcome_dist(rng) + 1;  // response is one-based index

      assert(*response_iter > 0);
      assert(*response_iter <= static_cast<int>(n_outcome));

      // increment for the next category
      std::advance(prob_i, n_outcome);
      std::advance(response_iter, 1);
    }
  }
}

std::vector<int> polca_parallel::RandomMarginal(
    std::size_t n_data, polca_parallel::NOutcomes n_outcomes,
    std::mt19937_64& rng) {
  std::vector<int> responses(n_data * n_outcomes.size());

  auto response_iter = responses.begin();
  for (std::size_t i_data = 0; i_data < n_data; ++i_data) {
    for (auto n_outcome_i : n_outcomes) {
      assert(response_iter < responses.end());

      std::uniform_int_distribution<int> dist(1, n_outcome_i);
      *response_iter = dist(rng);

      assert(*response_iter > 0);
      assert(*response_iter <= static_cast<int>(n_outcome_i));

      std::advance(response_iter, 1);
    }
  }
  return responses;
}

void polca_parallel::RandomProb(std::span<const size_t> n_outcomes,
                                const std::size_t n_cluster,
                                std::mt19937_64& rng, arma::Mat<double>& prob) {
  std::uniform_real_distribution<double> random_distribution(0.0, 1.0);
  for (auto& prob_i : prob) {
    prob_i = random_distribution(rng);
    assert(prob_i >= 0.0);
  }
  // normalise to probabilities
  for (std::size_t m = 0; m < n_cluster; ++m) {
    auto prob_col = prob.unsafe_col(m).begin();
    for (std::size_t n_outcome_i : n_outcomes) {
      assert(std::next(prob_col, n_outcome_i) <= prob.unsafe_col(m).end());

      arma::Col<double> prob_vector(prob_col, n_outcome_i, false, true);
      prob_vector /= arma::sum(prob_vector);
      std::advance(prob_col, n_outcome_i);
    }
  }
}

std::vector<double> polca_parallel::RandomInitialProb(
    polca_parallel::NOutcomes n_outcomes, const std::size_t n_cluster,
    std::size_t n_rep, std::mt19937_64& rng) {
  std::vector<double> initial_prob(n_rep * n_cluster * n_outcomes.sum());
  auto initial_prob_iter = initial_prob.begin();
  for (std::size_t i_rep = 0; i_rep < n_rep; ++i_rep) {
    assert(std::next(initial_prob_iter, n_outcomes.sum() * n_cluster) <=
           initial_prob.end());

    arma::Mat<double> prob_i(&*initial_prob_iter, n_outcomes.sum(), n_cluster,
                             false, true);
    polca_parallel::RandomProb(n_outcomes, n_cluster, rng, prob_i);
    std::advance(initial_prob_iter, n_outcomes.sum() * n_cluster);
  }
  return initial_prob;
}

void polca_parallel::SetMissingAtRandom(double missing_prob,
                                        std::mt19937_64& rng,
                                        std::span<int> responses) {
  std::bernoulli_distribution missing_dist(missing_prob);
  for (auto& response : responses) {
    if (missing_dist(rng)) {
      response = 0;
    }
    assert(response >= 0);
  }
}
