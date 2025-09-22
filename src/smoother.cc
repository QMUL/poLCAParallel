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

#include "smoother.h"

#include <cassert>
#include <iterator>

#include "arma.h"

polca_parallel::Smoother::Smoother(std::span<const double> probs,
                                   std::span<const double> prior,
                                   std::span<const double> posterior,
                                   std::size_t n_data,
                                   std::span<const std::size_t> n_outcomes,
                                   std::size_t n_cluster)
    : probs_(probs.begin(), probs.end()),
      prior_(prior.begin(), prior.end()),
      posterior_(posterior.begin(), posterior.end()),
      n_data_(n_data),
      n_outcomes_(n_outcomes),
      n_cluster_(n_cluster) {
  // std::vector makes a copy of the array
}

void polca_parallel::Smoother::Smooth() {
  // use posterior to get the estimate of number of data in each cluster
  arma::Mat<double> posterior(this->posterior_.data(), this->n_data_,
                              this->n_cluster_, false, true);
  arma::Row<double> n_data = arma::sum(posterior, 0);

  // smooth outcome probabilities
  auto probs = this->probs_.begin();
  for (double n_data_i : n_data) {
    for (std::size_t n_outcome_j : this->n_outcomes_) {
      assert(std::next(probs, n_outcome_j) <= this->probs_.end());
      this->Smooth(n_data_i, 1.0, static_cast<double>(n_outcome_j),
                   std::span<double>(probs, n_outcome_j));
      std::advance(probs, n_outcome_j);
    }
  }

  // perhaps smooth prior as well
  // for posterior update, use E step in EmAlgorithm
}

std::span<const double> polca_parallel::Smoother::get_probs() {
  return std::span<const double>(this->probs_);
}

std::span<const double> polca_parallel::Smoother::get_prior() {
  return std::span<const double>(this->prior_);
}

std::span<const double> polca_parallel::Smoother::get_posterior() {
  return std::span<const double>(this->posterior_);
}

void polca_parallel::Smoother::Smooth(double n_data, double num_add,
                                      double deno_add,
                                      std::span<double> probs) {
  arma::Col<double> probs_arma(probs.data(), probs.size(), false, true);
  probs_arma = (n_data * probs_arma + num_add) / (n_data + deno_add);
}
