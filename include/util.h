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

#ifndef POLCAPARALLEL_INCLUDE_UTIL_H_
#define POLCAPARALLEL_INCLUDE_UTIL_H_

#include <cstddef>
#include <random>
#include <span>

namespace polca_parallel {

class NOutcomes : public std::span<const std::size_t> {
 private:
  const std::size_t sum_;

 public:
  NOutcomes(const std::size_t* data, std::size_t size);

  [[nodiscard]] std::size_t sum() const;
};

/**
 * Generate random responses
 *
 * Generate random responses given the prior and outcome probabilities. Provide
 * a rng and the resulting random responses are saved to the given response
 * span
 *
 * @param prior Vector of prior probabilities
 * <ul>
 *   <li>dim 0: for each cluster</li>
 * </ul>
 * @param prob Vector of response probabilities for each category, flatten list
 * of matrices.
 * <ul>
 *   <li>dim 0: for each outcome</li>
 *   <li>dim 1: for each category</li>
 *   <li>dim 2: for each cluster</li>
 * </ul>
 * @param n_data Number of data points
 * @param n_outcomes Number of outcomes for each category
 * @param rng Random number generator
 * @param response To store results, design matrix transpose of responses
 * <ul>
 *   <li>dim 0: for each category</li>
 *   <li>dim 1: for each data point</li>
 * </ul>
 */
void Random(std::span<const double> prior, std::span<const double> prob,
            std::size_t n_data, NOutcomes n_outcomes, std::mt19937_64& rng,
            std::span<int> response);

}  // namespace polca_parallel

#endif  // POLCAPARALLEL_INCLUDE_UTIL_H_
