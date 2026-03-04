// poLCAParallel
// Copyright (C) 2026 Sherman Lo

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

#ifndef POLCAPARALLEL_INCLUDE_UTIL_RCPP_H_
#define POLCAPARALLEL_INCLUDE_UTIL_RCPP_H_

#include <RcppArmadillo.h>

#include <span>

namespace polca_parallel {

// Rcpp vector input

[[nodiscard]] std::span<int> VectorToSpan(Rcpp::IntegerVector& vector);

[[nodiscard]] std::span<const int> VectorToConstSpan(
    Rcpp::IntegerVector& vector);

[[nodiscard]] std::span<double> VectorToSpan(Rcpp::NumericVector& vector);

[[nodiscard]] std::span<const double> VectorToConstSpan(
    Rcpp::NumericVector& vector);

// Rcpp matrix input

[[nodiscard]] std::span<int> VectorToSpan(Rcpp::IntegerMatrix& vector);

[[nodiscard]] std::span<const int> VectorToConstSpan(
    Rcpp::IntegerMatrix& vector);

[[nodiscard]] std::span<double> VectorToSpan(Rcpp::NumericMatrix& vector);

[[nodiscard]] std::span<const double> VectorToConstSpan(
    Rcpp::NumericMatrix& vector);

}  // namespace polca_parallel

#endif  // POLCAPARALLEL_INCLUDE_UTIL_RCPP_H_
