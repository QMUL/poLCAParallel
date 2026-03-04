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

// Rcpp vector input

#include "util_rcpp.h"

std::span<int> polca_parallel::VectorToSpan(Rcpp::IntegerVector& vector) {
  return std::span<int>(vector.begin(), vector.size());
}

std::span<const int> polca_parallel::VectorToConstSpan(
    Rcpp::IntegerVector& vector) {
  return std::span<const int>(vector.begin(), vector.size());
}

std::span<double> polca_parallel::VectorToSpan(Rcpp::NumericVector& vector) {
  return std::span<double>(vector.begin(), vector.size());
}

std::span<const double> polca_parallel::VectorToConstSpan(
    Rcpp::NumericVector& vector) {
  return std::span<const double>(vector.begin(), vector.size());
}

// Rcpp matrix input

std::span<int> polca_parallel::VectorToSpan(Rcpp::IntegerMatrix& vector) {
  return std::span<int>(vector.begin(), vector.size());
}

std::span<const int> polca_parallel::VectorToConstSpan(
    Rcpp::IntegerMatrix& vector) {
  return std::span<const int>(vector.begin(), vector.size());
}

std::span<double> polca_parallel::VectorToSpan(Rcpp::NumericMatrix& vector) {
  return std::span<double>(vector.begin(), vector.size());
}

std::span<const double> polca_parallel::VectorToConstSpan(
    Rcpp::NumericMatrix& vector) {
  return std::span<const double>(vector.begin(), vector.size());
}
