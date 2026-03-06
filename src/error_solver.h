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

#ifndef POLCAPARALLEL_INCLUDE_ERROR_SOLVER_H
#define POLCAPARALLEL_INCLUDE_ERROR_SOLVER_H

#include <cstddef>
#include <optional>
#include <span>

#include "arma.h"

namespace polca_parallel {

/**
 * Abstract class for working out and assigning the standard errors
 *
 * Abstract class for working out and assigning the standard errors given the
 * score design matrix and the Jacobian matrix. Used by the StandardError class
 * and their derivatives, see standard_error.h.
 *
 * Pass the properties of the dataset and spans to save the resulting errors to
 * the constructor. Then call Solve(), passing the score and Jacobian matrices
 * to calculate the standard errors (and covariance where appropriate) and save
 * it at the provided spans.
 *
 * Derived classes are to implement the method Solve() to work out the standard
 * errors and save it, eg using eigen decomposition, SVD, <code>inv()</code>,
 * <code>pinv()</code>, ...etc
 */
class ErrorSolver {
 protected:
  /** Number of data points, ie height of the score matrix */
  const std::size_t n_data_;
  /** Number of features, only needed for the regression problem */
  std::optional<std::size_t> n_feature_;
  /** Sum of <code>n_outcomes</code> */
  const std::size_t sum_outcomes_;
  /** Number of clusters fitted */
  const std::size_t n_cluster_;
  /**
   * The size of the information matrix
   *
   * This is the same as the width of the score matrix and the height of the
   * Jacobian matrix
   */
  const std::size_t info_size_;
  /** The width of the Jacobian matrix */
  const std::size_t jacobian_width_;
  /**
   * Vector containing the standard error for the prior probabilities for each
   * cluster
   */
  std::span<double> prior_error_;
  /**
   * Vector containing the standard error for the outcome probabilities category
   * and cluster
   * flatten list of matrices
   * <ul>
   *   <li>dim 0: for each outcome</li>
   *   <li>dim 1: for each category</li>
   *   <li>dim 2: for each cluster</li>
   * </ul>
   */
  std::span<double> prob_error_;
  /**
   * Covariance matrix of the regression coefficient, only needed for the
   * regression problem
   */
  std::optional<arma::Mat<double>> regress_coeff_error_;

 public:
  /**
   * Constructs an ErrorSolver
   *
   * Pass the properties of the dataset and spans to save the resulting errors
   * to the constructor. Then call Solve(), passing the score and Jacobian
   * matrices to calculate the standard errors (and covariance where
   * appropriate) and save it at the provided spans.
   *
   * @param n_data Number of data points
   * @param sum_outcomes Sum of all integers in <code>n_outcomes</code>
   * @param n_cluster Number of clusters fitted
   * @param info_size The size of the information matrix
   * @param jacobian_width The width of the Jacobian matrix
   * @param prior_error Vector to contain the standard error for the prior
   * probabilities for each cluster, modified after calling Solve()
   * @param prob_error Vector to contain the standard error for the outcome
   * probabilities, conditioned on category and cluster, modified after calling
   * Solve(). Flatten list of matrices
   * <ul>
   *   <li>dim 0: for each outcome</li>
   *   <li>dim 1: for each category</li>
   *   <li>dim 2: for each cluster</li>
   * </ul>
   */
  ErrorSolver(std::size_t n_data, std::size_t sum_outcomes,
              std::size_t n_cluster, std::size_t info_size,
              std::size_t jacobian_width, std::span<double> prior_error,
              std::span<double> prob_error);

  virtual ~ErrorSolver() = default;

  /**
   * Solves equations to work out the standard error and saves it
   *
   * Solves equations to work out the standard error and saves it where the
   * member variables ErrorSolver::prior_error_, ErrorSolver::prob_error_ and
   * ErrorSolver::regress_coeff_error_ are pointing to
   *
   * @param score Score matrix with the following dimensions
   * <ul>
   *   <li>dim 0: size ErrorSolver::n_data_</li>
   *   <li>dim 1: size ErrorSolver::info_size_</li>
   * </ul>
   * @param jacobian Jacobian matrix with the following dimensions
   * <ul>
   *   <li>dim 0: size ErrorSolver::info_size_</li>
   *   <li>dim 1: size ErrorSolver::jacobian_width_</li>
   * </ul>
   */
  virtual void Solve(const arma::Mat<double>& score,
                     const arma::Mat<double>& jacobian) = 0;
};

/**
 * Calculates the standard errors from the eigencomposition of the info matrix
 *
 * Calculates the standard errors InfoEigenSolver::prior_error_ and
 * InfoEigenSolver::prob_error_. It calculates the information matrix, from the
 * score matrix, and inverts it. The information matrix is typically
 * ill-conditioned (very small positive and negative eigenvalue), hence the
 * justification for using an eigen decomposition. As with <code>pinv()</code>,
 * the inversion is done by inverting the large eigenvalues and setting the
 * small eigenvalues to zero. The root of the inverted eigenvalues are taken so
 * that the standard errors can be obtained by the root column sum of squares.
 *
 * Note: the information matrix is calculated by \f(S^T S\f) where \f(S\f) is
 * the score matrix. This may cause numerical instability as \f(S\f) is commonly
 * ill-conditioned.
 *
 * Let:
 * <ul>
 *   <li>\f(Q\f) be the matrix containing columns of eigenvectors</li>
 *   <li>\f(D\f) be a diagonal matrix of eigenvalues</li>
 *   <li>\f(J\f) be the jacobian matrix</li>
 * </ul>
 *
 * The eigendecomposition of the information matrix is given as \f(Q D Q^T\f)
 *
 * The covariance of interest is \f(J^T Q D^{-1} Q^T J\f)
 *
 * For standard errors, take root column sum of squares \f(D^{-1/2} Q^T J\f)
 */
class InfoEigenSolver : public polca_parallel::ErrorSolver {
 public:
  /**
   * @copydoc ErrorSolver::ErrorSolver
   */
  InfoEigenSolver(std::size_t n_data, std::size_t sum_outcomes,
                  std::size_t n_cluster, std::size_t info_size,
                  std::size_t jacobian_width, std::span<double> prior_error,
                  std::span<double> prob_error);

  ~InfoEigenSolver() override = default;

  void Solve(const arma::Mat<double>& score,
             const arma::Mat<double>& jacobian) override;

 protected:
  /**
   * Extract errors of interest from eigen calculations
   *
   * Extract errors of interest given the eigenvectors and inverse eigenvalues
   * of the information matrix. Saves them to the member variables such as
   * ErrorSolver::prior_error_, ErrorSolver::prob_error_ and
   * ErrorSolver::regress_coeff_error_ if applicable
   *
   * @param eigval_inv The inverse of the eigenvalues of the information matrix
   * @param eigvec Eigenvectors of the information matrix
   * @param jacobian The jacobian matrix
   */
  virtual void ExtractErrorGivenEigen(const arma::Col<double>& eigval_inv,
                                      const arma::Mat<double>& eigvec,
                                      const arma::Mat<double>& jacobian);
};

/**
 * Calculates the standard errors from the eigencomposition of the info matrix
 *
 * Calculates the standard errors InfoEigenRegressSolver::prior_error_,
 * InfoEigenRegressSolver::prob_error_ and
 * InfoEigenRegressSolver::regress_coeff_error_. It calculates the information
 * matrix, from the score matrix, and inverts it. The information matrix is
 * typically ill-conditioned (very small positive and negative eigenvalue),
 * hence the justification for using an eigen decomposition. As with
 * <code>pinv()</code>, the inversion is done by inverting the large eigenvalues
 * and setting the small eigenvalues to zero. The root of the inverted
 * eigenvalues are taken so that the standard errors can be obtained by the root
 * column sum of squares.
 *
 * The covariance matrix of the regression coefficients can be obtained directly
 * from the inverted information matrix, ie submatrix.
 *
 * Note: the information matrix is calculated by \f(S^T S\f) where \f(S\f) is
 * the score matrix. This may cause numerical instability as \f(S\f) is commonly
 * ill-conditioned.
 *
 * Let:
 * <ul>
 *   <li>\f(Q\f) be the matrix containing columns of eigenvectors</li>
 *   <li>\f(D\f) be a diagonal matrix of eigenvalues</li>
 *   <li>\f(J\f) be the jacobian matrix</li>
 * </ul>
 *
 * The eigendecomposition of the information matrix is given as \f(Q D Q^T\f)
 *
 * The covariance of interest is \f(J^T Q D^{-1} Q^T J\f)
 *
 * For standard errors, take root column sum of squares \f(D^{-1/2} Q^T J\f)
 *
 * For the regression coefficients covariance, take the top left (ie sub-matrix)
 * of the covariance of interest
 */
class InfoEigenRegressSolver : public polca_parallel::InfoEigenSolver {
 public:
  /**
   * Constructs an InfoEigenRegressSolver
   *
   * Pass the properties of the dataset and spans to save the resulting errors
   * to the constructor. Then call Solve(), passing the score and Jacobian
   * matrices to calculate the standard errors and covariance of the regression
   * coefficients. They are saved in the provided spans.
   *
   * @param n_data Number of data points
   * @param n_feature Number of features
   * @param sum_outcomes Sum of all integers in <code>n_outcomes</code>
   * @param n_cluster Number of clusters fitted
   * @param info_size The size of the information matrix
   * @param jacobian_width The width of the Jacobian matrix
   * @param prior_error Vector to contain the standard error for the prior
   * probabilities for each cluster, modified after calling Solve()
   * @param prob_error Vector to contain the standard error for the outcome
   * probabilities, conditioned on category and cluster, modified after calling
   * Solve(). Flatten list of matrices
   * <ul>
   *   <li>dim 0: for each outcome</li>
   *   <li>dim 1: for each category</li>
   *   <li>dim 2: for each cluster</li>
   * </ul>
   * @param regress_coeff_error Matrix to store the covariance matrix of the
   * regression coefficients, with dimensions
   * <ul>
   *   <li>dim 0: <code>n_feature * (n_cluster - 1)</code></li>
   *   <li>dim 1: <code>n_feature * (n_cluster - 1)</code></li>
   * </ul>
   */
  InfoEigenRegressSolver(std::size_t n_data, std::size_t n_feature,
                         std::size_t sum_outcomes, std::size_t n_cluster,
                         std::size_t info_size, std::size_t jacobian_width,
                         std::span<double> prior_error,
                         std::span<double> prob_error,
                         std::span<double> regress_coeff_error);

  ~InfoEigenRegressSolver() override = default;

 protected:
  void ExtractErrorGivenEigen(const arma::Col<double>& eigval_inv,
                              const arma::Mat<double>& eigvec,
                              const arma::Mat<double>& jacobian) override;
};

/**
 * Calculates standard errors from the eigencomposition of the info matrix
 *
 * Calculates the standard errors for ScoreSvdSolver::prior_error_ and
 * ScoreSvdSolver::prob_error_. It does an SVD decomposition of the score
 * matrix, which is typically ill-conditioned with very small positive (and
 * sometimes value zero) singular values. As with <code>pinv()</code>, the
 * inversion is done by inverting the large singular values and setting the
 * small singular values to zero. The standard errors can be obtained by the
 * root column sum of squares.
 *
 * The covariance matrix of the regression coefficients can be obtained directly
 * from the inverted information matrix, ie submatrix.
 *
 * This is supposed to be more numerically stable as it avoids doing \f(S^T S\f)
 * calculation. Benchmarks vs InfoEigenSolver varies depending on the size of
 * \f(S\f) and perhaps more.
 *
 * Let
 *
 * <ul>
 *   <li>\f(S\f) be the score matrix (size \f(n \times p\f))</li>
 *   <li>
 *       \f(U\f) be the left orthogonal matrix (not needed)
 *       (size \f(n \times n\f))
 *   </li>
 *   <li>\f(V\f) be the right orthogonal matrix (size \f(p \times p\f))</li>
 *   <li>
 *       \f(D\f) be the diagonal matrix containing singular values
 *       (size \f(n \times p\f))
 *   </li>
 *   <li>\f(J\f) be the Jacobian matrix</li>
 * </ul>
 *
 * then \f(S = U D V^T\f)
 *
 * The covariance of interest is \f(J^T (S^T S) ^{-1} J = J^T V D^{-2} V^T J\f)
 *
 * For standard errors, take root column sum of squares \f(D^{-1} V^T J\f)
 */
class ScoreSvdSolver : public polca_parallel::ErrorSolver {
 public:
  /**
   * @copydoc ErrorSolver::ErrorSolver
   */
  ScoreSvdSolver(std::size_t n_data, std::size_t sum_outcomes,
                 std::size_t n_cluster, std::size_t info_size,
                 std::size_t jacobian_width, std::span<double> prior_error,
                 std::span<double> prob_error);

  ~ScoreSvdSolver() override = default;

  void Solve(const arma::Mat<double>& score,
             const arma::Mat<double>& jacobian) override;

  /**
   * Extract errors of interest from the SVD
   *
   * Extract errors of interest given the SVD of the score matrix. Saves them to
   * the member variables such as ErrorSolver::prior_error_,
   * ErrorSolver::prob_error_ and ErrorSolver::regress_coeff_error_ if
   * applicable
   *
   *
   * @param singular_inv The inverse of the eigenvalues of the information
   * matrix
   * @param v_mat Eigenvectors of the information matrix
   * @param jacobian The jacobian matrix
   */
  virtual void ExtractErrorGivenEigen(const arma::Col<double>& singular_inv,
                                      const arma::Mat<double>& v_mat,
                                      const arma::Mat<double>& jacobian);
};

/**
 * Calculates standard errors from the eigencomposition of the info matrix
 *
 * Calculates the standard errors for the ScoreSvdRegressSolver::prior_error_,
 * ScoreSvdRegressSolver::prob_error_ and
 * ScoreSvdRegressSolver::regress_coeff_error_. It does an SVD decomposition of
 * the score matrix, which is typically ill-conditioned with very small positive
 * (and sometimes value zero) singular values. As with <code>pinv()</code>, the
 * inversion is done by inverting the large singular values and setting the
 * small singular values to zero. The standard errors can be obtained by the
 * root column sum of squares.
 *
 * This is supposed to be more numerically stable as it avoids doing \f(S^T S\f)
 * calculation. Benchmark vs InfoEigenSolver varies depending on the size of
 * \f(S\f) and perhaps more.
 *
 * Let
 *
 * <ul>
 *   <li>\f(S\f) be the score matrix (size \f(n \times p\f))</li>
 *   <li>
 *       \f(U\f) be the left orthogonal matrix (not needed)
 *       (size \f(n \times n\f))
 *   </li>
 *   <li>\f(V\f) be the right orthogonal matrix (size \f(p \times p\f))</li>
 *   <li>
 *       \f(D\f) be the diagonal matrix containing singular values
 *       (size \f(n \times p\f))
 *   </li>
 *   <li>\f(J\f) be the Jacobian matrix</li>
 * </ul>
 *
 * The covariance of interest is \f(J^T (S^T S) ^{-1} J = J^T V D^{-2} V^T J\f)
 *
 * For standard errors, take root column sum of squares \f(D^{-1} V^T J\f)
 *
 * For the regression coefficients covariance, take the top left (ie sub-matrix)
 * of the covariance of interest
 */
class ScoreSvdRegressSolver : public polca_parallel::ScoreSvdSolver {
 public:
  /**
   * Constructs an ScoreSvdRegressSolver
   *
   * Pass the properties of the dataset and spans to save the resulting errors
   * to the constructor. Then call Solve(), passing the score and Jacobian
   * matrices to calculate the standard errors and covariance of the regression
   * coefficients. They are saved in the provided spans.
   *
   * @param n_data Number of data points
   * @param n_feature Number of features
   * @param sum_outcomes Sum of all integers in <code>n_outcomes</code>
   * @param n_cluster Number of clusters fitted
   * @param info_size The size of the information matrix
   * @param jacobian_width The width of the Jacobian matrix
   * @param prior_error Vector to contain the standard error for the prior
   * probabilities for each cluster, modified after calling Solve()
   * @param prob_error Vector to contain the standard error for the outcome
   * probabilities, conditioned on category and cluster, modified after calling
   * Solve(). Flatten list of matrices
   * <ul>
   *   <li>dim 0: for each outcome</li>
   *   <li>dim 1: for each category</li>
   *   <li>dim 2: for each cluster</li>
   * </ul>
   * @param regress_coeff_error Matrix to store the covariance matrix of the
   * regression coefficients, with dimensions
   * <ul>
   *   <li>dim 0: <code>n_feature * (n_cluster - 1)</code></li>
   *   <li>dim 1: <code>n_feature * (n_cluster - 1)</code></li>
   * </ul>
   */
  ScoreSvdRegressSolver(std::size_t n_data, std::size_t n_feature,
                        std::size_t sum_outcomes, std::size_t n_cluster,
                        std::size_t info_size, std::size_t jacobian_width,
                        std::span<double> prior_error,
                        std::span<double> prob_error,
                        std::span<double> regress_coeff_error);

  ~ScoreSvdRegressSolver() override = default;

 protected:
  void ExtractErrorGivenEigen(const arma::Col<double>& singular_inv,
                              const arma::Mat<double>& v_mat_t,
                              const arma::Mat<double>& jacobian) override;
};

}  // namespace polca_parallel

#endif  // POLCAPARALLEL_INCLUDE_ERROR_SOLVER_H
