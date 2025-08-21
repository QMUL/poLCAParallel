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

#include "util_test.h"

#include <cassert>
#include <cmath>
#include <iterator>
#include <type_traits>

#include "em_algorithm_array_serial.h"
#include "em_algorithm_nan.h"
#include "em_algorithm_regress.h"
#include "regularised_error.h"
#include "standard_error.h"
#include "standard_error_regress.h"

std::size_t polca_parallel_test::CalcNObs(std::span<const int> responses,
                                          std::size_t n_data,
                                          std::size_t n_category) {
  std::size_t n_obs = 0;
  auto response_i = responses.begin();
  for (std::size_t i_data = 0; i_data < n_data; ++i_data) {
    bool is_fully_observed = true;
    for (std::size_t i_category = 0; i_category < n_category; ++i_category) {
      assert(response_i < responses.end());
      if (*response_i == 0) {
        is_fully_observed = false;
      }
      std::advance(response_i, 1);
    }
    n_obs += is_fully_observed;
  }
  return n_obs;
}

std::vector<int> polca_parallel_test::RandomMarginal(
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

void polca_parallel_test::SetMissingAtRandom(double missing_prob,
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

std::mt19937_64 polca_parallel_test::InitRng(
    std::vector<unsigned>& seed_array) {
  std::seed_seq seed_seq(seed_array.cbegin(), seed_array.cend());
  return polca_parallel_test::InitRng(seed_seq);
}

std::mt19937_64 polca_parallel_test::InitRng(std::seed_seq& seed_seq) {
  unsigned int seed;
  seed_seq.generate(&seed, &seed + 1);
  return std::mt19937_64(seed);
}

polca_parallel::NOutcomes polca_parallel_test::RandomNOutcomes(
    std::size_t max_n_outcome, std::vector<std::size_t>& n_outcomes_vec,
    std::mt19937_64& rng) {
  std::uniform_int_distribution<std::size_t> dist(2, max_n_outcome);
  for (auto& i : n_outcomes_vec) {
    i = dist(rng);
  }
  return polca_parallel::NOutcomes(n_outcomes_vec.data(),
                                   n_outcomes_vec.size());
}

arma::Mat<double> polca_parallel_test::RandomClusterProbs(
    std::size_t n_data, std::size_t n_cluster, std::mt19937_64& rng) {
  arma::Mat<double> cluster_probs(n_data, n_cluster);
  std::uniform_real_distribution<double> random_distribution(0.0, 1.0);
  for (auto& i : cluster_probs) {
    i = random_distribution(rng);
  }
  for (std::size_t i = 0; i < n_data; ++i) {
    cluster_probs.row(i) /= arma::sum(cluster_probs.row(i));
  }
  return cluster_probs;
}

std::tuple<std::vector<double>, std::vector<double>, std::vector<double>,
           std::vector<double>>
polca_parallel_test::InitOutputs(std::size_t n_data, std::size_t n_feature,
                                 polca_parallel::NOutcomes n_outcomes,
                                 std::size_t n_cluster) {
  std::vector<double> posterior(n_data * n_cluster);
  std::vector<double> prior(n_data * n_cluster);
  std::vector<double> estimated_prob(n_outcomes.sum() * n_cluster);
  std::vector<double> regress_coeff(n_feature * (n_cluster - 1));
  return {posterior, prior, estimated_prob, regress_coeff};
}

void polca_parallel_test::TestOutcomeProbs(polca_parallel::NOutcomes n_outcomes,
                                           std::size_t n_cluster,
                                           std::span<const double> probs) {
  for (auto i : probs) {
    CHECK(0.0 <= i);
    CHECK(i <= 1.0 + polca_parallel_test::TOLERANCE);
  }
  auto prob_i = probs.begin();
  for (std::size_t m = 0; m < n_cluster; ++m) {
    for (std::size_t n_outcome : n_outcomes) {
      double sum = std::accumulate(prob_i, std::next(prob_i, n_outcome), 0.0);
      CHECK(std::abs(sum - 1.0) < polca_parallel_test::TOLERANCE);
      std::advance(prob_i, n_outcome);
    }
  }
}

void polca_parallel_test::TestClusterProbs(
    std::span<const double> cluster_probs, std::size_t n_data,
    std::size_t n_cluster) {
  for (auto i : cluster_probs) {
    CHECK(0.0 <= i);
    CHECK(i <= 1.0);
  }
  arma::Mat<double> cluster_probs_arma(
      const_cast<double*>(cluster_probs.data()), n_data, n_cluster, false,
      true);
  arma::Col<double> row_sum = arma::sum(cluster_probs_arma, 1);
  for (auto i : row_sum) {
    CHECK(std::abs(i - 1.0) < polca_parallel_test::TOLERANCE);
  }
}

template void
polca_parallel_test::TestDefaultOutputs<polca_parallel::EmAlgorithm>(
    std::size_t n_data, polca_parallel::NOutcomes n_outcomes,
    std::size_t n_cluster, std::span<const double> posterior,
    std::span<const double> prior, std::span<const double> estimated_prob,
    std::span<const double> regress_coeff);
template void
polca_parallel_test::TestDefaultOutputs<polca_parallel::EmAlgorithmNan>(
    std::size_t n_data, polca_parallel::NOutcomes n_outcomes,
    std::size_t n_cluster, std::span<const double> posterior,
    std::span<const double> prior, std::span<const double> estimated_prob,
    std::span<const double> regress_coeff);
template void
polca_parallel_test::TestDefaultOutputs<polca_parallel::EmAlgorithmRegress>(
    std::size_t n_data, polca_parallel::NOutcomes n_outcomes,
    std::size_t n_cluster, std::span<const double> posterior,
    std::span<const double> prior, std::span<const double> estimated_prob,
    std::span<const double> regress_coeff);
template void
polca_parallel_test::TestDefaultOutputs<polca_parallel::EmAlgorithmNanRegress>(
    std::size_t n_data, polca_parallel::NOutcomes n_outcomes,
    std::size_t n_cluster, std::span<const double> posterior,
    std::span<const double> prior, std::span<const double> estimated_prob,
    std::span<const double> regress_coeff);

template <typename EmAlgorithmType>
void polca_parallel_test::TestDefaultOutputs(
    std::size_t n_data, polca_parallel::NOutcomes n_outcomes,
    std::size_t n_cluster, std::span<const double> posterior,
    std::span<const double> prior, std::span<const double> estimated_prob,
    std::span<const double> regress_coeff) {
  polca_parallel_test::TestClusterProbs(posterior, n_data, n_cluster);
  polca_parallel_test::TestClusterProbs(prior, n_data, n_cluster);
  polca_parallel_test::TestOutcomeProbs(n_outcomes, n_cluster, estimated_prob);

  // in the non-regression problem, the prior is the same for all data points
  if constexpr (std::is_same_v<EmAlgorithmType, polca_parallel::EmAlgorithm> ||
                std::is_same_v<EmAlgorithmType,
                               polca_parallel::EmAlgorithmNan>) {
    auto prior_i = prior.begin();
    for (std::size_t m = 0; m < n_cluster; ++m) {
      double prior_0 = *prior_i;
      std::advance(prior_i, 1);
      for (std::size_t i = 1; i < n_data; ++i) {
        CHECK(prior_0 == *prior_i);
        std::advance(prior_i, 1);
      }
    }
  } else {  // regression problem here
    for (auto i : regress_coeff) {
      CHECK(!std::isnan(i));
    }
  }
}

void polca_parallel_test::TestEmAlgorithmOptionalOutputs(
    polca_parallel::EmAlgorithm& fitter, std::size_t max_iter) {
  double ln_l = fitter.get_ln_l();
  unsigned int n_iter = fitter.get_n_iter();

  REQUIRE(ln_l <= 0.0);
  REQUIRE(0 <= n_iter);
  REQUIRE(n_iter <= max_iter);
}

template void
polca_parallel_test::BlackBoxTestEmAlgorithm<polca_parallel::EmAlgorithm>(
    std::span<const double> features, std::span<const int> responses,
    std::span<const double> initial_prob, std::size_t n_data,
    std::size_t n_feature, polca_parallel::NOutcomes n_outcomes,
    std::size_t n_cluster, unsigned int max_iter, double tolerance,
    unsigned int seed, bool is_full_constructor);
template void polca_parallel_test::BlackBoxTestEmAlgorithm<
    polca_parallel::EmAlgorithmRegress>(
    std::span<const double> features, std::span<const int> responses,
    std::span<const double> initial_prob, std::size_t n_data,
    std::size_t n_feature, polca_parallel::NOutcomes n_outcomes,
    std::size_t n_cluster, unsigned int max_iter, double tolerance,
    unsigned int seed, bool is_full_constructor);
template void
polca_parallel_test::BlackBoxTestEmAlgorithm<polca_parallel::EmAlgorithmNan>(
    std::span<const double> features, std::span<const int> responses,
    std::span<const double> initial_prob, std::size_t n_data,
    std::size_t n_feature, polca_parallel::NOutcomes n_outcomes,
    std::size_t n_cluster, unsigned int max_iter, double tolerance,
    unsigned int seed, bool is_full_constructor);
template void polca_parallel_test::BlackBoxTestEmAlgorithm<
    polca_parallel::EmAlgorithmNanRegress>(
    std::span<const double> features, std::span<const int> responses,
    std::span<const double> initial_prob, std::size_t n_data,
    std::size_t n_feature, polca_parallel::NOutcomes n_outcomes,
    std::size_t n_cluster, unsigned int max_iter, double tolerance,
    unsigned int seed, bool is_full_constructor);

template <typename EmAlgorithmType>
void polca_parallel_test::BlackBoxTestEmAlgorithm(
    std::span<const double> features, std::span<const int> responses,
    std::span<const double> initial_prob, std::size_t n_data,
    std::size_t n_feature, polca_parallel::NOutcomes n_outcomes,
    std::size_t n_cluster, unsigned int max_iter, double tolerance,
    unsigned int seed, bool is_full_constructor) {
  auto [posterior, prior, estimated_prob, regress_coeff] =
      polca_parallel_test::InitOutputs(n_data, n_feature, n_outcomes,
                                       n_cluster);

  std::unique_ptr<polca_parallel::EmAlgorithm> fitter;

  if constexpr (std::is_same_v<EmAlgorithmType, polca_parallel::EmAlgorithm> ||
                std::is_same_v<EmAlgorithmType,
                               polca_parallel::EmAlgorithmNan>) {
    if (is_full_constructor) {
      fitter = std::make_unique<EmAlgorithmType>(
          features, responses, initial_prob, n_data, n_feature, n_outcomes,
          n_cluster, max_iter, tolerance,
          std::span<double>(posterior.begin(), posterior.size()),
          std::span<double>(prior.begin(), prior.size()),
          std::span<double>(estimated_prob.begin(), estimated_prob.size()),
          std::span<double>(regress_coeff.begin(), regress_coeff.size()));
    } else {
      fitter = std::make_unique<EmAlgorithmType>(
          responses, initial_prob, n_data, n_outcomes, n_cluster, max_iter,
          tolerance, std::span<double>(posterior.begin(), posterior.size()),
          std::span<double>(prior.begin(), prior.size()),
          std::span<double>(estimated_prob.begin(), estimated_prob.size()));
    }
  } else {
    fitter = std::make_unique<EmAlgorithmType>(
        features, responses, initial_prob, n_data, n_feature, n_outcomes,
        n_cluster, max_iter, tolerance,
        std::span<double>(posterior.begin(), posterior.size()),
        std::span<double>(prior.begin(), prior.size()),
        std::span<double>(estimated_prob.begin(), estimated_prob.size()),
        std::span<double>(regress_coeff.begin(), regress_coeff.size()));
  }

  SECTION("test-outputs") {
    fitter->set_seed(seed);
    fitter->Fit();
    polca_parallel_test::TestDefaultOutputs<EmAlgorithmType>(
        n_data, n_outcomes, n_cluster, posterior, prior, estimated_prob,
        regress_coeff);
    polca_parallel_test::TestEmAlgorithmOptionalOutputs(*fitter, max_iter);
  }

  SECTION("test-further-optional-outputs") {
    std::vector<double> best_initial_prob(n_outcomes.sum() * n_cluster);
    fitter->set_best_initial_prob(
        std::span<double>(best_initial_prob.begin(), best_initial_prob.size()));

    fitter->set_seed(seed);
    fitter->Fit();
    polca_parallel_test::TestDefaultOutputs<EmAlgorithmType>(
        n_data, n_outcomes, n_cluster, posterior, prior, estimated_prob,
        regress_coeff);
    polca_parallel_test::TestEmAlgorithmOptionalOutputs(*fitter, max_iter);
    polca_parallel_test::TestOutcomeProbs(
        n_outcomes, n_cluster,
        std::span<double>(best_initial_prob.begin(), best_initial_prob.size()));

    SECTION("test-reproducible") {
      auto [posterior_2, prior_2, estimated_prob_2, regress_coeff_2] =
          polca_parallel_test::InitOutputs(n_data, n_feature, n_outcomes,
                                           n_cluster);
      std::vector<double> best_initial_prob_2(n_outcomes.sum() * n_cluster);

      std::unique_ptr<polca_parallel::EmAlgorithm> fitter_2 =
          std::make_unique<EmAlgorithmType>(
              features, responses, initial_prob, n_data, n_feature, n_outcomes,
              n_cluster, max_iter, tolerance,
              std::span<double>(posterior_2.begin(), posterior_2.size()),
              std::span<double>(prior_2.begin(), prior_2.size()),
              std::span<double>(estimated_prob_2.begin(),
                                estimated_prob_2.size()),
              std::span<double>(regress_coeff_2.begin(),
                                regress_coeff_2.size()));

      fitter_2->set_best_initial_prob(std::span<double>(
          best_initial_prob_2.begin(), best_initial_prob_2.size()));

      // differ by assigning rng rather than seed_seq
      std::unique_ptr<std::mt19937_64> rng_2 =
          std::make_unique<std::mt19937_64>(seed);
      fitter_2->set_rng(std::move(rng_2));
      fitter_2->Fit();

      // test results are all the same
      for (std::size_t i = 0; i < posterior.size(); ++i) {
        CHECK(posterior[i] == posterior_2[i]);
      }
      for (std::size_t i = 0; i < prior.size(); ++i) {
        CHECK(prior[i] == prior_2[i]);
      }
      for (std::size_t i = 0; i < estimated_prob.size(); ++i) {
        CHECK(estimated_prob[i] == estimated_prob_2[i]);
      }

      if constexpr (std::is_same_v<EmAlgorithmType,
                                   polca_parallel::EmAlgorithmRegress> ||
                    std::is_same_v<EmAlgorithmType,
                                   polca_parallel::EmAlgorithmNanRegress>) {
        for (std::size_t i = 0; i < regress_coeff.size(); ++i) {
          CHECK(regress_coeff[i] == regress_coeff_2[i]);
        }
      }
      REQUIRE(fitter->get_ln_l() == fitter_2->get_ln_l());
      REQUIRE(fitter->get_n_iter() == fitter_2->get_n_iter());
      REQUIRE(fitter->get_has_restarted() == fitter_2->get_has_restarted());
      for (std::size_t i = 0; i < best_initial_prob_2.size(); ++i) {
        CHECK(best_initial_prob[i] == best_initial_prob_2[i]);
      }

      SECTION("test-move-rng") {
        // test move_rng() and the rngs have the same internal states
        std::unique_ptr<std::mt19937_64> rng = fitter->move_rng();
        rng_2 = fitter_2->move_rng();
        REQUIRE(*rng == *rng_2);
      }
    }
  }
}

void polca_parallel_test::TestEmAlgorithmArrayOptionalOutputs(
    std::unique_ptr<polca_parallel::EmAlgorithmArray>& fitter,
    std::size_t n_rep, std::size_t max_iter) {
  std::size_t best_rep_index = fitter->get_best_rep_index();
  unsigned int n_iter = fitter->get_n_iter();

  REQUIRE(0 <= best_rep_index);
  REQUIRE(best_rep_index < n_rep);
  REQUIRE(0 <= n_iter);
  REQUIRE(n_iter <= max_iter);
}

template void polca_parallel_test::BlackBoxTestEmAlgorithmArray<
    polca_parallel::EmAlgorithmArray, polca_parallel::EmAlgorithm>(
    std::span<const double> features, std::span<const int> responses,
    std::span<const double> initial_prob, std::size_t n_data,
    std::size_t n_feature, polca_parallel::NOutcomes n_outcomes,
    std::size_t n_cluster, std::size_t n_rep, std::size_t n_thread,
    unsigned int max_iter, double tolerance, std::seed_seq& seed_seq,
    bool is_full_constructor);

template void polca_parallel_test::BlackBoxTestEmAlgorithmArray<
    polca_parallel::EmAlgorithmArray, polca_parallel::EmAlgorithmRegress>(
    std::span<const double> features, std::span<const int> responses,
    std::span<const double> initial_prob, std::size_t n_data,
    std::size_t n_feature, polca_parallel::NOutcomes n_outcomes,
    std::size_t n_cluster, std::size_t n_rep, std::size_t n_thread,
    unsigned int max_iter, double tolerance, std::seed_seq& seed_seq,
    bool is_full_constructor);

template void polca_parallel_test::BlackBoxTestEmAlgorithmArray<
    polca_parallel::EmAlgorithmArray, polca_parallel::EmAlgorithmNan>(
    std::span<const double> features, std::span<const int> responses,
    std::span<const double> initial_prob, std::size_t n_data,
    std::size_t n_feature, polca_parallel::NOutcomes n_outcomes,
    std::size_t n_cluster, std::size_t n_rep, std::size_t n_thread,
    unsigned int max_iter, double tolerance, std::seed_seq& seed_seq,
    bool is_full_constructor);

template void polca_parallel_test::BlackBoxTestEmAlgorithmArray<
    polca_parallel::EmAlgorithmArray, polca_parallel::EmAlgorithmNanRegress>(
    std::span<const double> features, std::span<const int> responses,
    std::span<const double> initial_prob, std::size_t n_data,
    std::size_t n_feature, polca_parallel::NOutcomes n_outcomes,
    std::size_t n_cluster, std::size_t n_rep, std::size_t n_thread,
    unsigned int max_iter, double tolerance, std::seed_seq& seed_seq,
    bool is_full_constructor);

template void polca_parallel_test::BlackBoxTestEmAlgorithmArray<
    polca_parallel::EmAlgorithmArraySerial, polca_parallel::EmAlgorithm>(
    std::span<const double> features, std::span<const int> responses,
    std::span<const double> initial_prob, std::size_t n_data,
    std::size_t n_feature, polca_parallel::NOutcomes n_outcomes,
    std::size_t n_cluster, std::size_t n_rep, std::size_t n_thread,
    unsigned int max_iter, double tolerance, std::seed_seq& seed_seq,
    bool is_full_constructor);

template void polca_parallel_test::BlackBoxTestEmAlgorithmArray<
    polca_parallel::EmAlgorithmArraySerial, polca_parallel::EmAlgorithmRegress>(
    std::span<const double> features, std::span<const int> responses,
    std::span<const double> initial_prob, std::size_t n_data,
    std::size_t n_feature, polca_parallel::NOutcomes n_outcomes,
    std::size_t n_cluster, std::size_t n_rep, std::size_t n_thread,
    unsigned int max_iter, double tolerance, std::seed_seq& seed_seq,
    bool is_full_constructor);

template void polca_parallel_test::BlackBoxTestEmAlgorithmArray<
    polca_parallel::EmAlgorithmArraySerial, polca_parallel::EmAlgorithmNan>(
    std::span<const double> features, std::span<const int> responses,
    std::span<const double> initial_prob, std::size_t n_data,
    std::size_t n_feature, polca_parallel::NOutcomes n_outcomes,
    std::size_t n_cluster, std::size_t n_rep, std::size_t n_thread,
    unsigned int max_iter, double tolerance, std::seed_seq& seed_seq,
    bool is_full_constructor);

template void polca_parallel_test::BlackBoxTestEmAlgorithmArray<
    polca_parallel::EmAlgorithmArraySerial,
    polca_parallel::EmAlgorithmNanRegress>(
    std::span<const double> features, std::span<const int> responses,
    std::span<const double> initial_prob, std::size_t n_data,
    std::size_t n_feature, polca_parallel::NOutcomes n_outcomes,
    std::size_t n_cluster, std::size_t n_rep, std::size_t n_thread,
    unsigned int max_iter, double tolerance, std::seed_seq& seed_seq,
    bool is_full_constructor);

template <typename EmAlgorithmArrayType, typename EmAlgorithmType>
void polca_parallel_test::BlackBoxTestEmAlgorithmArray(
    std::span<const double> features, std::span<const int> responses,
    std::span<const double> initial_prob, std::size_t n_data,
    std::size_t n_feature, polca_parallel::NOutcomes n_outcomes,
    std::size_t n_cluster, std::size_t n_rep, std::size_t n_thread,
    unsigned int max_iter, double tolerance, std::seed_seq& seed_seq,
    bool is_full_constructor) {
  auto [posterior, prior, estimated_prob, regress_coeff] =
      polca_parallel_test::InitOutputs(n_data, n_feature, n_outcomes,
                                       n_cluster);

  std::unique_ptr<polca_parallel::EmAlgorithmArray> fitter;

  if constexpr (std::is_same_v<EmAlgorithmArrayType,
                               polca_parallel::EmAlgorithmArray>) {
    if (is_full_constructor) {
      fitter = std::make_unique<EmAlgorithmArrayType>(
          features, responses, initial_prob, n_data, n_feature, n_outcomes,
          n_cluster, n_rep, n_thread, max_iter, tolerance,
          std::span<double>(posterior.begin(), posterior.size()),
          std::span<double>(prior.begin(), prior.size()),
          std::span<double>(estimated_prob.begin(), estimated_prob.size()),
          std::span<double>(regress_coeff.begin(), regress_coeff.size()));
    } else {
      fitter = std::make_unique<EmAlgorithmArrayType>(
          responses, initial_prob, n_data, n_outcomes, n_cluster, n_rep,
          n_thread, max_iter, tolerance,
          std::span<double>(posterior.begin(), posterior.size()),
          std::span<double>(prior.begin(), prior.size()),
          std::span<double>(estimated_prob.begin(), estimated_prob.size()));
    }
  } else {
    if (is_full_constructor) {
      fitter = std::make_unique<EmAlgorithmArrayType>(
          features, responses, initial_prob, n_data, n_feature, n_outcomes,
          n_cluster, n_rep, max_iter, tolerance,
          std::span<double>(posterior.begin(), posterior.size()),
          std::span<double>(prior.begin(), prior.size()),
          std::span<double>(estimated_prob.begin(), estimated_prob.size()),
          std::span<double>(regress_coeff.begin(), regress_coeff.size()));
    } else {
      fitter = std::make_unique<EmAlgorithmArrayType>(
          responses, initial_prob, n_data, n_outcomes, n_cluster, n_rep,
          max_iter, tolerance,
          std::span<double>(posterior.begin(), posterior.size()),
          std::span<double>(prior.begin(), prior.size()),
          std::span<double>(estimated_prob.begin(), estimated_prob.size()));
    }
  }

  SECTION("test-outputs") {
    fitter->SetSeed(seed_seq);
    fitter->Fit<EmAlgorithmType>();
    polca_parallel_test::TestDefaultOutputs<EmAlgorithmType>(
        n_data, n_outcomes, n_cluster, posterior, prior, estimated_prob,
        regress_coeff);
    polca_parallel_test::TestEmAlgorithmArrayOptionalOutputs(fitter, n_rep,
                                                             max_iter);
  }

  SECTION("test-further-optional-outputs") {
    std::vector<double> ln_l_array(n_rep);
    std::vector<double> best_initial_prob(n_outcomes.sum() * n_cluster);
    fitter->set_best_initial_prob(
        std::span<double>(best_initial_prob.begin(), best_initial_prob.size()));
    fitter->set_ln_l_array(
        std::span<double>(ln_l_array.begin(), ln_l_array.size()));
    fitter->SetSeed(seed_seq);
    fitter->Fit<EmAlgorithmType>();
    polca_parallel_test::TestDefaultOutputs<EmAlgorithmType>(
        n_data, n_outcomes, n_cluster, posterior, prior, estimated_prob,
        regress_coeff);
    polca_parallel_test::TestEmAlgorithmArrayOptionalOutputs(fitter, n_rep,
                                                             max_iter);
    polca_parallel_test::TestOutcomeProbs(n_outcomes, n_cluster,
                                          best_initial_prob);

    double best_ln_l = ln_l_array[fitter->get_best_rep_index()];
    for (double ln_l : ln_l_array) {
      CHECK(ln_l <= 0);
      CHECK(ln_l <= best_ln_l);
    }

    SECTION("test-reproducible") {
      auto [posterior_2, prior_2, estimated_prob_2, regress_coeff_2] =
          polca_parallel_test::InitOutputs(n_data, n_feature, n_outcomes,
                                           n_cluster);
      std::vector<double> ln_l_array_2(n_rep);
      std::vector<double> best_initial_prob_2(n_outcomes.sum() * n_cluster);

      n_thread = 1;
      std::unique_ptr<polca_parallel::EmAlgorithmArray> fitter_2;

      if constexpr (std::is_same_v<EmAlgorithmArrayType,
                                   polca_parallel::EmAlgorithmArray>) {
        fitter_2 = std::make_unique<EmAlgorithmArrayType>(
            features, responses, initial_prob, n_data, n_feature, n_outcomes,
            n_cluster, n_rep, n_thread, max_iter, tolerance,
            std::span<double>(posterior_2.begin(), posterior_2.size()),
            std::span<double>(prior_2.begin(), prior_2.size()),
            std::span<double>(estimated_prob_2.begin(),
                              estimated_prob_2.size()),
            std::span<double>(regress_coeff_2.begin(), regress_coeff_2.size()));
      } else {
        fitter_2 = std::make_unique<EmAlgorithmArrayType>(
            features, responses, initial_prob, n_data, n_feature, n_outcomes,
            n_cluster, n_rep, max_iter, tolerance,
            std::span<double>(posterior_2.begin(), posterior_2.size()),
            std::span<double>(prior_2.begin(), prior_2.size()),
            std::span<double>(estimated_prob_2.begin(),
                              estimated_prob_2.size()),
            std::span<double>(regress_coeff_2.begin(), regress_coeff_2.size()));
      }

      fitter_2->SetSeed(seed_seq);
      fitter_2->set_best_initial_prob(std::span<double>(
          best_initial_prob_2.begin(), best_initial_prob_2.size()));
      fitter_2->set_ln_l_array(
          std::span<double>(ln_l_array_2.begin(), ln_l_array_2.size()));

      fitter_2->Fit<EmAlgorithmType>();

      // test results are all the same
      for (std::size_t i = 0; i < posterior.size(); ++i) {
        CHECK(posterior[i] == posterior_2[i]);
      }
      for (std::size_t i = 0; i < prior.size(); ++i) {
        CHECK(prior[i] == prior_2[i]);
      }
      for (std::size_t i = 0; i < estimated_prob.size(); ++i) {
        CHECK(estimated_prob[i] == estimated_prob_2[i]);
      }
      if constexpr (std::is_same_v<EmAlgorithmType,
                                   polca_parallel::EmAlgorithmRegress> ||
                    std::is_same_v<EmAlgorithmType,
                                   polca_parallel::EmAlgorithmNanRegress>) {
        for (std::size_t i = 0; i < regress_coeff.size(); ++i) {
          CHECK(regress_coeff[i] == regress_coeff_2[i]);
        }
      }
      for (std::size_t i = 0; i < ln_l_array.size(); ++i) {
        CHECK(ln_l_array[i] == ln_l_array_2[i]);
      }
      for (std::size_t i = 0; i < best_initial_prob.size(); ++i) {
        CHECK(best_initial_prob[i] == best_initial_prob_2[i]);
      }
      CHECK(fitter->get_best_rep_index() == fitter_2->get_best_rep_index());
      CHECK(fitter->get_optimal_ln_l() == fitter_2->get_optimal_ln_l());
      CHECK(fitter->get_n_iter() == fitter_2->get_n_iter());
      CHECK(fitter->get_has_restarted() == fitter_2->get_has_restarted());
    }
  }
}

template void polca_parallel_test::BlackBoxTestStandardError<
    typename polca_parallel::StandardError>(
    std::span<const double> features, std::span<const int> responses,
    std::span<const double> probs, const arma::Mat<double>& posterior,
    const arma::Mat<double>& prior, std::size_t n_data, std::size_t n_feature,
    polca_parallel::NOutcomes n_outcomes, std::size_t n_cluster,
    bool is_full_constructor);

template void polca_parallel_test::BlackBoxTestStandardError<
    typename polca_parallel::StandardErrorRegress>(
    std::span<const double> features, std::span<const int> responses,
    std::span<const double> probs, const arma::Mat<double>& posterior,
    const arma::Mat<double>& prior, std::size_t n_data, std::size_t n_feature,
    polca_parallel::NOutcomes n_outcomes, std::size_t n_cluster,
    bool is_full_constructor);

template void polca_parallel_test::BlackBoxTestStandardError<
    typename polca_parallel::RegularisedError>(
    std::span<const double> features, std::span<const int> responses,
    std::span<const double> probs, const arma::Mat<double>& posterior,
    const arma::Mat<double>& prior, std::size_t n_data, std::size_t n_feature,
    polca_parallel::NOutcomes n_outcomes, std::size_t n_cluster,
    bool is_full_constructor);

template void polca_parallel_test::BlackBoxTestStandardError<
    typename polca_parallel::RegularisedErrorRegress>(
    std::span<const double> features, std::span<const int> responses,
    std::span<const double> probs, const arma::Mat<double>& posterior,
    const arma::Mat<double>& prior, std::size_t n_data, std::size_t n_feature,
    polca_parallel::NOutcomes n_outcomes, std::size_t n_cluster,
    bool is_full_constructor);

template <typename StandardErrorType>
void polca_parallel_test::BlackBoxTestStandardError(
    std::span<const double> features, std::span<const int> responses,
    std::span<const double> probs, const arma::Mat<double>& posterior,
    const arma::Mat<double>& prior, std::size_t n_data, std::size_t n_feature,
    polca_parallel::NOutcomes n_outcomes, std::size_t n_cluster,
    bool is_full_constructor) {
  std::size_t len_regress_coeff = n_feature * (n_cluster - 1);
  std::vector<double> prior_error(n_cluster);
  std::vector<double> probs_error(n_outcomes.sum() * n_cluster);
  std::vector<double> regress_coeff_error(len_regress_coeff *
                                          len_regress_coeff);

  std::unique_ptr<polca_parallel::StandardError> standard_error;
  if constexpr (std::is_base_of<polca_parallel::StandardErrorRegress,
                                StandardErrorType>::value) {
    standard_error = std::make_unique<StandardErrorType>(
        features, responses, probs,
        std::span<const double>(prior.begin(), prior.size()),
        std::span<const double>(posterior.begin(), posterior.size()), n_data,
        n_feature, n_outcomes, n_cluster,
        std::span<double>(prior_error.begin(), prior_error.size()),
        std::span<double>(probs_error.begin(), probs_error.size()),
        std::span<double>(regress_coeff_error.begin(),
                          regress_coeff_error.size()));
  } else {
    if (is_full_constructor) {
      standard_error = std::make_unique<StandardErrorType>(
          features, responses, probs,
          std::span<const double>(prior.begin(), prior.size()),
          std::span<const double>(posterior.begin(), posterior.size()), n_data,
          n_feature, n_outcomes, n_cluster,
          std::span<double>(prior_error.begin(), prior_error.size()),
          std::span<double>(probs_error.begin(), probs_error.size()),
          std::span<double>());
    } else {
      standard_error = std::make_unique<StandardErrorType>(
          responses, probs,
          std::span<const double>(prior.begin(), prior.size()),
          std::span<const double>(posterior.begin(), posterior.size()), n_data,
          n_outcomes, n_cluster,
          std::span<double>(prior_error.begin(), prior_error.size()),
          std::span<double>(probs_error.begin(), probs_error.size()));
    }
  }

  standard_error->Calc();

  for (auto i : prior_error) {
    CHECK(i >= 0.0);
  }
  for (auto i : probs_error) {
    CHECK(i >= 0.0);
  }
  if constexpr (std::is_base_of<polca_parallel::StandardErrorRegress,
                                StandardErrorType>::value) {
    for (std::size_t i = 0; i < len_regress_coeff; ++i) {
      CHECK(regress_coeff_error.at(i * (len_regress_coeff + 1)));
    }
  }
}
