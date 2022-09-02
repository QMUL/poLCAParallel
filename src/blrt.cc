#include "blrt.h"

polca_parallel::Blrt::Blrt(double* prior_null, double* prob_null,
                           int n_cluster_null, double* prior_alt,
                           double* prob_alt, int n_cluster_alt, int n_data,
                           int n_category, int* n_outcomes, int sum_outcomes,
                           int n_bootstrap, int n_rep, int n_thread,
                           int max_iter, double tolerance,
                           double* ratio_array) {
  this->prior_null_ = prior_null;
  this->prob_null_ = prob_null;
  this->n_cluster_null_ = n_cluster_null;
  this->prior_alt_ = prior_alt;
  this->prob_alt_ = prob_alt;
  this->n_cluster_alt_ = n_cluster_alt;
  this->n_data_ = n_data;
  this->n_category_ = n_category;
  this->n_outcomes_ = n_outcomes;
  this->sum_outcomes_ = sum_outcomes;
  this->n_bootstrap_ = n_bootstrap;
  this->n_rep_ = n_rep;
  this->n_thread_ = n_thread;
  this->max_iter_ = max_iter;
  this->tolerance_ = tolerance;
  this->ratio_array_ = ratio_array;
  this->n_bootstrap_done_lock_ = new std::mutex();
}

polca_parallel::Blrt::~Blrt() { delete this->n_bootstrap_done_lock_; }

void polca_parallel::Blrt::SetSeed(std::seed_seq* seed) {
  this->seed_array_ =
      std::unique_ptr<unsigned[]>(new unsigned[this->n_bootstrap_]);
  unsigned* seed_array = this->seed_array_.get();
  seed->generate(seed_array, seed_array + this->n_bootstrap_);
}

void polca_parallel::Blrt::Run() {
  std::thread thread_array[this->n_thread_ - 1];
  for (int i = 0; i < this->n_thread_ - 1; ++i) {
    thread_array[i] = std::thread(&Blrt::RunThread, this);
  }
  // main thread run
  this->RunThread();
  // join threads
  for (int i = 0; i < this->n_thread_ - 1; ++i) {
    thread_array[i].join();
  }
}

void polca_parallel::Blrt::RunThread() {
  bool is_working = true;
  int i_bootstrap;

  // to store the bootstrap samples
  int* bootstrap_data = new int[this->n_data_ * this->n_category_];

  // allocate memory for all required arrays, a lot of them aren't used after
  // fitting
  double* features = new double[this->n_data_];
  double* fitted_posterior_null =
      new double[this->n_data_ * this->n_cluster_null_];
  double* fitted_posterior_alt =
      new double[this->n_data_ * this->n_cluster_alt_];
  double* fitted_prior_null = new double[this->n_data_ * this->n_cluster_null_];
  double* fitted_prior_alt = new double[this->n_data_ * this->n_cluster_alt_];
  double* fitted_prob_null =
      new double[this->n_cluster_null_ * this->sum_outcomes_];
  double* fitted_prob_alt =
      new double[this->n_cluster_alt_ * this->sum_outcomes_];
  double* fitted_regress_coeff_null = new double[this->n_cluster_null_ - 1];
  double* fitted_regress_coeff_alt = new double[this->n_cluster_alt_ - 1];

  while (is_working) {
    // lock to retrive n_bootstrap_done_
    // shall be unlocked in both if and else branches
    this->n_bootstrap_done_lock_->lock();
    if (this->n_bootstrap_done_ < this->n_bootstrap_) {
      i_bootstrap = this->n_bootstrap_done_;
      // increment for the next worker to work on
      ++this->n_bootstrap_done_;
      this->n_bootstrap_done_lock_->unlock();

      // instantiate a rng
      std::unique_ptr<std::mt19937_64> rng(
          new std::mt19937_64(this->seed_array_[i_bootstrap]));

      // bootstrap data using null model
      this->Bootstrap(this->prior_null_, this->prob_null_,
                      this->n_cluster_null_, rng.get(), bootstrap_data);

      // null model fit
      double ln_l_array;
      polca_parallel::EmAlgorithmArraySerial null_model(
          features, bootstrap_data, this->prob_null_, this->n_data_, 1,
          this->n_category_, this->n_outcomes_, this->sum_outcomes_,
          this->n_cluster_null_, 1, 1, this->max_iter_, this->tolerance_,
          fitted_posterior_null, fitted_prior_null, fitted_prob_null,
          fitted_regress_coeff_null, &ln_l_array);
      null_model.SetRng(&rng);
      null_model.Fit();
      rng = null_model.MoveRng();

      // alt model fit
      polca_parallel::EmAlgorithmArraySerial alt_model(
          features, bootstrap_data, this->prob_alt_, this->n_data_, 1,
          this->n_category_, this->n_outcomes_, this->sum_outcomes_,
          this->n_cluster_alt_, 1, 1, this->max_iter_, this->tolerance_,
          fitted_posterior_alt, fitted_prior_alt, fitted_prob_alt,
          fitted_regress_coeff_alt, &ln_l_array);
      alt_model.SetRng(&rng);
      alt_model.Fit();
      rng = alt_model.MoveRng();

      // work out the log ratio, save it
      this->ratio_array_[i_bootstrap] =
          2 * (alt_model.get_optimal_ln_l() - null_model.get_optimal_ln_l());

    } else {
      // all bootstrap samples done, stop working
      this->n_bootstrap_done_lock_->unlock();
      is_working = false;
    }
  }

  delete[] bootstrap_data;
  delete[] features;
  delete[] fitted_posterior_null;
  delete[] fitted_posterior_alt;
  delete[] fitted_prior_null;
  delete[] fitted_prior_alt;
  delete[] fitted_prob_null;
  delete[] fitted_prob_alt;
  delete[] fitted_regress_coeff_null;
  delete[] fitted_regress_coeff_alt;
}

/**
 * Generate a bootstrap sample
 *
 * @param prior Vector of prior probabilities for the null
 * model, probability data point is in cluster m NOT given responses
 * <ul>
 *   <li>dim 0: for each cluster</li>
 * </ul>
 * @param prob Vector of estimated response probabilities for
 * each category, flatten list of matrices. Used as an initial value when
 * fitting onto the bootstrap sample.
 * <ul>
 *   <li>dim 0: for each outcome</li>
 *   <li>dim 1: for each cluster</li>
 *   <li>dim 2: for each category</li>
 * </ul>
 * @param n_cluster Number of clusters
 * @param rng Random number generator
 * @param response To store results, design matrix transpose of responses
 * <ul>
 *   <li>dim 0: for each category</li>
 *   <li>dim 1: for each data point</li>
 * </ul>
 */
void polca_parallel::Blrt::Bootstrap(double* prior, double* prob, int n_cluster,
                                     std::mt19937_64* rng, int* response) {
  int i_cluster;
  double* prob_i_cluster;  // pointer relative to prob
  double p_sum;            // used to sum up probabilities for each outcome
  double rand_uniform;
  int i_outcome;

  std::uniform_int_distribution<int> prior_dist(0, n_cluster - 1);
  std::uniform_real_distribution<double> uniform_dist(0, 1);

  for (int i_data = 0; i_data < this->n_data_; ++i_data) {
    i_cluster = prior_dist(*rng);  // select a random cluster
    // point to the corresponding probabilites for this random cluster
    prob_i_cluster = prob + i_cluster * this->sum_outcomes_;

    for (int i_category = 0; i_category < this->n_category_; ++i_category) {
      p_sum = 0.0;
      rand_uniform = uniform_dist(*rng);

      // use rand_uniform to randomly sample an outcome according to the
      // probabilities in prob_i_cluster[0], prob_i_cluster[1], ...
      // i_outcome is the randomly selected outcome
      i_outcome = -1;
      while (rand_uniform > p_sum) {
        ++i_outcome;
        p_sum += prob_i_cluster[i_outcome];
      }
      *response = i_outcome + 1;  // response is one-based index

      // increment for the next category
      prob_i_cluster += this->n_outcomes_[i_category];
      ++response;
    }
  }
}
