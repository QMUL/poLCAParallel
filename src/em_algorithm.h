#ifndef EM_ALGORITHM_H
#define EM_ALGORITHM_H

#include <math.h>

// [[Rcpp::depends(RcppArmadillo)]]
#include "RcppArmadillo.h"

using namespace arma;

class EmAlgorithm {

  protected:
    double* features_;  // design matrix of features, matrix n_data x n_feature
    // design matrix transpose of responses, matrix n_category x n_data
    int* responses_;
    // vector of initial probabilities for each category and responses,
    // flatten list of matrices
      // dim 0: for each outcome
      // dim 1: for each category
      // dim 2: for each model
    double* initial_prob_;
    int n_data_;  // number of data points
    int n_feature_;  // number of features
    int n_category_;  // number of categories
    int* n_outcomes_;  // vector of number of outcomes for each category
    int sum_outcomes_;  // sum of n_outcomes
    int n_cluster_;  // number of clusters (classes in lit) to fit
    int max_iter_;  // maximum number of iterations for EM algorithm
    // tolerance for difference in log likelihood, used for stopping condition
    double tolerance_;
    // design matrix of posterior probabilities (also called responsibility)
      // probability data point is in cluster m given responses
      // matrix, dim 0: for each data, dim 1: for each cluster
    double* posterior_;
    // design matrix of prior probabilities, probability data point is in
      // cluster m NOT given responses
      // after calculations, it shall be in matrix form,
      // dim 0: for each data, dim 1: for each cluster
      // during the start and calculations, it may take on a different form,
      // use the method GetPrior() to get the prior for a data point and cluster
    double* prior_;
    // vector of estimated response probabilities for each category,
    // flatten list of matrices
      // dim 0: for each outcome
      // dim 1: for each cluster
      // dim 2: for each category
    double* estimated_prob_;
    // vector length n_features_*(n_cluster-1), linear regression coefficient
      // in matrix form, to be multiplied to the features and linked to the
      // prior using softmax
    double* regress_coeff_;

    double ln_l_;  // log likelihood, updated at each iteration of EM
    // vector, for each data point, log likelihood for each data point
      // the log likelihood is the sum
    double* ln_l_array_;
    int n_iter_;  // number of iterations done right now

  public:

    // for arguments for this constructor, see description of the member
      // variables
    // the following content pointed too shall modified:
      // posterior, prior, estimated_prob, regress_coeff
    EmAlgorithm(
        double* features,
        int* responses,
        double* initial_prob,
        int n_data,
        int n_feature,
        int n_category,
        int* n_outcomes,
        int sum_outcomes,
        int n_cluster,
        int max_iter,
        double tolerance,
        double* posterior,
        double* prior,
        double* estimated_prob,
        double* regress_coeff) {

      this->features_ = features;
      this->responses_ = responses;
      this->initial_prob_ = initial_prob;
      this->n_data_ = n_data;
      this->n_feature_ = n_feature;
      this->n_category_ = n_category;
      this->n_outcomes_ = n_outcomes;
      this->sum_outcomes_ = sum_outcomes;
      this->n_cluster_ = n_cluster;
      this->max_iter_ = max_iter;
      this->tolerance_ = tolerance;
      this->posterior_ = posterior;
      this->prior_ = prior;
      this->estimated_prob_ = estimated_prob;
      this->regress_coeff_ = regress_coeff;
      this->ln_l_array_ = new double[this->n_data_];
    }

    virtual ~EmAlgorithm() {
      delete this->ln_l_array_;
    }

    // fit data to model using EM algorithm
    // data is provided through the constructor
    // important results are stored in the member variables:
      // posterior_
      // prior_
      // estimated_prob_
      // ln_l_array_
      // ln_l_
      // n_iter_
    void Fit() {
      //copy initial prob to estimated prob
      std::memcpy(this->estimated_prob_, this->initial_prob_,
          this->n_cluster_*this->sum_outcomes_*sizeof(double));

      double ln_l_difference;
      double ln_l_before = -INFINITY;
      bool isError = false;

      // initalise prior probabilities, for each cluster
      this->InitPrior();

      // do EM algorithm
      for (this->n_iter_=0; this->n_iter_<this->max_iter_; this->n_iter_++) {
        //E step updates prior probabilities
        this->EStep();

        // E step updates ln_l_array_, use that to calculate log likelihood
        // use log likelihood to determine stopping condition
        Col<double> ln_l_array(this->ln_l_array_, this->n_data_, false);
        this->ln_l_ = sum(ln_l_array);
        ln_l_difference = this->ln_l_ - ln_l_before;
        if (ln_l_difference < this->tolerance_) {
          break;
        }
        ln_l_before = this->ln_l_;

        // M step updates posterior probabilities and estimated probabilities
        this->MStep();
      }

      // reformat prior
      this->FinalPrior();
    }

    // getter function for log likelihood
    double get_ln_l() {
      return this->ln_l_;
    }

    // getter function for number of iterations
    int get_n_iter() {
      return this->n_iter_;
    }

  protected:

    // Initalise prior probabilities
    // Modify the content of prior_ which contains prior probabilities for each
      // cluster
    virtual void InitPrior() {
      // prior probabilities are the same for all data points in this
        // implementation
      for (int i=0; i<this->n_cluster_; i++) {
        this->prior_[i] = 1.0 / ((double) this->n_cluster_);
      }
    }

    // Adjust prior return value to matrix format
    virtual void FinalPrior() {
      // Copying prior probabilities as each data point as the same prior
      double prior_copy[this->n_cluster_];
      std::memcpy(&prior_copy, this->prior_, this->n_cluster_*sizeof(double));
      for (int m=0; m<this->n_cluster_; m++){
        for (int i=0; i<this->n_data_; i++) {
          this->prior_[m*this->n_data_ + i] = prior_copy[m];
        }
      }
    }

    // Get prior during the EM algorithm
    virtual double GetPrior(int data_index, int cluster_index) {
      return this->prior_[cluster_index];
    }

    // E step
    // update the posterior probabilities given the prior probabilities and
      // estimated response probabilities
    // modifies the member variables posterior_ and ln_l_array_
    // calculations for the E step also provides the elements for ln_l_array
    //
    // IMPORTANT DEV NOTES: p is iteratively being multiplied, underflow errors
      // may occur if the number of data points is large, say more than 300
      // consider summing over log probabilities  rather than multiplying
      // probabilities
    void EStep() {
      double* estimated_prob;  // for pointing to elements in estimated_prob_
      int n_outcome;  // number of outcomes while iterating through categories
      // used for conditioned on cluster m likelihood calculation
        // for a data point
        // P(Y^{(i)} | cluster m)
      double p;
      // used for likelihood calculation for a data point
        // P(Y^{(i)})
      double normaliser;
      // used for calculating posterior probability, conditioned on a cluster m,
        // for a data point
        // P(cluster m | Y^{(i)})
      double posterior_iter;
      int y;  // for getting a response from responses_

      for (int i=0; i<this->n_data_; i++) {

        // normaliser noramlise over cluster, so loop over cluster here
        normaliser = 0;

        estimated_prob = this->estimated_prob_;
        for (int m=0; m<this->n_cluster_; m++) {
          // calculate conditioned on cluster m likelihood
          p = 1;
          for (int j=0; j<this->n_category_; j++) {
            n_outcome = this->n_outcomes_[j];
            y = this->responses_[i*this->n_category_ + j];
            p *= estimated_prob[y - 1];
            // increment to point to the next category
            estimated_prob += n_outcome;
          }
          // posterior = likelihood x prior
          posterior_iter = p * this->GetPrior(i, m);
          this->posterior_[m*this->n_data_ + i] = posterior_iter;
          normaliser += posterior_iter;
        }
        // normalise
        for (int m=0; m<this->n_cluster_; m++) {
          this->posterior_[m*this->n_data_ + i] /= normaliser;
        }
        // store the log likelihood for this data point
        this->ln_l_array_[i] = log(normaliser);
      }
    }

    // M Step
    // update the prior probabilities and estimated response probabilities
      // given the posterior probabilities
    // modifies the member variables prior_ and estimated_prob_
    virtual void MStep() {

      // estimate prior
      // for this implementation, the mean posterior, taking the mean over data
        // points
      Mat<double> posterior_arma(this->posterior_, this->n_data_,
                                 this->n_cluster_, false);
      Row<double> prior = mean(posterior_arma, 0);
      std::memcpy(this->prior_, prior.begin(), this->n_cluster_*sizeof(double));

      int y;  // for getting a response from responses_
      double* estimated_prob;  // for pointing to elements in estimated_prob_
      double* estimated_prob_m; // points to estimated_prob_ for given cluster
      int n_outcome;  // number of outcomes while iterating through categories
      double posterior_iter;

      // set all estimated response probability to zero
      for (int i=0; i<this->n_cluster_*this->sum_outcomes_; i++) {
        this->estimated_prob_[i] = 0;
      }

      // for each cluster
      estimated_prob = this->estimated_prob_;
      estimated_prob_m = estimated_prob;
      for (int m=0; m<this->n_cluster_; m++) {
        // estimate outcome probabilities
        for (int i=0; i<this->n_data_; i++) {
          estimated_prob = estimated_prob_m;
          for (int j=0; j<this->n_category_; j++) {
            n_outcome = this->n_outcomes_[j];
            y = this->responses_[i*this->n_category_ + j];
            posterior_iter = this->posterior_[m*this->n_data_ + i];
            estimated_prob[y - 1] += posterior_iter;
            estimated_prob += n_outcome;
          }
        }
        // normalise by the sum of posteriors
          // calculations can be reused as the prior is the mean of posteriors
          // from the E step
        for (int j=0; j<this->n_category_; j++) {
          n_outcome = this->n_outcomes_[j];
          for (int k=0; k<n_outcome; k++) {
            estimated_prob_m[k] /=
                ((double) this->n_data_) * this->prior_[m];
          }
          estimated_prob_m += n_outcome;
        }
      }
    }
};

#endif
