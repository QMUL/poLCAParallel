// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// EmAlgorithmRcpp
Rcpp::List EmAlgorithmRcpp(Rcpp::NumericMatrix features, Rcpp::IntegerMatrix responses, Rcpp::NumericVector initial_prob, int n_data, int n_feature, int n_category, Rcpp::IntegerVector n_outcomes, int n_cluster, int n_rep, int n_thread, int max_iter, double tolerance, Rcpp::IntegerVector seed);
RcppExport SEXP _poLCAParallel_EmAlgorithmRcpp(SEXP featuresSEXP, SEXP responsesSEXP, SEXP initial_probSEXP, SEXP n_dataSEXP, SEXP n_featureSEXP, SEXP n_categorySEXP, SEXP n_outcomesSEXP, SEXP n_clusterSEXP, SEXP n_repSEXP, SEXP n_threadSEXP, SEXP max_iterSEXP, SEXP toleranceSEXP, SEXP seedSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type features(featuresSEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerMatrix >::type responses(responsesSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type initial_prob(initial_probSEXP);
    Rcpp::traits::input_parameter< int >::type n_data(n_dataSEXP);
    Rcpp::traits::input_parameter< int >::type n_feature(n_featureSEXP);
    Rcpp::traits::input_parameter< int >::type n_category(n_categorySEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerVector >::type n_outcomes(n_outcomesSEXP);
    Rcpp::traits::input_parameter< int >::type n_cluster(n_clusterSEXP);
    Rcpp::traits::input_parameter< int >::type n_rep(n_repSEXP);
    Rcpp::traits::input_parameter< int >::type n_thread(n_threadSEXP);
    Rcpp::traits::input_parameter< int >::type max_iter(max_iterSEXP);
    Rcpp::traits::input_parameter< double >::type tolerance(toleranceSEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerVector >::type seed(seedSEXP);
    rcpp_result_gen = Rcpp::wrap(EmAlgorithmRcpp(features, responses, initial_prob, n_data, n_feature, n_category, n_outcomes, n_cluster, n_rep, n_thread, max_iter, tolerance, seed));
    return rcpp_result_gen;
END_RCPP
}
// ylik
Rcpp::NumericVector ylik(Rcpp::NumericVector probs, Rcpp::IntegerVector y, int obs, int items, Rcpp::IntegerVector numChoices, int classes);
RcppExport SEXP _poLCAParallel_ylik(SEXP probsSEXP, SEXP ySEXP, SEXP obsSEXP, SEXP itemsSEXP, SEXP numChoicesSEXP, SEXP classesSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type probs(probsSEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerVector >::type y(ySEXP);
    Rcpp::traits::input_parameter< int >::type obs(obsSEXP);
    Rcpp::traits::input_parameter< int >::type items(itemsSEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerVector >::type numChoices(numChoicesSEXP);
    Rcpp::traits::input_parameter< int >::type classes(classesSEXP);
    rcpp_result_gen = Rcpp::wrap(ylik(probs, y, obs, items, numChoices, classes));
    return rcpp_result_gen;
END_RCPP
}
// GoodnessFitRcpp
Rcpp::List GoodnessFitRcpp(Rcpp::IntegerMatrix responses, Rcpp::NumericVector prior, Rcpp::NumericVector outcome_prob, int n_data, int n_category, Rcpp::IntegerVector n_outcomes, int n_cluster);
RcppExport SEXP _poLCAParallel_GoodnessFitRcpp(SEXP responsesSEXP, SEXP priorSEXP, SEXP outcome_probSEXP, SEXP n_dataSEXP, SEXP n_categorySEXP, SEXP n_outcomesSEXP, SEXP n_clusterSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::IntegerMatrix >::type responses(responsesSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type prior(priorSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type outcome_prob(outcome_probSEXP);
    Rcpp::traits::input_parameter< int >::type n_data(n_dataSEXP);
    Rcpp::traits::input_parameter< int >::type n_category(n_categorySEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerVector >::type n_outcomes(n_outcomesSEXP);
    Rcpp::traits::input_parameter< int >::type n_cluster(n_clusterSEXP);
    rcpp_result_gen = Rcpp::wrap(GoodnessFitRcpp(responses, prior, outcome_prob, n_data, n_category, n_outcomes, n_cluster));
    return rcpp_result_gen;
END_RCPP
}

RcppExport void d2lldbeta2(void *, void *, void *, void *, void *, void *, void *, void *);
RcppExport void postclass(void *, void *, void *, void *, void *, void *, void *, void *);
RcppExport void probhat(void *, void *, void *, void *, void *, void *, void *);

static const R_CallMethodDef CallEntries[] = {
    {"_poLCAParallel_EmAlgorithmRcpp", (DL_FUNC) &_poLCAParallel_EmAlgorithmRcpp, 13},
    {"_poLCAParallel_ylik", (DL_FUNC) &_poLCAParallel_ylik, 6},
    {"_poLCAParallel_GoodnessFitRcpp", (DL_FUNC) &_poLCAParallel_GoodnessFitRcpp, 7},
    {"d2lldbeta2", (DL_FUNC) &d2lldbeta2, 8},
    {"postclass",  (DL_FUNC) &postclass,  8},
    {"probhat",    (DL_FUNC) &probhat,    7},
    {NULL, NULL, 0}
};

RcppExport void R_init_poLCAParallel(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
