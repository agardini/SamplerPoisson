#define RCPPDIST_DONT_USE_ARMA

#include <RcppEigen.h>
#include <RcppGSL.h>
#include <Rcpp.h>
#include <complex>
#include <RcppDist.h>
#include <random>
#include "MCMCtools.h"

using namespace Rcpp;
using namespace Eigen;

 Eigen::VectorXd compute_n_mix(double x,
                               Eigen::VectorXd weights,
                               Eigen::VectorXd means,
                               Eigen::VectorXd vars){
   int n_comp = weights.size();
   Eigen::VectorXd probs(n_comp);
   for(int j = 0; j < n_comp; j++){
     probs[j] = weights[j] * R::dnorm(x, means[j], std::sqrt(vars[j]), 0);
   }
   return(probs);
 }

double compute_log_gumbel(double x, double nu){
  double log_p=0;
  log_p = -nu * x - std::exp(-x) - std::lgamma(nu);
  return(log_p);
}

 // // [[Rcpp::export]]
 // double compute_log_mix(double x,
 //                        Eigen::VectorXd weights,
 //                        Eigen::VectorXd means,
 //                        Eigen::VectorXd vars){
 //   int n_comp = weights.size();
 //   double log_p=0;
 //   for(int j = 0; j < n_comp; j++){
 //     log_p = log_p + weights[j] * R::dnorm(x, means[j], std::sqrt(vars[j]), 0);
 //   }
 //   log_p = std::log(log_p);
 //   return(log_p);
 // }
 //
 // // [[Rcpp::export]]
 double compute_log_mix(double x,
                        Eigen::VectorXd weights,
                        Eigen::VectorXd means,
                        Eigen::VectorXd vars){
   int n_comp = weights.size();
   Eigen::VectorXd term(n_comp);
   double konst=0.0;
   double log_p=0;
   konst = std::log(weights[0]) + R::dnorm(x, means[0], std::sqrt(vars[0]), 1);
   for(int j = 0; j < n_comp; j++){
     term[j] = std::log(weights[j]) + R::dnorm(x, means[j], std::sqrt(vars[j]), 1);
     if(term[j] > konst){
       konst = term[j];
     }
   }
   //Rprintf("%f\n", konst);
   for(int j = 0; j < n_comp; j++){
     log_p += std::exp(term[j] - konst);
   }
   log_p = konst + std::log(log_p);
   return(log_p);
 }



  int rnd_discr(Eigen::VectorXd p_xd){

   int n_comp = p_xd.size();
   std::vector<double> p(n_comp);
   for(int j = 0; j < n_comp; j++){
     p[j] = p_xd[j];
   }
   std::random_device generator;
   std::discrete_distribution<int> distribution(p.begin(), p.end());
   int number = distribution(generator);
   return number; //number o number + 1??? direi number

 }

 void sample_aux_Poisson_nolik(Eigen::VectorXd log_lambda,
                               Eigen::VectorXi y,
                               int n0,
                               std::vector<Eigen::VectorXd> w_mix,
                               std::vector<Eigen::VectorXd> m_mix,
                               std::vector<Eigen::VectorXd> v_mix,
                               Eigen::VectorXd& z_aux,
                               Eigen::VectorXd& var_aux,
                               Eigen::VectorXd& mean_aux){
   int n = y.size();
   Eigen::VectorXd tau1(n), tau2(n - n0),  var1(n), var2(n - n0),
   mean1(n), mean2(n - n0);

   int r1, r2;
   Eigen::VectorXd probs;
   double xi_i;
   // (a) sample xi_i
   for(int j = 0; j < n0; j++){
     xi_i = R::rexp(1.0 / std::exp(log_lambda[j]));
     tau1[j] = 1.0 + xi_i;

     probs = compute_n_mix(- std::log(tau1[j]) - log_lambda[j],
                           w_mix[j],
                                m_mix[j],
                                     v_mix[j]);
     r1 = rnd_discr(probs);
     var1[j] = v_mix[j][r1];
     mean1[j] = m_mix[j][r1];
   }
   for(int j = n0; j < n; j++){
     xi_i = R::rexp(1.0 / std::exp(log_lambda[j]));
     tau2[j - n0] = R::rbeta(y[j] * 1.0, 1.0);
     tau1[j] = 1.0 - tau2[j - n0] + xi_i;

     probs = compute_n_mix(-std::log(tau1[j])-log_lambda[j],
                           w_mix[j],
                                m_mix[j],
                                     v_mix[j]);
     r1 = rnd_discr(probs);
     var1[j] = v_mix[j][r1];
     mean1[j] = m_mix[j][r1];

     probs = compute_n_mix(-std::log(tau2[j-n0])-log_lambda[j],
                           w_mix[j + n - n0],
                                m_mix[j + n - n0],
                                     v_mix[j + n - n0]);
     r2 = rnd_discr(probs);
     var2[j - n0] = v_mix[j + n - n0][r2];
     mean2[j - n0] = m_mix[j + n - n0][r2];
   }

   z_aux(2 * n - n0);

   z_aux << tau1, tau2;
   for(int j = 0; j < (2 * n - n0); j++){
     z_aux[j] = - std::log(z_aux[j]);
   }
   var_aux(2 * n - n0);
   var_aux << var1, var2;

   mean_aux(2 * n - n0);
   mean_aux << mean1, mean2;
 }


 void compute_lliks_app_aux(Eigen::VectorXd log_lambda,
                            Eigen::VectorXi y,
                            int n0,
                            std::vector<Eigen::VectorXd> w_mix,
                            std::vector<Eigen::VectorXd> m_mix,
                            std::vector<Eigen::VectorXd> v_mix,
                            Eigen::VectorXd z_aux,
                            double& l_lik_app,
                            double& l_lik_aux
 ){
   int n = y.size();
   l_lik_app = 0.0;
   l_lik_aux = 0.0;

   // (a) sample xi_i
   for(int j = 0; j < n0; j++){
     l_lik_app += compute_log_mix(
       z_aux[j] - log_lambda[j], w_mix[j], m_mix[j], v_mix[j]
     );
     l_lik_aux += compute_log_gumbel(z_aux[j] - log_lambda[j], 1.0);
   }
   for(int j = n0; j < n; j++){
     l_lik_app += compute_log_mix(
       z_aux[j] - log_lambda[j], w_mix[j], m_mix[j], v_mix[j]
     );
     l_lik_aux += compute_log_gumbel(z_aux[j] - log_lambda[j], 1.0);

     l_lik_app += compute_log_mix(
       z_aux[n + j- n0] - log_lambda[j], w_mix[n + j - n0], m_mix[n + j - n0], v_mix[n + j - n0]
     );
     l_lik_aux += compute_log_gumbel(z_aux[n + j - n0] - log_lambda[j], 1.0*y[j]);
   }
 }

  void gibbs_beta_poisson(Eigen::VectorXd& beta,
                         Eigen::VectorXd& Xbeta,
                         Eigen::MatrixXd X,
                         Eigen::VectorXd var_aux,
                         Eigen::MatrixXd Q_beta,
                         Eigen::VectorXd psres){

   Eigen::DiagonalMatrix<double, Dynamic> W = var_aux.cwiseInverse().asDiagonal();
   Eigen::LLT<MatrixXd> chol_Q_fc(X.transpose() * W * X + Q_beta);  //L
   Eigen::VectorXd mu = chol_Q_fc.solve((1.0) * X.transpose() * W * psres); //B
   Eigen::VectorXd z = Rcpp::as<VectorXd>(Rcpp::rnorm(X.cols()));

   beta = chol_Q_fc.matrixU().solve(z) + mu; //??
   Xbeta = X * beta;
 }


 // [[Rcpp::export]]
 Rcpp::List sample_poisson(Eigen::VectorXi y,
                           Eigen::MatrixXd X,////mistura, ad ora
                           int n0,
                           std::vector<Eigen::VectorXd> m_mix_orig,
                           std::vector<Eigen::VectorXd> v_mix_orig,
                           std::vector<Eigen::VectorXd> w_mix_orig,
                           Eigen::VectorXd log_offset,
                           Eigen::MatrixXd S_beta,
                           Eigen::VectorXd beta_init,
                           int niter, int pr, int thin){

   ///////////////////////////////////////////////////////////////////////
   // Initialise mcmc objects - s2e and s2b
   ///////////////////////////////////////////////////////////////////////

   ////////////// s2b  //////////////
   int p = X.cols(), n = y.size(), n_aux = 2 * n - n0;

   //////////////  auxiliary variables  //////////////
   Eigen::VectorXd
   z_aux(n_aux), mean_aux(n_aux), var_aux(n_aux), res(n_aux);

   Eigen::VectorXd
   is_adjusted = Eigen::VectorXd::Zero(n_aux);

   ///////////////////////////////////////////////////////////////////////
   // Initialise mcmc objects - beta
   ///////////////////////////////////////////////////////////////////////
   Eigen::VectorXd
   beta = beta_init, Xbeta = X * beta;

   Eigen::MatrixXd
   Q_beta = S_beta.inverse();

   std::vector<Eigen::VectorXd> w_mix = w_mix_orig, m_mix = m_mix_orig, v_mix = v_mix_orig;

   ///////////////////////////////////////////////////////////////////////
   // Initialise objects for storage
   ///////////////////////////////////////////////////////////////////////
   int storeind = 0, nstore = niter / thin;

   Eigen::MatrixXd
   beta_mc(nstore, p);

   ///////////////////////////////////////////////////////////////////////
   // MCMC sampler
   ///////////////////////////////////////////////////////////////////////
   time_t now;
   for(int k=0;k<niter;k++){

     ///////////////////////////////////////////////////////////////////////
     // Sampling z_aux
     ///////////////////////////////////////////////////////////////////////
     sample_aux_Poisson_nolik(Xbeta + log_offset, y, n0,
                              w_mix, m_mix, v_mix,
                              z_aux, var_aux, mean_aux);

     ///////////////////////////////////////////////////////////////////////
     // Sampling beta and s2b
     ///////////////////////////////////////////////////////////////////////
     gibbs_beta_poisson(beta, Xbeta, X,
                        var_aux, Q_beta, z_aux - mean_aux - log_offset);


     ///////////////////////////////////////////////////////////////////////
     // Storage
     ///////////////////////////////////////////////////////////////////////
     if ((k+1)%thin == 0){
       beta_mc.row(storeind) = beta;
       storeind += 1;
     }
     if ((k+1)%pr==0){
       time(&now);
       Rprintf("Iteration: %d: %s", k+1, ctime(&now));
     }
   }

   return Rcpp::List::create(
     Rcpp::Named("beta") = beta_mc
   );
 }


 // [[Rcpp::export]]
 Rcpp::List sample_poisson_MH(Eigen::VectorXi y,
                              Eigen::MatrixXd X,
                              int n0,
                              std::vector<Eigen::VectorXd> m_mix_orig,
                              std::vector<Eigen::VectorXd> v_mix_orig,
                              std::vector<Eigen::VectorXd> w_mix_orig,
                              std::vector<Eigen::VectorXd> m_mix_adj,
                              std::vector<Eigen::VectorXd> v_mix_adj,
                              std::vector<Eigen::VectorXd> w_mix_adj,
                              Eigen::VectorXd check_mix,
                              int b_gibbs_start,
                              int b_check,
                              int adjust,
                              Eigen::VectorXd log_offset,
                              Eigen::MatrixXd S_beta,
                              Eigen::VectorXd beta_init,
                              int niter, int pr, int thin){

   ///////////////////////////////////////////////////////////////////////
   // Initialise mcmc objects - s2e and s2b
   ///////////////////////////////////////////////////////////////////////

   ////////////// s2b  //////////////
   int p = X.cols(), n = y.size(), n_aux = 2 * n - n0;

   //////////////  auxiliary variables  //////////////
   Eigen::VectorXd
   z_aux(n_aux), mean_aux(n_aux), var_aux(n_aux), res(n_aux);

   Eigen::VectorXd
   is_adjusted = Eigen::VectorXd::Zero(n_aux);

   ///////////////////////////////////////////////////////////////////////
   // Initialise mcmc objects - beta
   ///////////////////////////////////////////////////////////////////////
   Eigen::VectorXd
   beta = beta_init, Xbeta = X * beta, beta_curr = beta_init, Xbeta_curr = X * beta_curr;

   double log_lik_appr_prop, log_lik_aux_prop, log_lik_appr_curr, log_lik_aux_curr;


   Eigen::MatrixXd
   Q_beta = S_beta.inverse();

   Eigen::MatrixXd
   sampled_mean_aux(niter, n_aux);

   std::vector<Eigen::VectorXd> w_mix = w_mix_orig, m_mix = m_mix_orig, v_mix = v_mix_orig;

   compute_lliks_app_aux(Xbeta_curr + log_offset, y, n0, w_mix, m_mix, v_mix, z_aux,
                         log_lik_appr_curr, log_lik_aux_curr
   );


   ///////////////////////////////////////////////////////////////////////
   // Initialise objects for storage
   ///////////////////////////////////////////////////////////////////////
   int storeind = 0, nstore = niter / thin;

   Eigen::MatrixXd
   beta_mc(nstore, p);

   int acc_beta;
   Eigen::VectorXi accept_beta(nstore);


   ///////////////////////////////////////////////////////////////////////
   // MCMC sampler
   ///////////////////////////////////////////////////////////////////////
   time_t now;
   for(int k=0;k<niter;k++){

     ///////////////////////////////////////////////////////////////////////
     // Sampling z_aux
     ///////////////////////////////////////////////////////////////////////
     sample_aux_Poisson_nolik(Xbeta_curr + log_offset, y, n0,
                              w_mix, m_mix, v_mix,
                              z_aux, var_aux, mean_aux);

     ///////////////////////////////////////////////////////////////////////
     // Sampling beta and s2b
     ///////////////////////////////////////////////////////////////////////
     gibbs_beta_poisson(beta, Xbeta, X,
                        var_aux, Q_beta, z_aux - mean_aux - log_offset);

     compute_lliks_app_aux(Xbeta + log_offset, y, n0, w_mix, m_mix, v_mix, z_aux,
                           log_lik_appr_prop, log_lik_aux_prop);

     compute_lliks_app_aux(Xbeta_curr + log_offset, y, n0, w_mix, m_mix, v_mix, z_aux,
                           log_lik_appr_curr, log_lik_aux_curr);

     acc_beta = 0;

     if(k < b_gibbs_start){
       acc_beta = 1;
     } else{
       acc_beta = metropolis_acceptance(log_lik_aux_prop - log_lik_appr_prop,
                                        log_lik_aux_curr - log_lik_appr_curr);
     }

     if(acc_beta == 1L){
       beta_curr = beta;
       Xbeta_curr = Xbeta;
     }

     if((k < (b_gibbs_start + b_check)) & (k >= b_gibbs_start) & (adjust == 1)){
       res = z_aux - log_offset - Xbeta_curr;
       for(int j=0; j<n_aux; j++){
         if(res[j]>check_mix[j]){
           w_mix[j] = w_mix_adj[j];
           m_mix[j] = m_mix_adj[j];
           v_mix[j] = v_mix_adj[j];
           is_adjusted[j] += 1;
         }
       }
     }


     ///////////////////////////////////////////////////////////////////////
     // Storage
     ///////////////////////////////////////////////////////////////////////
     if ((k+1)%thin == 0){
       accept_beta[storeind] = acc_beta;
       beta_mc.row(storeind) = beta_curr;
       storeind += 1;
     }
     if ((k+1)%pr==0){
       time(&now);
       Rprintf("Iteration: %d: %s", k+1, ctime(&now));
     }
   }

   return Rcpp::List::create(
     Rcpp::Named("beta") = beta_mc,
     Rcpp::Named("accept_MH") = accept_beta,
     Rcpp::Named("is_adjusted") = is_adjusted
   );
 }


 // [[Rcpp::export]]
 Rcpp::List sample_poisson_auto(Eigen::VectorXi y,
                                Eigen::MatrixXd X,////mistura, ad ora
                                int n0,
                                std::vector<Eigen::VectorXd> m_mix_orig,
                                std::vector<Eigen::VectorXd> v_mix_orig,
                                std::vector<Eigen::VectorXd> w_mix_orig,
                                std::vector<Eigen::VectorXd> m_mix_adj,
                                std::vector<Eigen::VectorXd> v_mix_adj,
                                std::vector<Eigen::VectorXd> w_mix_adj,
                                Eigen::VectorXd check_mix_min,
                                Eigen::VectorXd check_mix_max,
                                int b_gibbs_start,
                                int b_check,
                                double threshold_MH,
                                double threshold_adj,
                                Eigen::VectorXd log_offset,
                                Eigen::MatrixXd S_beta,
                                Eigen::VectorXd beta_init,
                                int niter, int pr, int thin){

   ///////////////////////////////////////////////////////////////////////
   // Initialise mcmc objects - s2e and s2b
   ///////////////////////////////////////////////////////////////////////


   ////////////// s2b  //////////////
   int p = X.cols(), n = y.size(), n_aux = 2 * n - n0;
   int MH = 0;
   Eigen::VectorXd
   is_adjusted = Eigen::VectorXd::Zero(n_aux), is_MH = Eigen::VectorXd::Zero(n_aux);


   //////////////  auxiliary variables  //////////////
   Eigen::VectorXd
   z_aux(n_aux), mean_aux(n_aux), var_aux(n_aux), res(n_aux);

   ///////////////////////////////////////////////////////////////////////
   // Initialise mcmc objects - beta
   ///////////////////////////////////////////////////////////////////////
   Eigen::VectorXd
   beta = beta_init, Xbeta = X * beta, beta_curr = beta_init, Xbeta_curr = X * beta_curr;

   double log_lik_appr_prop, log_lik_aux_prop, log_lik_appr_curr, log_lik_aux_curr;


   Eigen::MatrixXd
   Q_beta = S_beta.inverse();

   Eigen::MatrixXd
   sampled_mean_aux(niter, n_aux);

   std::vector<Eigen::VectorXd> w_mix = w_mix_orig, m_mix = m_mix_orig, v_mix = v_mix_orig;

   compute_lliks_app_aux(Xbeta_curr + log_offset, y, n0, w_mix, m_mix, v_mix, z_aux,
                         log_lik_appr_curr, log_lik_aux_curr
   );

   ///////////////////////////////////////////////////////////////////////
   // Initialise objects for storage
   ///////////////////////////////////////////////////////////////////////
   int storeind = 0, nstore = niter / thin;

   Eigen::MatrixXd
   beta_mc(nstore, p),
   y_star_aux(nstore, n_aux),
   res_mc(nstore, n_aux);

   int acc_beta;
   Eigen::VectorXi accept_beta(nstore);

   ///////////////////////////////////////////////////////////////////////
   // MCMC sampler
   ///////////////////////////////////////////////////////////////////////
   time_t now;
   for(int k=0;k<niter;k++){

     ///////////////////////////////////////////////////////////////////////
     // Sampling z_aux
     ///////////////////////////////////////////////////////////////////////
     sample_aux_Poisson_nolik(Xbeta_curr + log_offset, y, n0,
                              w_mix, m_mix, v_mix,
                              z_aux, var_aux, mean_aux);

     ///////////////////////////////////////////////////////////////////////
     // Sampling beta and s2b
     ///////////////////////////////////////////////////////////////////////
     gibbs_beta_poisson(beta, Xbeta, X,
                        var_aux, Q_beta, z_aux - mean_aux - log_offset);

     acc_beta = 1L;
     if(MH > 0){
       compute_lliks_app_aux(Xbeta + log_offset, y, n0, w_mix, m_mix, v_mix, z_aux,
                             log_lik_appr_prop, log_lik_aux_prop);

       compute_lliks_app_aux(Xbeta_curr + log_offset, y, n0, w_mix, m_mix, v_mix, z_aux,
                             log_lik_appr_curr, log_lik_aux_curr);

       acc_beta = metropolis_acceptance(log_lik_aux_prop - log_lik_appr_prop,
                                        log_lik_aux_curr - log_lik_appr_curr);
     }

     if(acc_beta == 1L){
       beta_curr = beta;
       Xbeta_curr = Xbeta;
     }

     // Primo check
     if((k > (b_gibbs_start - 1)) & (k < (b_gibbs_start + b_check))){
       res = z_aux - log_offset - Xbeta_curr;

       for(int j = 0; j < n_aux; j++){

         if(res[j] > check_mix_max[j]){
           is_adjusted[j] += 1;
         }

         if(res[j] < check_mix_min[j]){
           is_MH[j] += 1;
         }
       }

       if( k == (b_gibbs_start + b_check - 1)){
         for(int j = 0; j < n_aux; j++){
           if((is_MH[j] / b_check) > threshold_MH){
             MH += 1;
           }
           if((is_adjusted[j] / b_check) > threshold_adj){
             w_mix[j] = w_mix_adj[j];
             m_mix[j] = m_mix_adj[j];
             v_mix[j] = v_mix_adj[j];
             MH += 1;
           }
         }
         //Rprintf("IS MH: %d", MH);
       }
     }

     ///////////////////////////////////////////////////////////////////////
     // Storage
     ///////////////////////////////////////////////////////////////////////
     if ((k+1)%thin == 0){
       accept_beta[storeind] = acc_beta;
       beta_mc.row(storeind) = beta_curr;
       storeind += 1;
     }
     if ((k+1)%pr==0){
       time(&now);
       Rprintf("Iteration: %d: %s", k+1, ctime(&now));
     }
   }

   return Rcpp::List::create(
     Rcpp::Named("beta") = beta_mc,
     Rcpp::Named("accept_MH") = accept_beta,
     Rcpp::Named("is_adjusted") = is_adjusted,
     Rcpp::Named("is_MH") = is_MH);
 }


void compute_plp_gamma(Eigen::VectorXd& plp, int j,
                       Eigen::VectorXd Xbeta,
                       std::vector<Eigen::SparseMatrix<double> > Z,
                       std::vector<Eigen::VectorXd> gamma_curr){

  plp = Xbeta;
  for(int i=0; i < Z.size(); i++){
    if(i!=j){
      plp += Z[i] * gamma_curr[i];
    }
  }

}


void compute_Qb_g_fc_poisson(Eigen::SparseMatrix<double>& Q,
                             Eigen::VectorXd& b,
                             Eigen::VectorXd var_aux,
                             Eigen::SparseMatrix<double> K_g,
                             Eigen::SparseMatrix<double> Z,
                             Eigen::VectorXd psres,
                             double s2g){

  Eigen::DiagonalMatrix<double, Dynamic> W = var_aux.cwiseInverse().asDiagonal();
  Q = Z.transpose() * W * Z + K_g/s2g;
  b = (1.0) * Z.transpose() * W * psres;

}

// double lfc_block_poisson(Eigen::VectorXd gamma,
//                          double s2g,
//                          Eigen::VectorXd mu,
//                          Eigen::SparseMatrix<double> Q,
//                          double l_det_Q,
//                          Eigen::MatrixXd A,
//                          Eigen::VectorXd e,
//                          Eigen::MatrixXd W,
//                          Eigen::SparseMatrix<double> K,
//                          int rank,
//                          Rcpp::List pri_s2,
//                          double l_lik_p){
//
//   double lfc_gamma, lfc_joint, lfc;
//
//   Eigen::VectorXd g_mu = gamma - mu;
//
//   lfc_gamma = 0.5 * l_det_Q - 0.5 * g_mu.transpose() * Q * g_mu;
//
//   if (!NumericVector::is_na(A(0,0))){
//     Eigen::VectorXd e_Amu = e - A * mu;
//     lfc_gamma -= - 0.5 * std::log(W.determinant()) - 0.5 * e_Amu.transpose() * W.inverse() * e_Amu;
//   }
//
//   lfc_joint = l_lik_p - (rank / 2.0) * std::log(s2g) -
//     (1.0 / (2.0 * s2g)) * gamma.transpose() * K * gamma +
//     l_pri_s2(s2g, pri_s2);
//   lfc = lfc_joint - lfc_gamma;
//
//   return(lfc);
// }
//

 double l_lik_gauss_poisson(Eigen::VectorXd res,
                            Eigen::VectorXd vars){
   int n_aux = res.size();
   Eigen::VectorXd lliks(n_aux);
   double out = 0.0;

   for(int j=0; j < n_aux; j++){
     out += - 0.5 * std::log(vars[j]) - ((1.0 / (2.0 * vars[j])) * (res[j] * res[j]));
   }

   return(out);
 }
//
//  // [[Rcpp::export]]
//  Rcpp::List sample_poissonm_MH(const Eigen::VectorXi y,//n ordinato
//                                const int n0,
//                                const Eigen::MatrixXd X,//2n-n0
//                                const Rcpp::List Z_list,//2n-n0
//                                const Rcpp::List K_list,
//                                std::vector<Eigen::VectorXd> m_mix_orig,
//                                std::vector<Eigen::VectorXd> v_mix_orig,
//                                std::vector<Eigen::VectorXd> w_mix_orig,
//                                std::vector<Eigen::VectorXd> m_mix_adj,
//                                std::vector<Eigen::VectorXd> v_mix_adj,
//                                std::vector<Eigen::VectorXd> w_mix_adj,
//                                Eigen::VectorXd check_mix,
//                                int b_gibbs,
//                                int adjust,
//                                const Eigen::VectorXd log_offset,
//                                const std::vector<int> rank_K_g,
//                                const Rcpp::List A_list,
//                                const Rcpp::List e_list,
//                                const Eigen::MatrixXd S_beta,
//                                const Rcpp::List pri_s2g,
//                                const Eigen::VectorXd beta_init,
//                                const Rcpp::List g_init_list,
//                                const Eigen::VectorXd S2g_init,
//                                Eigen::VectorXd FFg,
//                                const int niter, const int pr, const int thin,
//                                const int ntuning, const int stop_tuning, double target_accrate){
//
//    //////////////////////////////////////////////////////////////////////
//    // Fixed quantities
//    ///////////////////////////////////////////////////////////////////////
//    int p = X.cols(), n = y.size(),
//      q = Z_list.length(), n_aux = 2 * n - n0;
//
//    Eigen::MatrixXd Q_beta = S_beta.inverse();
//
//    ///////////////////////////////////////////////////////////////////////
//    // Initialise mcmc objects
//    ///////////////////////////////////////////////////////////////////////
//
//
//    ////////////// s2g  //////////////
//    double s2g_prop;
//
//    Eigen::VectorXd
//    s2g_curr = S2g_init;
//
//    std::vector<int>
//      n_acc_s2g(q), Ta_s2g(q);
//
//    //////////////  auxiliary variables  //////////////
//    Eigen::VectorXd
//    z_aux(n_aux), mean_aux(n_aux), var_aux(n_aux), log_lambda(n_aux), res(n_aux);
//
//    std::vector<Eigen::VectorXd> w_mix = w_mix_orig, m_mix = m_mix_orig, v_mix = v_mix_orig;
//
//    //////////////  beta  //////////////
//    Eigen::VectorXd
//    beta = beta_init, Xbeta = X * beta, beta_curr = beta_init, Xbeta_curr = X * beta_curr;
//
//    //////////////  gamma  //////////////
//    std::vector<Eigen::VectorXd>
//      g_curr(q);
//
//    for(int j=0; j<q ;j++){
//      g_curr[j] = Rcpp::as<Eigen::VectorXd>(g_init_list[j]);
//    }
//
//    Eigen::VectorXd
//    Zg = Eigen::VectorXd::Zero(n_aux),
//      g_prop, res_g, b_g, mu_g_fc, plp;
//
//    Eigen::SparseMatrix<double>
//      Q_g_fc;
//
//    Eigen::MatrixXd
//    V, W;
//
//    double
//      ldet_Q_g_fc, llik_p,
//      l_fc_g_s2g_prop, l_fc_g_s2g_curr;
//
//    double log_lik_appr_prop, log_lik_aux_prop, log_lik_appr_curr, log_lik_aux_curr;
//
//    ///////////////////////////////////////////////////////////////////////
//    // Initialise fixed quantities - gamma
//    ///////////////////////////////////////////////////////////////////////
//    std::vector<Eigen::SparseMatrix<double> >
//      Z(q), K_g(q);
//
//    std::vector<Eigen::MatrixXd>
//      A(q);
//
//    std::vector<Eigen::VectorXd>
//      e(q);
//
//    std::vector<int>
//      mj(q);
//
//    for(int j=0; j<q ;j++){
//      Z[j] = Rcpp::as<Eigen::SparseMatrix<double> >(Z_list[j]);
//      mj[j] = Z[j].cols();
//      K_g[j] = Rcpp::as<Eigen::SparseMatrix<double> >(K_list[j]);
//      A[j] = Rcpp::as<Eigen::MatrixXd>(A_list[j]);
//      e[j] = Rcpp::as<Eigen::VectorXd>(e_list[j]);
//    }
//
//    std::vector<Eigen::SimplicialLLT<Eigen::SparseMatrix<double>, Eigen::Lower> >
//      Chol_Q_g(q);
//
//    for(int j=0; j<q ;j++){
//      compute_res_gamma(res_g, j, z_aux, Xbeta + log_offset + mean_aux, Z, g_curr);
//      compute_Qb_g_fc_poisson(Q_g_fc, b_g, var_aux, K_g[j], Z[j], res_g, s2g_curr[j]);
//      Chol_Q_g[j].analyzePattern(Q_g_fc);
//    }
//
//
//    ///////////////////////////////////////////////////////////////////////
//    // Initialise objects for storage
//    ///////////////////////////////////////////////////////////////////////
//    int storeind = 0, nstore = niter / thin;
//
//
//    Eigen::MatrixXd
//    beta_mc(nstore, p),
//    z_store(nstore, n_aux),
//    eps_store(nstore, n_aux);
//
//    Eigen::MatrixXd
//    s2g_mc(nstore, q);
//
//    std::vector<Eigen::MatrixXd>
//      g_mc(q);
//    for(int j=0; j<q; j++){
//      g_mc[j] = Eigen::MatrixXd::Zero(nstore, mj[j]);
//    }
//
//    Eigen::MatrixXd
//    FFgs(nstore, q);
//
//    int acc_beta = 1, acc_gamma = 1;
//    Eigen::VectorXi accept_beta(nstore);
//
//    compute_lliks_app_aux(Xbeta_curr + Zg + log_offset, y, n0, w_mix, m_mix, v_mix, z_aux,
//                          log_lik_appr_curr, log_lik_aux_curr
//    );
//
//    ///////////////////////////////////////////////////////////////////////
//    // MCMC sampler
//    ///////////////////////////////////////////////////////////////////////
//    time_t now;
//    for(int k=0;k<niter;k++){
//
//      ///////////////////////////////////////////////////////////////////////
//      // Sampling z_aux
//      ///////////////////////////////////////////////////////////////////////
//
//      log_lambda = Xbeta_curr + Zg + log_offset;
//
//      //////////////////////////////////////////////////////////////////////
//      /// COMMENTATO QUA
//      sample_aux_Poisson_nolik(log_lambda, y, n0,
//                               w_mix, m_mix, v_mix,
//                               z_aux, var_aux, mean_aux);
//
//      ///////////////////////////////////////////////////////////////////////
//      // Sampling beta and s2b
//      ///////////////////////////////////////////////////////////////////////
//      gibbs_beta_poisson(beta, Xbeta, X,
//                         var_aux, Q_beta, z_aux - Zg - mean_aux - log_offset);
//
//      compute_lliks_app_aux(
//        Xbeta + Zg + log_offset, y, n0, w_mix, m_mix, v_mix, z_aux, log_lik_appr_prop, log_lik_aux_prop
//      );
//      compute_lliks_app_aux(
//        Xbeta_curr + Zg + log_offset, y, n0, w_mix, m_mix, v_mix, z_aux, log_lik_appr_curr, log_lik_aux_curr
//      );
//
//      //tentativo di partire col gibbs per risolvere il problema iniziali
//      if(k > b_gibbs){
//        acc_beta = metropolis_acceptance(log_lik_aux_prop - log_lik_appr_prop,
//                                         log_lik_aux_curr - log_lik_appr_curr);
//      }
//
//      if(acc_beta == 1L){
//        beta_curr = beta;
//        Xbeta_curr = Xbeta;
//        log_lik_aux_curr = log_lik_aux_prop;
//        log_lik_appr_curr = log_lik_appr_prop;
//      }
//
//      ///////////////////////////////////////////////////////////////////////
//      // Sampling (gamma, s2g)
//      ///////////////////////////////////////////////////////////////////////
//      Zg = Zg * 0.0;
//      for(int j=0; j<q ;j++){
//        compute_plp_gamma(plp, j, Xbeta_curr + log_offset, Z, g_curr);
//        res_g = z_aux - plp - mean_aux;
//
//        rprop_rw(s2g_prop, s2g_curr[j], FFg[j]);
//
//        // Full conditionals parameters - curr
//        compute_Qb_g_fc_poisson(Q_g_fc, b_g, var_aux, K_g[j], Z[j], res_g, s2g_curr[j]);
//        Chol_Q_g[j].factorize(Q_g_fc);
//        mu_g_fc = Chol_Q_g[j].solve(b_g);
//        ldet_Q_g_fc = 2.0 * Chol_Q_g[j].matrixL().toDense().diagonal().array().log().sum();
//        if (!Rcpp::NumericVector::is_na(A[j](0,0))){
//          V = Chol_Q_g[j].solve(A[j].transpose());
//          W = A[j] * V;
//          //g_curr[j] è già costretto
//        }
//        llik_p = l_lik_gauss_poisson(res_g - Z[j] * g_curr[j], var_aux);
//        l_fc_g_s2g_curr = lfc_block_poisson(g_curr[j], s2g_curr[j],
//                                            mu_g_fc, Q_g_fc, ldet_Q_g_fc, A[j], e[j], W,
//                                            K_g[j], rank_K_g[j], pri_s2g[j], llik_p);
//
//        // Full conditionals parameters - prop
//        compute_Qb_g_fc_poisson(Q_g_fc, b_g, var_aux, K_g[j], Z[j], res_g, s2g_prop);
//        Chol_Q_g[j].factorize(Q_g_fc);
//        mu_g_fc = Chol_Q_g[j].solve(b_g);
//        ldet_Q_g_fc = 2.0 * Chol_Q_g[j].matrixL().toDense().diagonal().array().log().sum();
//
//        // proposal of gamma
//        g_prop = Chol_Q_g[j].permutationPinv() *
//          Chol_Q_g[j].matrixU().solve(Rcpp::as<Eigen::VectorXd>(Rcpp::rnorm(K_g[j].cols())));
//        g_prop += mu_g_fc;
//
//        if (!Rcpp::NumericVector::is_na(A[j](0,0))){
//          V = Chol_Q_g[j].solve(A[j].transpose());
//          W = A[j] * V;
//          correction_sample_ulc(g_prop, A[j], e[j], W, V);
//        }
//        llik_p = l_lik_gauss_poisson(res_g - Z[j] * g_prop, var_aux);
//        l_fc_g_s2g_prop = lfc_block_poisson(g_prop, s2g_prop,
//                                            mu_g_fc, Q_g_fc, ldet_Q_g_fc, A[j], e[j], W,
//                                            K_g[j], rank_K_g[j], pri_s2g[j], llik_p);
//
//        // acceptance step
//        if(metropolis_acceptance(l_fc_g_s2g_prop, l_fc_g_s2g_curr) == 1L){
//          compute_lliks_app_aux(
//            plp + Z[j]*g_prop, y, n0, w_mix, m_mix, v_mix, z_aux, log_lik_appr_prop, log_lik_aux_prop
//          );
//          compute_lliks_app_aux(
//            plp + Z[j]*g_curr[j], y, n0, w_mix, m_mix, v_mix, z_aux, log_lik_appr_curr, log_lik_aux_curr
//          );
//
//          if(k > b_gibbs){
//            acc_gamma = metropolis_acceptance(log_lik_aux_prop - log_lik_appr_prop,
//                                              log_lik_aux_curr - log_lik_appr_curr);
//          }
//
//          if(acc_gamma == 1L){
//            g_curr[j] = g_prop;
//            s2g_curr[j] = s2g_prop;
//            log_lik_aux_curr = log_lik_aux_prop;
//            log_lik_appr_curr = log_lik_appr_prop;
//            n_acc_s2g[j] += 1;
//          }
//        }
//
//        Zg += Z[j]*g_curr[j];
//      }
//
//      if((k < b_gibbs) & (adjust == 1)){
//        res = z_aux - log_offset - Xbeta_curr - Zg;
//        for(int j = 0; j < n_aux; j++){
//          if(res[j] > check_mix[j]){
//            w_mix[j] = w_mix_adj[j];
//            m_mix[j] = m_mix_adj[j];
//            v_mix[j] = v_mix_adj[j];
//          }
//        }
//      }
//
//      ///////////////////////////////////////////////////////////////////////
//      // Tuning
//      ///////////////////////////////////////////////////////////////////////
//      if(k < stop_tuning){
//        if((k+1)%ntuning==0) {
//          for(int j=0; j<q ;j++){
//            tune(FFg[j], n_acc_s2g[j], ntuning, target_accrate, Ta_s2g[j]);
//            FFgs(Ta_s2g[j]-1, j) = FFg[j];
//          }
//        }
//      }
//
//      ///////////////////////////////////////////////////////////////////////
//      // Storage
//      ///////////////////////////////////////////////////////////////////////
//      if ((k+1)%thin == 0){
//        beta_mc.row(storeind) = beta_curr;
//        accept_beta(storeind) = acc_beta;
//        s2g_mc.row(storeind) = s2g_curr;
//        for(int j=0;j<q;j++){
//          g_mc[j].row(storeind) = g_curr[j];
//        }
//        storeind += 1;
//      }
//
//      if ((k+1)%pr==0){
//        time(&now);
//        Rprintf("Iteration: %d: %s", k+1, ctime(&now));
//      }
//
//    }
//
//    return Rcpp::List::create(
//      Rcpp::Named("s2g") = s2g_mc,
//      Rcpp::Named("beta") = beta_mc,
//      Rcpp::Named("gamma") = g_mc,
//      Rcpp::Named("FFgs") = FFgs,
//      Rcpp::Named("accept_beta")= accept_beta,
//      Rcpp::Named("n_acc_s2g")= n_acc_s2g);
//  }
//



 // [[Rcpp::export]]
 Rcpp::List sample_poissonm_nob(const Eigen::VectorXi y,//n ordinato
                                const int n0,
                                const int MH,
                                const Eigen::VectorXd a_pri,
                                const Eigen::VectorXd b_pri,
                                const Eigen::MatrixXd X,//2n-n0
                                const Rcpp::List Z_list,//2n-n0
                                const Rcpp::List K_list,
                                std::vector<Eigen::VectorXd> m_mix_orig,
                                std::vector<Eigen::VectorXd> v_mix_orig,
                                std::vector<Eigen::VectorXd> w_mix_orig,
                                std::vector<Eigen::VectorXd> m_mix_adj,
                                std::vector<Eigen::VectorXd> v_mix_adj,
                                std::vector<Eigen::VectorXd> w_mix_adj,
                                Eigen::VectorXd check_mix,
                                int b_check,
                                int adjust,
                                int b_gibbs_start,
                                const Eigen::VectorXd log_offset,
                                const std::vector<int> rank_K_g,
                                const Rcpp::List A_list,
                                const Rcpp::List e_list,
                                const Eigen::MatrixXd S_beta,
                                const Eigen::VectorXd beta_init,
                                const Rcpp::List g_init_list,
                                const Eigen::VectorXd S2g_init,
                                const int niter, const int pr, const int thin){

   //////////////////////////////////////////////////////////////////////
   // Fixed quantities
   ///////////////////////////////////////////////////////////////////////
   int p = X.cols(), n = y.size(),
     q = Z_list.length(), n_aux = 2 * n - n0;

   Eigen::MatrixXd Q_beta = S_beta.inverse();

   ///////////////////////////////////////////////////////////////////////
   // Initialise mcmc objects
   ///////////////////////////////////////////////////////////////////////



   ////////////// s2g  //////////////

   Eigen::VectorXd
   s2g_curr = S2g_init;

   std::vector<int>
     n_acc_s2g(q), Ta_s2g(q);

   //////////////  auxiliary variables  //////////////
   Eigen::VectorXd
   z_aux(n_aux), mean_aux(n_aux), var_aux(n_aux), log_lambda(n_aux),
   epsilon(n_aux), res(n_aux);

   std::vector<Eigen::VectorXd> w_mix = w_mix_orig, m_mix = m_mix_orig, v_mix = v_mix_orig;

   //////////////  beta  //////////////
   Eigen::VectorXd
   beta = beta_init, Xbeta = X * beta, beta_curr = beta_init, Xbeta_curr = X * beta_curr;


   //////////////  gamma  //////////////
   std::vector<Eigen::VectorXd>
     g_curr(q);

   for(int j=0; j<q ;j++){
     g_curr[j] = Rcpp::as<Eigen::VectorXd>(g_init_list[j]);
   }

   Eigen::VectorXd
   Zg = Eigen::VectorXd::Zero(n_aux),
     g_prop, res_g, b_g, mu_g_fc, plp;

   Eigen::SparseMatrix<double>
     Q_g_fc;

   Eigen::MatrixXd
   V, W;

   double log_lik_appr_prop, log_lik_aux_prop, log_lik_appr_curr, log_lik_aux_curr;

   ///////////////////////////////////////////////////////////////////////
   // Initialise fixed quantities - gamma
   ///////////////////////////////////////////////////////////////////////
   std::vector<Eigen::SparseMatrix<double> >
     Z(q), K_g(q);

   std::vector<Eigen::MatrixXd>
     A(q);

   std::vector<Eigen::VectorXd>
     e(q);

   std::vector<int>
     mj(q);

   for(int j=0; j<q ;j++){
     Z[j] = Rcpp::as<Eigen::SparseMatrix<double> >(Z_list[j]);
     mj[j] = Z[j].cols();
     K_g[j] = Rcpp::as<Eigen::SparseMatrix<double> >(K_list[j]);
     A[j] = Rcpp::as<Eigen::MatrixXd>(A_list[j]);
     e[j] = Rcpp::as<Eigen::VectorXd>(e_list[j]);
   }

   std::vector<Eigen::SimplicialLLT<Eigen::SparseMatrix<double>, Eigen::Lower> >
     Chol_Q_g(q);

   for(int j=0; j<q ;j++){
     compute_res_gamma(res_g, j, z_aux, Xbeta + log_offset + mean_aux, Z, g_curr);
     compute_Qb_g_fc_poisson(Q_g_fc, b_g, var_aux, K_g[j], Z[j], res_g, s2g_curr[j]);
     Chol_Q_g[j].analyzePattern(Q_g_fc);
   }

   ///////////////////////////////////////////////////////////////////////
   // Initialise objects for storage
   ///////////////////////////////////////////////////////////////////////
   int storeind = 0, nstore = niter / thin, acc_beta = 0, acc_gamma = 0;

   Eigen::MatrixXd
   beta_mc(nstore, p),
   z_store(nstore, n_aux),
   eps_store(nstore, n_aux),
   log_lik_appr(nstore, n_aux),
   log_lik_aux(nstore, n_aux);

   Eigen::VectorXd accept_beta(nstore);

   Eigen::MatrixXd
   s2g_mc(nstore, q);

   std::vector<Eigen::MatrixXd>
     g_mc(q);
   for(int j=0; j<q; j++){
     g_mc[j] = Eigen::MatrixXd::Zero(nstore, mj[j]);
   }

   compute_lliks_app_aux(Xbeta_curr + Zg + log_offset, y, n0, w_mix, m_mix, v_mix, z_aux,
                         log_lik_appr_curr, log_lik_aux_curr
   );



   ///////////////////////////////////////////////////////////////////////
   // MCMC sampler
   ///////////////////////////////////////////////////////////////////////
   time_t now;
   for(int k=0;k<niter;k++){

     ///////////////////////////////////////////////////////////////////////
     // Sampling z_aux
     ///////////////////////////////////////////////////////////////////////

     log_lambda = Xbeta_curr + Zg + log_offset;

     //////////////////////////////////////////////////////////////////////
     /// COMMENTATO QUA
     sample_aux_Poisson_nolik(log_lambda, y, n0,
                              w_mix, m_mix, v_mix,
                              z_aux, var_aux, mean_aux);

     epsilon = z_aux - log_lambda;

     ///////////////////////////////////////////////////////////////////////
     // Sampling beta and s2b
     ///////////////////////////////////////////////////////////////////////
     gibbs_beta_poisson(beta, Xbeta, X,
                        var_aux, Q_beta, z_aux - Zg - mean_aux - log_offset);

     if(MH == 1){
       compute_lliks_app_aux(
         Xbeta + Zg + log_offset, y, n0, w_mix, m_mix, v_mix, z_aux, log_lik_appr_prop, log_lik_aux_prop
       );
       compute_lliks_app_aux(
         Xbeta_curr + Zg + log_offset, y, n0, w_mix, m_mix, v_mix, z_aux, log_lik_appr_curr, log_lik_aux_curr
       );

       acc_beta = 0;
       //tentativo di partire col gibbs per risolvere il problema iniziali
       if(k<b_gibbs_start){
         acc_beta = 1;
       } else{
         acc_beta = metropolis_acceptance(log_lik_aux_prop - log_lik_appr_prop,
                                          log_lik_aux_curr - log_lik_appr_curr);
       }

       if(acc_beta == 1L){
         beta_curr = beta;
         Xbeta_curr = Xbeta;
         log_lik_aux_curr = log_lik_aux_prop;
         log_lik_appr_curr = log_lik_appr_prop;
       }
     } else{
       beta_curr = beta;
       Xbeta_curr = Xbeta;
     }

     ///////////////////////////////////////////////////////////////////////
     // Sampling (gamma, s2g)
     ///////////////////////////////////////////////////////////////////////
     Zg = Zg * 0.0;
     for(int j=0; j<q ;j++){
       compute_plp_gamma(plp, j, Xbeta_curr + log_offset, Z, g_curr);
       res_g = z_aux - plp - mean_aux;

       // Full conditionals parameters - prop
       compute_Qb_g_fc_poisson(Q_g_fc, b_g, var_aux, K_g[j], Z[j], res_g, s2g_curr[j]);
       Chol_Q_g[j].factorize(Q_g_fc);
       mu_g_fc = Chol_Q_g[j].solve(b_g);

       // proposal of gamma
       g_prop = Chol_Q_g[j].permutationPinv() *
         Chol_Q_g[j].matrixU().solve(Rcpp::as<Eigen::VectorXd>(Rcpp::rnorm(K_g[j].cols())));
       g_prop += mu_g_fc;


       /////////////////// QUI SOTTO VINCOLO
       //
       if (!Rcpp::NumericVector::is_na(A[j](0,0))){
         V = Chol_Q_g[j].solve(A[j].transpose());
         W = A[j] * V;
         correction_sample_ulc(g_prop, A[j], e[j], W, V);
       }

       if(MH == 1){
         compute_lliks_app_aux(
           plp + Z[j]*g_prop, y, n0, w_mix, m_mix, v_mix, z_aux, log_lik_appr_prop, log_lik_aux_prop
         );
         compute_lliks_app_aux(
           plp + Z[j]*g_curr[j], y, n0, w_mix, m_mix, v_mix, z_aux, log_lik_appr_curr, log_lik_aux_curr
         );

         acc_gamma = 0;
         //tentativo di partire col gibbs per risolvere il problema iniziali
         if(k<b_gibbs_start){
           acc_gamma = 1;
         } else{
           acc_gamma = metropolis_acceptance(log_lik_aux_prop - log_lik_appr_prop,
                                             log_lik_aux_curr - log_lik_appr_curr);
         }

         if(acc_gamma == 1L){
           g_curr[j] = g_prop;
           log_lik_aux_curr = log_lik_aux_prop;
           log_lik_appr_curr = log_lik_appr_prop;
           n_acc_s2g[j] += 1;
         }
       } else{
         g_curr[j] = g_prop;
         n_acc_s2g[j] += 1;
       }

       s2g_curr[j] = 1.0 / Rcpp::rgamma(1, a_pri[j] + 0.5*rank_K_g[j],
                                        1.0/(b_pri[j] + 0.5 *g_curr[j].transpose() * K_g[j] * g_curr[j]))[0];

       Zg += Z[j]*g_curr[j];
     }

     res = z_aux - log_offset - Xbeta_curr - Zg;

     if((k < b_gibbs_start + b_check) & (adjust == 1)){
       for(int j=0; j<n_aux; j++){
         if(res[j]>check_mix[j]){
           w_mix[j] = w_mix_adj[j];
           m_mix[j] = m_mix_adj[j];
           v_mix[j] = v_mix_adj[j];
         }
       }
     }

     ///////////////////////////////////////////////////////////////////////
     // Storage
     ///////////////////////////////////////////////////////////////////////
     if ((k+1)%thin == 0){
       beta_mc.row(storeind) = beta_curr;
       accept_beta(storeind) = acc_beta;
       s2g_mc.row(storeind) = s2g_curr;
       z_store.row(storeind) = z_aux;
       eps_store.row(storeind) = epsilon;
       for(int j=0;j<q;j++){
         g_mc[j].row(storeind) = g_curr[j];
       }
       storeind += 1;
     }

     if ((k+1)%pr==0){
       time(&now);
       Rprintf("Iteration: %d: %s", k+1, ctime(&now));
     }

   }

   return Rcpp::List::create(
     Rcpp::Named("s2g") = s2g_mc,
     Rcpp::Named("beta") = beta_mc,
     Rcpp::Named("zeta") = z_store,
     Rcpp::Named("eps") = eps_store,
     Rcpp::Named("gamma") = g_mc,
     Rcpp::Named("n_acc_s2g") = n_acc_s2g,
     Rcpp::Named("accept_beta") = accept_beta,
     Rcpp::Named("l_lik_app") = log_lik_appr,
     Rcpp::Named("l_lik_aux") = log_lik_aux);

 }


 // [[Rcpp::export]]
 Rcpp::List sample_poissonm_nob_auto(const Eigen::VectorXi y,//n ordinato
                                     const int n0,
                                     const Eigen::VectorXd a_pri,
                                     const Eigen::VectorXd b_pri,
                                     const Eigen::MatrixXd X,//2n-n0
                                     const Rcpp::List Z_list,//2n-n0
                                     const Rcpp::List K_list,
                                     std::vector<Eigen::VectorXd> m_mix_orig,
                                     std::vector<Eigen::VectorXd> v_mix_orig,
                                     std::vector<Eigen::VectorXd> w_mix_orig,
                                     std::vector<Eigen::VectorXd> m_mix_adj,
                                     std::vector<Eigen::VectorXd> v_mix_adj,
                                     std::vector<Eigen::VectorXd> w_mix_adj,
                                     Eigen::VectorXd check_mix_min,
                                     Eigen::VectorXd check_mix_max,
                                     int b_gibbs_start,
                                     int b_check,
                                     double threshold_MH,
                                     double threshold_adj,
                                     const Eigen::VectorXd log_offset,
                                     const std::vector<int> rank_K_g,
                                     const Rcpp::List A_list,
                                     const Rcpp::List e_list,
                                     const Eigen::MatrixXd S_beta,
                                     const Eigen::VectorXd beta_init,
                                     const Rcpp::List g_init_list,
                                     const Eigen::VectorXd S2g_init,
                                     const int niter, const int pr, const int thin){

   //////////////////////////////////////////////////////////////////////
   // Fixed quantities
   ///////////////////////////////////////////////////////////////////////
   int p = X.cols(), n = y.size(),
     q = Z_list.length(), n_aux = 2 * n - n0;

   Eigen::MatrixXd Q_beta = S_beta.inverse();

   ///////////////////////////////////////////////////////////////////////
   // Initialise mcmc objects
   ///////////////////////////////////////////////////////////////////////
   int MH = 0;
   Eigen::VectorXd
   is_adjusted = Eigen::VectorXd::Zero(n_aux), is_MH = Eigen::VectorXd::Zero(n_aux);


   ////////////// s2g  //////////////
   Eigen::VectorXd
   s2g_curr = S2g_init;

   std::vector<int>
     n_acc_s2g(q), Ta_s2g(q);

   //////////////  auxiliary variables  //////////////
   Eigen::VectorXd
   z_aux(n_aux), mean_aux(n_aux), var_aux(n_aux), log_lambda(n_aux),
   epsilon(n_aux), res(n_aux);

   std::vector<Eigen::VectorXd> w_mix = w_mix_orig, m_mix = m_mix_orig, v_mix = v_mix_orig;

   //////////////  beta  //////////////
   Eigen::VectorXd
   beta = beta_init, Xbeta = X * beta, beta_curr = beta_init, Xbeta_curr = X * beta_curr;


   //////////////  gamma  //////////////
   std::vector<Eigen::VectorXd>
     g_curr(q);

   for(int j=0; j<q ;j++){
     g_curr[j] = Rcpp::as<Eigen::VectorXd>(g_init_list[j]);
   }

   Eigen::VectorXd
   Zg = Eigen::VectorXd::Zero(n_aux),
     g_prop, res_g, b_g, mu_g_fc, plp;

   Eigen::SparseMatrix<double>
     Q_g_fc;

   Eigen::MatrixXd
   V, W;

   double log_lik_appr_prop, log_lik_aux_prop, log_lik_appr_curr, log_lik_aux_curr;

   ///////////////////////////////////////////////////////////////////////
   // Initialise fixed quantities - gamma
   ///////////////////////////////////////////////////////////////////////
   std::vector<Eigen::SparseMatrix<double> >
     Z(q), K_g(q);

   std::vector<Eigen::MatrixXd>
     A(q);

   std::vector<Eigen::VectorXd>
     e(q);

   std::vector<int>
     mj(q);

   for(int j=0; j<q ;j++){
     Z[j] = Rcpp::as<Eigen::SparseMatrix<double> >(Z_list[j]);
     mj[j] = Z[j].cols();
     K_g[j] = Rcpp::as<Eigen::SparseMatrix<double> >(K_list[j]);
     A[j] = Rcpp::as<Eigen::MatrixXd>(A_list[j]);
     e[j] = Rcpp::as<Eigen::VectorXd>(e_list[j]);
   }

   std::vector<Eigen::SimplicialLLT<Eigen::SparseMatrix<double>, Eigen::Lower> >
     Chol_Q_g(q);

   for(int j=0; j<q ;j++){
     compute_res_gamma(res_g, j, z_aux, Xbeta + log_offset + mean_aux, Z, g_curr);
     compute_Qb_g_fc_poisson(Q_g_fc, b_g, var_aux, K_g[j], Z[j], res_g, s2g_curr[j]);
     Chol_Q_g[j].analyzePattern(Q_g_fc);
   }

   ///////////////////////////////////////////////////////////////////////
   // Initialise objects for storage
   ///////////////////////////////////////////////////////////////////////
   int storeind = 0, nstore = niter / thin, acc_beta = 0, acc_gamma = 0;

   Eigen::MatrixXd
   beta_mc(nstore, p),
   z_store(nstore, n_aux),
   eps_store(nstore, n_aux),
   log_lik_appr(nstore, n_aux),
   log_lik_aux(nstore, n_aux);

   Eigen::VectorXd accept_beta(nstore);

   Eigen::MatrixXd
   s2g_mc(nstore, q);

   std::vector<Eigen::MatrixXd>
     g_mc(q);
   for(int j=0; j<q; j++){
     g_mc[j] = Eigen::MatrixXd::Zero(nstore, mj[j]);
   }

   compute_lliks_app_aux(Xbeta_curr + Zg + log_offset, y, n0, w_mix, m_mix, v_mix, z_aux,
                         log_lik_appr_curr, log_lik_aux_curr
   );



   ///////////////////////////////////////////////////////////////////////
   // MCMC sampler
   ///////////////////////////////////////////////////////////////////////
   time_t now;
   for(int k=0;k<niter;k++){

     ///////////////////////////////////////////////////////////////////////
     // Sampling z_aux
     ///////////////////////////////////////////////////////////////////////

     log_lambda = Xbeta_curr + Zg + log_offset;

     //////////////////////////////////////////////////////////////////////
     /// COMMENTATO QUA
     sample_aux_Poisson_nolik(log_lambda, y, n0,
                              w_mix, m_mix, v_mix,
                              z_aux, var_aux, mean_aux);

     epsilon = z_aux - log_lambda;

     ///////////////////////////////////////////////////////////////////////
     // Sampling beta and s2b
     ///////////////////////////////////////////////////////////////////////
     gibbs_beta_poisson(beta, Xbeta, X,
                        var_aux, Q_beta, z_aux - Zg - mean_aux - log_offset);

     if(MH > 0){
       compute_lliks_app_aux(
         Xbeta + Zg + log_offset, y, n0, w_mix, m_mix, v_mix, z_aux, log_lik_appr_prop, log_lik_aux_prop
       );
       compute_lliks_app_aux(
         Xbeta_curr + Zg + log_offset, y, n0, w_mix, m_mix, v_mix, z_aux, log_lik_appr_curr, log_lik_aux_curr
       );


       //Rprintf("%f, %f \n", log_lik_aux_prop, log_lik_appr_prop);
       acc_beta = metropolis_acceptance(log_lik_aux_prop - log_lik_appr_prop,
                                        log_lik_aux_curr - log_lik_appr_curr);

       if(acc_beta == 1L){
         beta_curr = beta;
         Xbeta_curr = Xbeta;
         log_lik_aux_curr = log_lik_aux_prop;
         log_lik_appr_curr = log_lik_appr_prop;
       }
     } else{
       beta_curr = beta;
       Xbeta_curr = Xbeta;
     }

     ///////////////////////////////////////////////////////////////////////
     // Sampling (gamma, s2g)
     ///////////////////////////////////////////////////////////////////////
     Zg = Zg * 0.0;
     for(int j=0; j<q ;j++){
       compute_plp_gamma(plp, j, Xbeta_curr + log_offset, Z, g_curr);
       res_g = z_aux - plp - mean_aux;

       // Full conditionals parameters - prop
       compute_Qb_g_fc_poisson(Q_g_fc, b_g, var_aux, K_g[j], Z[j], res_g, s2g_curr[j]);
       Chol_Q_g[j].factorize(Q_g_fc);
       mu_g_fc = Chol_Q_g[j].solve(b_g);

       // proposal of gamma
       g_prop = Chol_Q_g[j].permutationPinv() *
         Chol_Q_g[j].matrixU().solve(Rcpp::as<Eigen::VectorXd>(Rcpp::rnorm(K_g[j].cols())));
       g_prop += mu_g_fc;


       /////////////////// QUI SOTTO VINCOLO
       //
       if (!Rcpp::NumericVector::is_na(A[j](0,0))){
         V = Chol_Q_g[j].solve(A[j].transpose());
         W = A[j] * V;
         correction_sample_ulc(g_prop, A[j], e[j], W, V);
       }

       if(MH > 0){
         compute_lliks_app_aux(
           plp + Z[j]*g_prop, y, n0, w_mix, m_mix, v_mix, z_aux, log_lik_appr_prop, log_lik_aux_prop
         );
         compute_lliks_app_aux(
           plp + Z[j]*g_curr[j], y, n0, w_mix, m_mix, v_mix, z_aux, log_lik_appr_curr, log_lik_aux_curr
         );
         //Rprintf("%f, %f \n", log_lik_aux_prop, log_lik_appr_prop);
         acc_gamma = metropolis_acceptance(log_lik_aux_prop - log_lik_appr_prop,
                                           log_lik_aux_curr - log_lik_appr_curr);

         if(acc_gamma == 1L){
           g_curr[j] = g_prop;
           log_lik_aux_curr = log_lik_aux_prop;
           log_lik_appr_curr = log_lik_appr_prop;
           n_acc_s2g[j] += 1;
         }
       } else{
         g_curr[j] = g_prop;
         n_acc_s2g[j] += 1;
       }

       s2g_curr[j] = 1.0 / Rcpp::rgamma(1, a_pri[j] + 0.5*rank_K_g[j],
                                        1.0/(b_pri[j] + 0.5 *g_curr[j].transpose() * K_g[j] * g_curr[j]))[0];

       Zg += Z[j]*g_curr[j];
     }


     // Primo check
     if((k > (b_gibbs_start - 1)) & (k < (b_gibbs_start + b_check))){
       res = z_aux - log_offset - Xbeta_curr - Zg;

       for(int j = 0; j < n_aux; j++){

         if(res[j] > check_mix_max[j]){
           is_adjusted[j] += 1;
         }

         if(res[j] < check_mix_min[j]){
           is_MH[j] += 1;
         }
       }

       if( k == (b_gibbs_start + b_check - 1)){
         for(int j = 0; j < n_aux; j++){
           if((is_MH[j] / b_check) > threshold_MH){
             MH += 1;
           }
           if((is_adjusted[j] / b_check) > threshold_adj){
             w_mix[j] = w_mix_adj[j];
             m_mix[j] = m_mix_adj[j];
             v_mix[j] = v_mix_adj[j];
             MH += 1;
           }
         }
       }
     }




     ///////////////////////////////////////////////////////////////////////
     // Storage
     ///////////////////////////////////////////////////////////////////////
     if ((k+1)%thin == 0){
       beta_mc.row(storeind) = beta_curr;
       accept_beta(storeind) = acc_beta;
       s2g_mc.row(storeind) = s2g_curr;
       z_store.row(storeind) = z_aux;
       eps_store.row(storeind) = epsilon;
       for(int j=0;j<q;j++){
         g_mc[j].row(storeind) = g_curr[j];
       }
       storeind += 1;
     }

     if ((k+1)%pr==0){
       time(&now);
       Rprintf("Iteration: %d: %s", k+1, ctime(&now));
     }

   }

   return Rcpp::List::create(
     Rcpp::Named("s2g") = s2g_mc,
     Rcpp::Named("beta") = beta_mc,
     Rcpp::Named("gamma") = g_mc,
     Rcpp::Named("accept_beta") = accept_beta,
     Rcpp::Named("n_acc_s2g") = n_acc_s2g,
     //Rcpp::Named("resid_aux") = eps_store,
     Rcpp::Named("is_adjusted") = is_adjusted,
     Rcpp::Named("is_MH") = is_MH);

 }

