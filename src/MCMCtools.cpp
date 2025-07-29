#define RCPPDIST_DONT_USE_ARMA

#include <RcppEigen.h>
#include <Rcpp.h>
#include <complex>
#include <RcppDist.h>

using namespace Rcpp;
using namespace Eigen;


//
// void rprop_rw(double& s2_prop,
//               double s2_curr,
//               double FF){
//   double x_gen;
//   int xx;
//   double ld;
//   if(FF <= 1.0){
//     x_gen = 1.0;
//   }else{
//     ld = FF - 1.0 / FF;
//     xx = Rcpp::as<int>(Rcpp::rbinom(1, 1, ld / (ld + 2.0 * std::log(FF))));
//     if(xx > 0){
//       x_gen = 1.0 / FF + Rcpp::as<double>(Rcpp::runif(1, 0.0, ld));
//     }else{
//       x_gen = std::pow(FF, Rcpp::as<double>(Rcpp::runif(1,-1.0,1.0)));
//     }
//
//   }
//   s2_prop = x_gen * s2_curr;
// }
//
int metropolis_acceptance(double lfc_prop,
                          double lfc_curr){
  double prob_acc = (lfc_prop - lfc_curr);

  //Rprintf("log prob: %f \n", prob_acc);
  prob_acc = std::exp(prob_acc);
  //Rprintf("prob: %f \n", prob_acc);

  if(prob_acc > 1.0){
    prob_acc = 1.0;
  }
  int acc = Rcpp::rbinom(1, 1, prob_acc)[0];
  return(acc);
}

// void tune(double& FF,
//           int& n_acc,
//           int ntuning,
//           double a_opt,
//           int& Ta){
//
//   double acc_r = (n_acc * 1.0) / (ntuning * 1.0);
//
//   FF = FF * std::exp(10.0/std::pow(Ta*1.0+3.0,0.8)*(acc_r-a_opt));
//
//   if(FF < 1.01) {
//     FF = 1.01;
//   }
//   n_acc = 0;
//   Ta += 1;
// }
//



int is_mixture_pri_beta(Rcpp::List pri){
  int out =1;
  std::string check = pri["Mixture"];
  if(check != "Yes"){
    out = 0;
  }
  return(out);
}


// void metropolis_s2b(Eigen::VectorXd& s2b,
//                     Eigen::MatrixXd& Q_beta,
//                     Eigen::VectorXd beta,
//                     Rcpp::List pri_s2b,
//                     Eigen::VectorXd FFb,
//                     std::vector<int>& n_acc_s2b){
//
//   double s2b_prop, lfc_curr, lfc_prop;
//
//   for(int j = 0; j < beta.size(); j++){
//     if(is_mixture_pri_beta(pri_s2b[j]) == 1){
//       lfc_curr = - 0.5 * std::log(s2b[j]) - 0.5 * beta[j] * beta[j] / s2b[j];
//       lfc_curr += l_pri_s2(s2b[j], pri_s2b[j]);
//
//       rprop_rw(s2b_prop, s2b[j], FFb[j]);
//       lfc_prop = - 0.5 * std::log(s2b_prop) - 0.5 * beta[j] * beta[j] / s2b_prop;
//       lfc_prop += l_pri_s2(s2b_prop, pri_s2b[j]);
//
//       if(metropolis_acceptance(lfc_prop, lfc_curr) == 1L){
//         s2b[j] = s2b_prop;
//         n_acc_s2b[j] +=1;
//       }
//     }
//   }
//   Q_beta.diagonal() = s2b.cwiseInverse();
// }


void compute_res_gamma(Eigen::VectorXd& res_gamma, int j,
                       Eigen::VectorXd y,
                       Eigen::VectorXd Xbeta,
                       std::vector<Eigen::SparseMatrix<double> > Z,
                       std::vector<Eigen::VectorXd> gamma_curr){

  res_gamma = y - Xbeta;
  for(int i=0; i < Z.size(); i++){
    if(i!=j){
      res_gamma -= Z[i] * gamma_curr[i];
    }
  }

}


void correction_sample_ulc(Eigen::VectorXd& gamma,
                           Eigen::MatrixXd A,
                           Eigen::VectorXd e,
                           Eigen::MatrixXd W,
                           Eigen::MatrixXd V){
  Eigen::MatrixXd U = W.llt().solve(V.transpose());
  Eigen::VectorXd C = A * gamma - e;
  gamma -= U.transpose() * C;

}



