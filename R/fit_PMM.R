sample_poissonm_nob <- function(y, n0, MH, a_pri, b_pri, X, Z_list, K_list, m_mix_orig, v_mix_orig, w_mix_orig, m_mix_adj, v_mix_adj, w_mix_adj, check_mix, b_check, adjust, b_gibbs_start, log_offset, rank_K_g, A_list, e_list, S_beta, beta_init, g_init_list, S2g_init, niter, pr, thin) {
  .Call(`_SamplerPoisson_sample_poissonm_nob`, y, n0, MH, a_pri, b_pri, X, Z_list, K_list, m_mix_orig, v_mix_orig, w_mix_orig, m_mix_adj, v_mix_adj, w_mix_adj, check_mix, b_check, adjust, b_gibbs_start, log_offset, rank_K_g, A_list, e_list, S_beta, beta_init, g_init_list, S2g_init, niter, pr, thin)
}

sample_poissonm_nob_auto <- function(y, n0, a_pri, b_pri, X, Z_list, K_list, m_mix_orig, v_mix_orig, w_mix_orig, m_mix_adj, v_mix_adj, w_mix_adj, check_mix_min, check_mix_max, b_gibbs_start, b_check, threshold_MH, threshold_adj, log_offset, rank_K_g, A_list, e_list, S_beta, beta_init, g_init_list, S2g_init, niter, pr, thin) {
  .Call(`_SamplerPoisson_sample_poissonm_nob_auto`, y, n0, a_pri, b_pri, X, Z_list, K_list, m_mix_orig, v_mix_orig, w_mix_orig, m_mix_adj, v_mix_adj, w_mix_adj, check_mix_min, check_mix_max, b_gibbs_start, b_check, threshold_MH, threshold_adj, log_offset, rank_K_g, A_list, e_list, S_beta, beta_init, g_init_list, S2g_init, niter, pr, thin)
}

#' @title Draw posterior samples from a Bayesian linear mixed model
#'
#' @description Allow the user to specify a Bayesian linear mixed model, with flexibility in the prior choices.
#'
#' @param y Vector of observed responses.
#' @param X Matrix with fixed effects. The intercept is added by the function.
#' @param offset Vector with possible offsets. Default all 1 values.
#' @param reff_list List of lists containing information on random effects. For each term, the list must contain: a design matrix `Z` with a number of columns `m_j`, a precision matrix `K` with its rank `rank_K` and a logical `K_full_rank` (`TRUE` if `K` is full rank), and the possible linear constraint defined by a matrix `A` and a vector `e`.
#' @param a_pri_s2g Vector with the shape parameters of the Inverse-Gamma priors for the random effects variances.
#' @param b_pri_s2g Vector with the scale parameters of the Inverse-Gamma priors for the random effects variances.
#' @param prior_precision_beta Precision of the Gaussian prior of the regression coefficients.
#' @param beta_init Initial values for the vector of regression coefficients.
#' @param s2g_init Initial values for the random effects variances.
#' @param algorithm Choice of the algorithm: "Automatic", "IAMS", "MH-IAMS" and "RIAMS".
#' @param n_iter Number of MCMC iterations.
#' @param pr Frequency of iterations for printing the progress.
#' @param thin Number of thinned observations.
#' @param T1 Number of iterations with IAMS algorithm for initializing the algorithm (used in "Automatic", "MH-IAMS" and "RIAMS")
#' @param T2 Number of iterations in which the need for mixture adjustment ("Automatic" and "RIAMS") and MH step ("Automatic") is verified.
#' @param pL Threshold for the proportion of problematic approximations to activate the MH step ("Automatic").
#' @param pU Threshold for the proportion of problematic approximations to activate the mixture adjustment ("Automatic").
#'
#' @return
#' The function returns a list with:
#'  * `algorithm`.
#'  * `out`: List with posterior samples of regression coefficients.
#'  * `elapsed_time`.
#'  * `accept_beta`: acceptance rate for the vector of regression coefficients (except for "IAMS" algorithm).
#'  * `accept_gamma`: acceptance rates for the vectors of random effects (except for "IAMS" algorithm).
#'  * `kU`: for each residual, it contains the proportion of preliminary iterations falling in the upper tail ("RIAMS" and "Automatic" algorithms).
#'  * `kL`: for each residual, it contains the proportion of preliminary iterations falling in the lower tail ("Automatic" algorithm).
#'
#'
#'
#'
#'
#' @export
#'





sample_PMM <-  function(y, X, offset = rep(1, length(y)),
                        reff_list,
                        a_pri_s2g, b_pri_s2g,
                        prior_precision_beta = 0.001,
                        beta_init = rep(0, ncol(X)),
                        s2g_init = rep(1, length(reff_list)),
                        algorithm = c("Automatic","IAMS", "MH-IAMS", "RIAMS"),
                        n_iter = 10000, pr = n_iter / 10, thin = 1,
                        T1 = 500, T2 = 250,
                        pL = 0.05, pU = 0.05){

  alg <- match.arg(algorithm)

  # ricevuti in input y e X
  ord <- order(y)
  y_s <- y[ord]
  X_s <- X[ord, ]
  n_0 <- sum(y_s == 0)
  n <- length(y_s)
  n_aux <- 2 * n - n_0
  if(ncol(X)>1){
    X_mix <- rbind(X_s, X_s[y_s>0,])
  }else{
    X_mix <- matrix(c(X_s, X_s[y_s>0]), ncol=1)
  }
  E_mix <- offset[ord]
  E_mix <- c(E_mix, E_mix[y_s>0])

  if(length(a_pri_s2g) == 1 | length(b_pri_s2g) == 1){
    a_pri_s2g <- rep(a_pri_s2g[1], times = length(reff_list))
    b_pri_s2g <- rep(b_pri_s2g[1], times = length(reff_list))
  }
  if(length(a_pri_s2g) != length(reff_list) | length(b_pri_s2g) != length(reff_list)){
    stop("The arguments 'a_pri_s2g' and 'b_pri_s2g' must be scalars or vector with the same size of 'reff_list'.")
  }

  # Setting Random effects
  K_list <- purrr::map(reff_list, ~.$K)
  names(K_list)<-NULL
  Z_list <- purrr::map(reff_list, ~.$Z[ord,])
  rank_K <- purrr::map_dbl(reff_list, ~.$rank)
  names(rank_K)<-NULL
  A_list <- purrr::map(reff_list, ~.$A)
  names(A_list)<-NULL
  e_list <- purrr::map(reff_list, ~.$e)
  names(e_list)<-NULL

  for(i in 1:length(Z_list)){
    Z_list[[i]] <- rbind(Z_list[[i]], Z_list[[i]][y_s>0,])
  }


  w_list_orig <- vector("list", n_aux)
  m_list_orig <- vector("list", n_aux)
  v_list_orig <- vector("list", n_aux)
  for(k in 1:n){
    w_list_orig[[k]] <- SamplerPoisson::original_mixtures[[1]]$w
    m_list_orig[[k]] <- SamplerPoisson::original_mixtures[[1]]$m
    v_list_orig[[k]] <- SamplerPoisson::original_mixtures[[1]]$v
  }
  for(k in (n + 1):n_aux){
    index <- y_s[n_0 + k - n]
    if(index > 30000){
      index<-30000
    }
    w_list_orig[[k]] <- SamplerPoisson::original_mixtures[[index]]$w
    m_list_orig[[k]] <- SamplerPoisson::original_mixtures[[index]]$m
    v_list_orig[[k]] <- SamplerPoisson::original_mixtures[[index]]$v
  }

  ################################
  w_list_adj <- vector("list", n_aux)
  m_list_adj <- vector("list", n_aux)
  v_list_adj <- vector("list", n_aux)
  for(k in 1:n){
    w_list_adj[[k]] <- SamplerPoisson::adjusted_mixtures[[1]]$w
    m_list_adj[[k]] <- SamplerPoisson::adjusted_mixtures[[1]]$m
    v_list_adj[[k]] <- SamplerPoisson::adjusted_mixtures[[1]]$v
  }
  for(k in (n + 1):n_aux){
    index <- y_s[n_0 + k - n]
    if(index > 30000){
      index<-30000
    }
    w_list_adj[[k]] <- SamplerPoisson::adjusted_mixtures[[index]]$w
    m_list_adj[[k]] <- SamplerPoisson::adjusted_mixtures[[index]]$m
    v_list_adj[[k]] <- SamplerPoisson::adjusted_mixtures[[index]]$v
  }

  S_beta <- (1 / prior_precision_beta) * diag(ncol(X_mix))

  ## inits
  g_init_list = purrr::map2(rep(0, length(Z_list)), sapply(Z_list, ncol), rep)

  check_mix <- c(rep(SamplerPoisson::eps_max[1], n), SamplerPoisson::eps_max[y_s[y_s>0]])
  ind_check <- 1000

  start_time <- lubridate::now()

  if(alg == "MH-IAMS"){
    out <- sample_poissonm_nob(y = y_s, n0 = n_0,
                               MH = 1, a_pri = a_pri_s2g, b_pri = b_pri_s2g,
                               X = X_mix, Z_list = Z_list, K_list = K_list,
                               m_mix_orig = m_list_orig, v_mix_orig = v_list_orig, w_mix_orig = w_list_orig,
                               m_mix_adj = m_list_adj, v_mix_adj = v_list_adj, w_mix_adj = w_list_adj,
                               check_mix = check_mix,  b_check = T2, adjust = 0, b_gibbs_start = T1,
                               log_offset = log(E_mix), rank_K_g = rank_K,
                               A_list = A_list, e_list = e_list, S_beta = S_beta,
                               beta_init = beta_init, g_init_list = g_init_list,
                               S2g_init = s2g_init,
                               niter = n_iter, pr = pr, thin = thin)
  }

  if(alg == "RIAMS"){
    out <- sample_poissonm_nob(y = y_s, n0 = n_0,
                               MH = 1, a_pri = a_pri_s2g, b_pri = b_pri_s2g,
                               X = X_mix, Z_list = Z_list, K_list = K_list,
                               m_mix_orig = m_list_orig, v_mix_orig = v_list_orig, w_mix_orig = w_list_orig,
                               m_mix_adj = m_list_adj, v_mix_adj = v_list_adj, w_mix_adj = w_list_adj,
                               check_mix = check_mix, b_check = T2, adjust = 1, b_gibbs_start=T1,
                               log_offset = log(E_mix), rank_K_g = rank_K,
                               A_list = A_list, e_list = e_list, S_beta = S_beta,
                               beta_init = beta_init, g_init_list = g_init_list,
                               S2g_init = s2g_init,
                               niter = n_iter, pr = pr, thin = thin)
  }
  if(alg == "IAMS"){
    out <- sample_poissonm_nob(y = y_s, n0 = n_0,
                               MH = 0, a_pri = a_pri_s2g, b_pri = b_pri_s2g,
                               X = X_mix, Z_list = Z_list, K_list = K_list,
                               m_mix_orig = m_list_orig, v_mix_orig = v_list_orig, w_mix_orig = w_list_orig,
                               m_mix_adj = m_list_adj, v_mix_adj = v_list_adj, w_mix_adj = w_list_adj,
                               check_mix = check_mix,  b_check = T2, adjust = 0, b_gibbs_start=T1,
                               log_offset = log(E_mix), rank_K_g = rank_K,
                               A_list = A_list, e_list = e_list, S_beta = S_beta,
                               beta_init = beta_init, g_init_list = g_init_list,
                               S2g_init = s2g_init,
                               niter = n_iter, pr = pr, thin = thin)
  }

  if(alg == "Automatic"){
    check_mix_max <- c(rep(SamplerPoisson::eps_max[1], n), SamplerPoisson::eps_max[y_s[y_s>0]])
    check_mix_min <- c(rep(SamplerPoisson::eps_min[1], n), SamplerPoisson::eps_min[y_s[y_s>0]])

    out <- sample_poissonm_nob_auto(
      y = y_s, n0 = n_0, a_pri = a_pri_s2g, b_pri = b_pri_s2g,
      X = X_mix, Z_list = Z_list, K_list = K_list,
      m_mix_orig = m_list_orig, v_mix_orig = v_list_orig, w_mix_orig = w_list_orig,
      m_mix_adj = m_list_adj, v_mix_adj = v_list_adj, w_mix_adj = w_list_adj,
      check_mix_max = check_mix_max, check_mix_min = check_mix_min,
      b_gibbs_start = T1, b_check = T2, threshold_MH = pL, threshold_adj = pU,
      log_offset = log(E_mix), rank_K_g = rank_K,
      A_list = A_list, e_list = e_list, S_beta = S_beta,
      beta_init = beta_init, g_init_list = g_init_list,
      S2g_init = s2g_init, niter = n_iter, pr = pr, thin = thin)
  }
  end_time <- lubridate::now()
  elapsed_time <- end_time - start_time

  out_mcmc <- out[c("beta", "gamma", "s2g")]
  if(is.null(colnames(X))){
    colnames(out_mcmc$beta) <- paste0("beta_", 0:(ncol(X)-1))}
  if(!is.null(colnames(X))){
    colnames(out_mcmc$beta) <- colnames(X)}

  # random effects
  if(is.null(names(Z_list))){
    names(Z_list) <- paste0("nu_",1:length(Z_list))
  }
  names(out_mcmc$gamma) <- names(Z_list)

  # reff's scalers
  if(length(Z_list) == 1){
    out_mcmc$s2g <- matrix(out_mcmc$s2g, ncol=1)
  }
  colnames(out_mcmc$s2g) <- paste0("s2_", names(Z_list))



  out_def <- list()
  out_def[["beta"]] <- out_mcmc[["beta"]]
  for(k in 1:length(Z_list)){
    out_def[[names(Z_list)[k]]] <- out_mcmc$gamma[[names(Z_list)[k]]]
    colnames(out_def[[names(Z_list)[k]]]) <- paste0(names(Z_list)[k], "_", 1:ncol(Z_list[[k]]))
  }
  out_def[["s2"]] <- out_mcmc[["s2g"]]


  out_def <- purrr::map(out_def, dplyr::as_tibble)


  adjusted <- NULL
  if(alg %in% c("RIAMS")){
    diff_comp <- purrr::map_dbl(w_list_orig, length)-purrr::map_dbl(out$w_mix, length)
    adjusted <- ifelse(diff_comp!=0, 1, 0)
  }




  output<-list(algorithm = algorithm,
               out = out_def,
               elapsed_time = elapsed_time)

  if(alg %in% c("MH-IAMS", "RIAMS", "Automatic")){
    output$accept_beta <- mean(out$accept_beta)
    output$accept_gamma <- out$n_acc_s2g / (n_iter / thin)
  }
  if(alg %in% c("Automatic")){
    output$kU <- out[["is_adjusted"]]/T2
  }
  if(alg %in% c("Automatic")){
    output$kL <- out[["is_MH"]]/T2
    output$redid_aux <- out$resid_aux
    if(all(output$kU<pU) & all(output$kL<pL)){
      output$sampler <- "IAMS"
      cat("-- IAMS algorithm used --")
    }else if (all(output$kU<pU) & any(output$kL>pL)){
      output$sampler <- "MH-IAMS"
      cat("-- MH-IAMS algorithm used --")
    }else{
      output$sampler <- "RIAMS"
      cat("-- RIAMS algorithm used --")
    }
  }



  return(
    output
  )

}
