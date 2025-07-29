sample_poisson <- function(y, X, n0, m_mix_orig, v_mix_orig, w_mix_orig, log_offset, S_beta, beta_init, niter, pr, thin) {
  .Call(`_SamplerPoisson_sample_poisson`, y, X, n0, m_mix_orig, v_mix_orig, w_mix_orig, log_offset, S_beta, beta_init, niter, pr, thin)
}
sample_poisson_MH <- function(y, X, n0, m_mix_orig, v_mix_orig, w_mix_orig, m_mix_adj, v_mix_adj, w_mix_adj, check_mix, b_gibbs_start, b_check, adjust, log_offset, S_beta, beta_init, niter, pr, thin) {
  .Call(`_SamplerPoisson_sample_poisson_MH`, y, X, n0, m_mix_orig, v_mix_orig, w_mix_orig, m_mix_adj, v_mix_adj, w_mix_adj, check_mix, b_gibbs_start, b_check, adjust, log_offset, S_beta, beta_init, niter, pr, thin)
}

sample_poisson_auto <- function(y, X, n0, m_mix_orig, v_mix_orig, w_mix_orig, m_mix_adj, v_mix_adj, w_mix_adj, check_mix_min, check_mix_max, b_gibbs_start, b_check, threshold_MH, threshold_adj, log_offset, S_beta, beta_init, niter, pr, thin) {
  .Call(`_SamplerPoisson_sample_poisson_auto`, y, X, n0, m_mix_orig, v_mix_orig, w_mix_orig, m_mix_adj, v_mix_adj, w_mix_adj, check_mix_min, check_mix_max, b_gibbs_start, b_check, threshold_MH, threshold_adj, log_offset, S_beta, beta_init, niter, pr, thin)
}


#' @title Draw posterior samples from a Bayesian Poisson regression model
#'
#' @description Allow the user to specify a Bayesian Poisson regression model and draw samples from the posterior distributions of parameters through different algorithms.
#'
#' @param y Vector of observed responses.
#' @param X Matrix with fixed effects. The intercept is added by the function.
#' @param offset Vector with possible offsets. Default all 1 values.
#' @param prior_precision_beta Precision of the Gaussian prior of the regression coefficients.
#' @param algorithm Choice of the algorithm: "Automatic", "IAMS", "MH-IAMS" and "RIAMS".
#' @param n_iter Number of MCMC iterations.
#' @param print Frequency of iterations for printing the progress.
#' @param thin Number of thinned observations.
#' @param T1 Number of iterations with IAMS algorithm for initializing the algorithm (used in "Automatic", "MH-IAMS" and "RIAMS")
#' @param T2 Number of iterations in which the need for mixture adjustment ("Automatic" and "RIAMS") and MH step ("Automatic") is verified.
#' @param pL Threshold for the proportion of problematic approximations to activate the MH step ("Automatic").
#' @param pU Threshold for the proportion of problematic approximations to activate the mixture adjustment ("Automatic").
#'
#'
#' @return
#'
#' The function returns a list with:
#'  * `algorithm`.
#'  * `out`: List with posterior samples of regression coefficients.
#'  * `elapsed_time`.
#'  * `accept_beta`: acceptance rate for the vector of regression coefficients (except for "IAMS" algorithm).
#'  * `kU`: for each residual, it contains the proportion of preliminary iterations falling in the upper tail ("RIAMS" and "Automatic" algorithms).
#'  * `kL`: for each residual, it contains the proportion of preliminary iterations falling in the lower tail ("Automatic" algorithm).
#'
#'
#'
#'
#' @export
#'


sample_PM <-  function(y, X, offset = rep(1, length(y)),
                       prior_precision_beta = 0.001,
                       algorithm = c("Automatic","IAMS", "MH-IAMS", "RIAMS"),
                       n_iter = 10000, print = n_iter / 10, thin = 1,
                       T1 = 500, T2 = 250,
                       pL = 0.05, pU = 0.05){

  alg <- match.arg(algorithm)

  # ricevuti in input y e X
  ord <- order(y)
  y_s <- y[ord]
  X_s <- X[ord, ]
  n <- length(y_s)
  n_0 <- sum(y_s == 0)
  n_aux <- 2 * n - n_0
  if(ncol(X)>1){
    X_mix <- rbind(X_s, X_s[y_s>0,])
  }else{
    X_mix <- matrix(c(X_s, X_s[y_s>0]), ncol=1)
  }
  E_mix <- offset[ord]
  E_mix <- c(E_mix, E_mix[y_s>0])

  w_list_orig <- vector("list", n_aux)
  m_list_orig <- vector("list", n_aux)
  v_list_orig <- vector("list", n_aux)
  for(k in 1:n){
    w_list_orig[[k]] <- SamplerPoisson::original_mixtures[[1]]$w
    m_list_orig[[k]] <- SamplerPoisson::original_mixtures[[1]]$m
    v_list_orig[[k]] <- SamplerPoisson::original_mixtures[[1]]$v
  }
  for(k in (n + 1):n_aux){
    w_list_orig[[k]] <- SamplerPoisson::original_mixtures[[y_s[n_0 + k - n]]]$w
    m_list_orig[[k]] <- SamplerPoisson::original_mixtures[[y_s[n_0 + k - n]]]$m
    v_list_orig[[k]] <- SamplerPoisson::original_mixtures[[y_s[n_0 + k - n]]]$v
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
    w_list_adj[[k]] <- SamplerPoisson::adjusted_mixtures[[y_s[n_0 + k - n]]]$w
    m_list_adj[[k]] <- SamplerPoisson::adjusted_mixtures[[y_s[n_0 + k - n]]]$m
    v_list_adj[[k]] <- SamplerPoisson::adjusted_mixtures[[y_s[n_0 + k - n]]]$v
  }

  S_beta <- (1 / prior_precision_beta) * diag(ncol(X_mix))

  niter <- n_iter


  start_time <- lubridate::now()
  if(alg == "IAMS"){
    out <- sample_poisson(
      y = y_s, X = X_mix, n0 = n_0,
      m_mix_orig = m_list_orig, v_mix_orig = v_list_orig, w_mix_orig = w_list_orig,
      log_offset = log(E_mix),
      S_beta = S_beta, beta_init = rep(0, ncol(X_mix)),
      niter = niter, pr = print, thin = thin)
  }

  if(alg == "MH-IAMS"){
    check_mix <- c(rep(SamplerPoisson::eps_max[1], n), SamplerPoisson::eps_max[y_s[y_s>0]])

    out <- sample_poisson_MH(
      y = y_s, X = X_mix, n0 = n_0, m_mix_orig = m_list_orig, v_mix_orig = v_list_orig, w_mix_orig = w_list_orig,
      m_mix_adj = m_list_adj, v_mix_adj = v_list_adj, w_mix_adj = w_list_adj, check_mix = check_mix,
      b_gibbs_start = T1, b_check = T2, adjust = 0, log_offset = log(E_mix),
      S_beta = S_beta, beta_init = rep(0, ncol(X_mix)),
      niter = niter, pr = print, thin = thin)
  }

  if(alg == "RIAMS"){
    check_mix <- c(rep(SamplerPoisson::eps_max[1], n), SamplerPoisson::eps_max[y_s[y_s>0]])

    out <- sample_poisson_MH(
      y = y_s, X = X_mix, n0 = n_0, m_mix_orig = m_list_orig, v_mix_orig = v_list_orig, w_mix_orig = w_list_orig,
      m_mix_adj = m_list_adj, v_mix_adj = v_list_adj, w_mix_adj = w_list_adj, check_mix = check_mix,
      b_gibbs_start = T1, b_check = T2, adjust = 1, log_offset = log(E_mix),
      S_beta = S_beta, beta_init = rep(0, ncol(X_mix)),
      niter = niter, pr = print, thin = thin)
  }

  if(alg == "Automatic"){

    check_mix_max <- c(rep(SamplerPoisson::eps_max[1], n), SamplerPoisson::eps_max[y_s[y_s>0]])
    check_mix_min <- c(rep(SamplerPoisson::eps_min[1], n), SamplerPoisson::eps_min[y_s[y_s>0]])

    out <- sample_poisson_auto(
      y = y_s, X = X_mix, n0 = n_0, m_mix_orig = m_list_orig, v_mix_orig = v_list_orig, w_mix_orig = w_list_orig,
      m_mix_adj = m_list_adj, v_mix_adj = v_list_adj, w_mix_adj = w_list_adj, check_mix_min = check_mix_min, check_mix_max = check_mix_max,
      b_gibbs_start = T1, b_check = T2, log_offset = log(E_mix),
      threshold_MH = pL, threshold_adj = pU,
      S_beta = S_beta, beta_init = rep(0, ncol(X_mix)),
      niter = niter, pr = print, thin = thin)
  }

  end_time <- lubridate::now()
  elapsed_time <- end_time - start_time

  out_mcmc <- out["beta"]
  if(is.null(colnames(X))){
    colnames(out_mcmc$beta) <- paste0("beta_", 0:(ncol(X)-1))}
  if(!is.null(colnames(X))){
    colnames(out_mcmc$beta) <- colnames(X)}
  out_mcmc <- purrr::map(out_mcmc, dplyr::as_tibble)


  accept_beta <- NA

  output<-list(algorithm = algorithm,
               out = out_mcmc,
               elapsed_time = elapsed_time)

  if(alg %in% c("MH-IAMS", "RIAMS", "Automatic")){
    output$accept_beta <- mean(out$accept_MH)
  }
  if(alg %in% c("Automatic", "RIAMS")){
    output$kU <- out[["is_adjusted"]]/T2
  }
  if(alg %in% c("Automatic")){
    output$kL <- out[["is_MH"]]/T2
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
