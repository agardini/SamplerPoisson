#' Pipe operator
#'
#' See \code{magrittr::\link[magrittr:pipe]{\%>\%}} for details.
#'
#' @name %>%
#' @rdname pipe
#' @keywords internal
#' @export
#' @importFrom magrittr %>%
#' @usage lhs \%>\% rhs
#' @param lhs A value or the magrittr placeholder.
#' @param rhs A function call using the magrittr semantics.
#' @return The result of calling `rhs(lhs)`.
NULL



#' Burn-in and thinning of posterior drowns
#'
#'@description Providing as input an object created by \code{sample_PM} and \code{sample_PMM}, the function performs burn-in and thinning of the chain.
#'
#'@param out Object `out` in the output created by \code{sample_PM} and \code{sample_PMM}.
#'@param n_burn Length burn-in period.
#'@param n_thin Length thinning interval.
#'
#'
#'
#' @export
burn_thin <- function(out, n_burn, n_thin = 1){
  n_it <- nrow(out[[1]])
  sel_thin <- seq((n_burn + 1), n_it, by = n_thin)
  purrr::map(out, dplyr::slice, sel_thin)
}

#'
#' Computation of posterior summaries
#'
#'@description Providing as input an object created by \code{sample_PM}, \code{sample_PMM} or \code{burn_thin}, the function computes posterior summaries.
#'
#'@param model_out Object created by \code{sample_PM}, \code{sample_PM} or \code{burn_thin}.
#'
#'
#' @export


posterior_summaries <- function(model_out){
  model_out %>%
    purrr::map(coda::as.mcmc) %>%
    purrr::map(MCMCvis::MCMCsummary, Rhat = FALSE) %>%
    purrr::map(dplyr::as_tibble, rownames = "parameter")
}

