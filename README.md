# SamplerPoisson

**SamplerPoisson** provides tools to fit Bayesian Poisson models, including latent Gaussian models, using different versions of auxiliary mixture sampling schemes.

The package implements:

- The **improved auxiliary mixture sampler** from Frühwirth-Schnatter et al. (2009)  
- A **robust version** of this sampler, introduced in Gardini et al. (2025)

These methods enable efficient Bayesian inference in models with Poisson likelihoods and latent Gaussian structures, covering both standard Poisson regression and more complex hierarchical models.

---

## Installation

To install the package from GitHub:

```r
# install.packages("devtools")  # if not already installed
devtools::install_github("yourusername/SamplerPoisson")
