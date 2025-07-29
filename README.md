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
devtools::install_github("agardini/SamplerPoisson")
```

## References

- Gardini, A., Greco, F., & Trivisano, C. (2025). A note on auxiliary mixture sampling for Bayesian Poisson models. _arXiv preprint_ url: https://arxiv.org/abs/2502.04938
- Frühwirth-Schnatter, S., Frühwirth, R., Held, L., & Rue, H. (2009). Improved auxiliary mixture sampling for hierarchical models of non-Gaussian data. _Statistics and Computing_, 19(4), 479.

