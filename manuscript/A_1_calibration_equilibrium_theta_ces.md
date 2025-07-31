### Market Tightness Equilibrium Derivation {#sec-appendix-ces}
**Optimal Vacancy Posting**

The firm's FOC is independent of the matching function's form. Solving for $v(\psi)$ yields:
$$
v(\psi) = \left( \frac{B(\psi)}{\kappa_{0}} q \right)^{1/\kappa_{1}}
$$
Now, we substitute the CES job-filling rate, $q(\theta) = (\theta^{\gamma} + 1)^{-1/\gamma}$:
$$
v(\psi) = \left( \frac{B(\psi)}{\kappa_{0}} (\theta^{\gamma} + 1)^{-1/\gamma} \right)^{1/\kappa_{1}}
$$

**Equilibrium Market Tightness**

The aggregate number of vacancies, $V$, is the integral of individual firms' optimal vacancies:
$$
V = \int v(\psi) dF_{\psi}(\psi) = \int \left( \frac{B(\psi)}{\kappa_{0}} (\theta^{\gamma} + 1)^{-1/\gamma} \right)^{1/\kappa_{1}} dF_{\psi}(\psi)
$$
Factoring out the terms dependent on $\theta$:
$$
V = \left( (\theta^{\gamma} + 1)^{-1/\gamma} \right)^{1/\kappa_1} \int \left( \frac{B(\psi)}{\kappa_{0}} \right)^{1/\kappa_{1}} dF_{\psi}(\psi)
$$
$$
V = (\theta^{\gamma} + 1)^{-1/(\gamma\kappa_1)} \int \left( \frac{B(\psi)}{\kappa_{0}} \right)^{1/\kappa_{1}} dF_{\psi}(\psi)
$$
We impose the equilibrium condition $V = \theta L$:
$$
\theta L = (\theta^{\gamma} + 1)^{-1/(\gamma\kappa_1)} \int \left( \frac{B(\psi)}{\kappa_{0}} \right)^{1/\kappa_{1}} dF_{\psi}(\psi)
$$
To solve for $\theta$, we group all terms involving $\theta$ on one side:
$$
\theta (\theta^{\gamma} + 1)^{1/(\gamma\kappa_1)} = \frac{1}{L} \int \left( \frac{B(\psi)}{\kappa_{0}} \right)^{1/\kappa_{1}} dF_{\psi}(\psi)
$$
This final equation implicitly defines the equilibrium market tightness $\theta$. Unlike the Cobb-Douglas case, we cannot isolate $\theta$ to find a clean, closed-form solution. The equilibrium $\theta$ is the value that solves this equation, which must be found numerically.