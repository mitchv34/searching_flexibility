### Market Tightness Equilibrium Derivation {#sec-appendix-cobb-douglas}

We assume a standard Cobb-Douglas matching function, which exhibits constant returns to scale.
$$
M(L, V) = \gamma_{0} L^{\gamma_{1}} V^{1-\gamma_{1}}
$$
This function yields the following probabilities:
*   **Job-Filling Rate:** $q(\theta) = \gamma_{0} \theta^{-\gamma_{1}}$
*   **Job-Finding Rate:** $p(\theta) = \gamma_{0} \theta^{1-\gamma_{1}}$

**Optimal Vacancy Posting**

We begin with the firm's FOC and solve for the optimal number of vacancies $v(\psi)$:
$$
q B(\psi) = \kappa_{0} v(\psi)^{\kappa_{1}} \implies v(\psi) = \left( \frac{B(\psi)}{\kappa_{0}} q \right)^{1/\kappa_{1}}
$$
Substituting the Cobb-Douglas job-filling rate, $q(\theta) = \gamma_{0} \theta^{-\gamma_{1}}$, gives the expression for $v(\psi)$ as a function of market tightness $\theta$ and the firm's expected benefit $B(\psi)$:
$$
v(\psi) = \left( \frac{B(\psi)}{\kappa_{0}} \frac{\gamma_{0}}{\theta^{\gamma_{1}}} \right)^{1/\kappa_{1}}
$$

**Equilibrium Market Tightness**

The aggregate number of vacancies, $V$, is found by integrating $v(\psi)$ across all firm types:
$$
V = \int v(\psi) dF_{\psi}(\psi) = \int \left( \frac{\gamma_0 B(\psi)}{\kappa_0 \theta^{\gamma_1}} \right)^{1/\kappa_1} dF_{\psi}(\psi)
$$
We can factor out the terms that do not depend on the firm type $\psi$:
$$
V = \left( \frac{\gamma_0}{\kappa_0 \theta^{\gamma_1}} \right)^{1/\kappa_1} \int [B(\psi)]^{1/\kappa_1} dF_{\psi}(\psi) = \frac{1}{\theta^{\gamma_1/\kappa_1}} \left( \frac{\gamma_0}{\kappa_0} \right)^{1/\kappa_1} \int [B(\psi)]^{1/\kappa_1} dF_{\psi}(\psi)
$$
To find the equilibrium, we substitute the definition of market tightness, $V = \theta L$, into the left-hand side:
$$
\theta L = \frac{1}{\theta^{\gamma_1/\kappa_1}} \left( \frac{\gamma_0}{\kappa_0} \right)^{1/\kappa_1} \int [B(\psi)]^{1/\kappa_1} dF_{\psi}(\psi)
$$
Now, we solve for $\theta$ by grouping all $\theta$ terms on the left:
$$
\theta^{1 + \gamma_1/\kappa_1} L = \left( \frac{\gamma_0}{\kappa_0} \right)^{1/\kappa_1} \int [B(\psi)]^{1/\kappa_1} dF_{\psi}(\psi)
$$
Finally, isolating $\theta$ yields the closed-form solution for equilibrium market tightness:
$$
\theta = \left( \frac{1}{L} \left( \frac{\gamma_0}{\kappa_0} \right)^{1/\kappa_1} \int [B(\psi)]^{1/\kappa_1} dF_{\psi}(\psi) \right)^{\frac{\kappa_1}{\kappa_1 + \gamma_1}}
$$
