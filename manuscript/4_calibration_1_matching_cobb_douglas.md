**Matching and Vacancy Costs:**

- We assume a standard Cobb-Douglas matching function, which exhibits constant returns to scale.
$$
M(L, V) = \gamma_{0} L^{\gamma_{1}} V^{1-\gamma_{1}}
$$ 
where $\gamma_0 > 0$ is matching efficiency and $\gamma_1 \in (0,1)$ is the elasticity of matches with respect to unemployment. 

#### Firm's Vacancy Posting Problem

A firm of type $\psi$ chooses the number of vacancies $v$ to post to maximize its expected profit. The profit function is the expected benefit minus the total cost:
$$
\Pi(v(\psi)) = q B(\psi)  v - c(v)
$$
where:
*   $B(\psi) = (1-\xi)\mathbb{E}_h[S(h,\psi)^{+}]$ is the firm's expected benefit *per filled vacancy*.
*  **Vacancy Cost Function:** The cost of posting vacancies is assumed to be a convex function, implying a rising marginal cost to posting.
    $$c(v) = \frac{\kappa_{0} v^{1+\kappa_{1}}}{1+\kappa_{1}}$$
    where $\kappa_{0} > 0$ is a cost scaling parameter and $\kappa_{1} > 0$ ensures convexity.

The firm's first-order condition (FOC) for profit maximization is $\Pi'(v) = 0$, which implies that the marginal benefit must equal the marginal cost:
$$
\Pi'(v) = 0 \quad \implies \quad q B(\psi) = c'(v) \quad \implies \quad q B(\psi) = \kappa_{0} v(\psi)^{\kappa_{1}}
$$
This condition allows us to derive an equilibrium condition for $\theta$. Details of this derivation are provided in @sec-appendix-cobb-douglas. 