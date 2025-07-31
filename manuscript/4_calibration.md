### Quantitative Specification and Parameterization

To take the model to the data and analyze its quantitative implications, we must specify functional forms for the model's key components and assign numerical values to the resulting parameters. We divide the parameters into two groups: those set externally based on standard values in the literature, and those calibrated internally to match key moments of the U.S. labor market data.

#### Functional Forms

\textcolor{red}{NEED TO SETTLE ON ONE FUNCTIONAL FORM FOR THE $g(h, \psi)$ FUNCTION}

**Production and Cost Functions:**
We adopt specific functional forms for the production and worker cost functions to derive an analytical solution for the optimal work arrangement, $\alpha^*$.

*   **In-Office Cost Function:** The worker's disutility from in-office work is assumed to be a convex function of the time spent in the office, $(1-\alpha)$.
    $$c(1-\alpha) = c_0 \frac{(1-\alpha)^{1+\chi}}{1+\chi}$$
    The parameter $c_0 > 0$ scales the overall disutility, and the curvature parameter $\chi > 0$ ensures that the marginal disutility of in-office time is increasing, a standard feature that helps ensure well-behaved interior solutions for $\alpha$.

{{< include "4_calibration_cobb_douglas_g_function.md" >}}
{{< include "4_calibration_quasilinear_g_function.md" >}}

{{< include "4_calibration_1_matching_cobb_douglas.md" >}}
#### Calibration

We choose the parameter values to align the model's steady state with key features of the pre-pandemic U.S. labor market. The model is calibrated at a monthly frequency. The parameters are summarized in Table 1.

**Externally Set Parameters:**
Several parameters are set to standard values from the literature. The discount factor $\beta$ is set to 0.996, corresponding to a 5% annual real interest rate. The elasticity of the matching function, $\eta$, is set to 0.5, a common value in the search and matching literature (Petrongolo and Pissarides, 2001).

**Internally Calibrated Parameters:**
The remaining parameters are calibrated jointly to match a set of moments from the data. 

\textcolor{red}{EXAMPLE TEXT}
"The parameters of the production function ($A_1, \nu, \psi_0, \phi$) and the in-office cost function ($c_0, \chi$) are chosen to match the average labor productivity, the observed share of workers in remote/hybrid arrangements, and the estimated compensating wage differential for remote work from survey evidence (e.g., Barrero et al., 2023a). The vacancy cost parameters ($c_v, \gamma$) and matching efficiency ($M_0$) are disciplined by the average labor market tightness ($\theta$), the job-filling rate ($q$), and the elasticity of vacancies to productivity shocks..."


\textcolor{red}{EXAMPLE TABLE}

\begin{table}[h!]
\centering
\caption{Parameter Values}
\label{tab:parameters}
\begin{tabular}{@{}llcl@{}}
\toprule
\textbf{Parameter} & \textbf{Description} & \textbf{Value} & \textbf{Target/Source} \\
\midrule
\multicolumn{4}{l}{\textit{Preferences \& Technology}} \\
$\beta$ & Discount Factor & 0.996 & 5\% annual interest rate \\
$\chi$ & Curvature of in-office cost & 1.5 & Internally calibrated \\
$c_0$ & In-office cost scale & 0.25 & Match compensating differential \\
$A_1$ & Productivity scale & 1.0 & Normalization \\
$\phi$ & Skill-remote complementarity & 0.1 & Match skill premium \\
... & ... & ... & ... \\
\addlinespace % Adds a bit of vertical space for separation
\multicolumn{4}{l}{\textit{Search \& Matching}} \\
$\eta$ & Matching elasticity & 0.5 & Petrongolo \& Pissarides (2001) \\
$\delta$ & Exogenous separation rate & 0.025 & Match avg. job duration \\
$\xi$ & Worker bargaining power & 0.5 & Shimer (2005) \\
... & ... & ... & ... \\
\bottomrule
\end{tabular}
\end{table}