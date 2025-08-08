### **Project Plan: A Structural Model of Remote Work, Sorting, and Wages**

#### **Phase 0: Foundational Theory and Model Specification**
*The goal of this phase is to finalize the model on paper before intensive coding. This ensures the theoretical foundations are solid.*

- [x] **Task 0.1: Finalize and Document Functional Forms.** ✅ 2025-08-06
    - [x] Formally adopt a specific functional form for g(h, ψ) (e.g., Cobb-Douglas: ψ₀ h^φ ψ^ν) as it offers the best properties for empirical identification. ✅ 2025-08-06
    - [x] Create a "Model Specification" document that lists all key functions: ✅ 2025-08-06
        - [x] **Utility:** u(w, α) = w - c₀ * (1-α)^(1+χ) / (1+χ) ✅ 2025-08-06
        - [x] **Production:** Y(α | h, ψ) = A₁h * [(1-α) + α * g(h, ψ)] ✅ 2025-08-06
        - [x] **Remote Productivity:** g(h, ψ) = ψ₀ * h^φ * ψ^ν ✅ 2025-08-06
        - [x] **Matching:** M(L, V) = γ₀ * L^γ₁ * V^(1-γ₁) ✅ 2025-08-06
        - [x] **Vacancy Costs:** c(v) = κ₀ * v^(1+κ₁) / (1+κ₁) ✅ 2025-08-06


- [x] **Task 0.2: Derive All Key Theoretical Objects.** ✅ 2025-08-06
    - [x] Create a clean, self-contained mathematical appendix deriving the model's key equations based on the finalized functional forms. ✅ 2025-08-06
    - [x] The appendix must include derivations for: ✅ 2025-08-06
        - [x] The optimal remote share α*(h, ψ). ✅ 2025-08-06
        - [x] The lower threshold underline{ψ}(h). ✅ 2025-08-06
        - [x] The upper threshold overline{ψ}(h). ✅ 2025-08-06
        - [x] The equilibrium wage w*(h, ψ), including the compensating differential. ✅ 2025-08-06
        - [x] The match surplus S(h, ψ). ✅ 2025-08-06
        - [x] The equilibrium market tightness condition θ. ✅ 2025-08-06
    - **Status:** Completed; full derivations are available in the Appendix of `manuscript/0_index.pdf`.

- [x] **Task 0.3: Analyze and Document Theoretical Properties.**
    - [x] Analyze the derivatives of the threshold functions to understand the model's predicted sorting patterns. ✅ 2025-08-06
    - [x] Show that overline{ψ}(h) is always decreasing for φ > 0. ✅ 2025-08-06
    - [x] Show that underline{ψ}(h) is U-shaped, and derive the critical value h_crit where its slope changes. ✅ 2025-08-06
    - [x] Derive the formula for the size of the hybrid region, Δψ(h). ✅ 2025-08-06
    - **Status:** Completed; analytic proofs and plots are in the Appendix of `manuscript/0_index.pdf`.

---

#### **Phase 1: Implementation and Baseline Calibration**
*The goal of this phase is to translate the theory into a working, solvable model and establish a plausible baseline economy.*

- [x] **Task 1.1: Code the Model Structure and Equilibrium Solver.**
    - [x] Create data structures for Primitives and Results. ✅ 2025-08-06
    - [x] Write functions that compute the policy functions (α*, w*, S, etc.) for a given set of parameters and grids. ✅ 2025-08-06
    - [x] Implement the main equilibrium loop that solves for the market tightness θ that satisfies the aggregate vacancy condition. ✅ 2025-08-06
    -   **Status:** Completed and validated in `Types.jl` and `ModelSolver.jl`.
    -   **Next steps:** Add unit tests for edge cases in `tests/model_solver_tests.jl`.

- [x] **Task 1.2: Implement the Theory-Driven Calibration Routine.**
    - [x] Code the flexible calibrate_model! function. ✅ 2025-08-06
    - [x] Ensure it calculates φ based on the slope condition (h_min >= h_crit). ✅ 2025-08-06
    - [x] Ensure it calculates ψ₀ by forcing the underline{ψ}(h) curve to pass through a specified target point (e.g., (h_min, ψ_max)). ✅ 2025-08-06
    -   **Status:** Calibration logic in `ModelRunner.jl` passes existing tests.
    -   **Next steps:** Write an integration test in `tests/calibration_tests.jl` to confirm ψ₀ and φ targets.

- [ ] **Task 1.3: Generate and Analyze the Baseline Economy.**
    - [ ] Set all "off-the-shelf" parameters (β, δ, ξ, etc.) and exogenous choices (c₀, χ, κ₁).
    - [ ] Run calibrate_model! to determine φ and ψ₀.
    - [ ] Solve for the equilibrium θ and the corresponding κ₀ that targets a 5% unemployment rate.
    - [ ] Save the complete Primitives and Results objects for this benchmark.
    - **Next steps:** Configure `parameters/initial/benchmark.yaml`; implement κ₀ root-finding in `ModelRunner.jl`; serialize outputs to `results/benchmark/`.

- [ ] **Task 1.4: Create Key Baseline Visualizations.**
    - [ ] Plot the **Work Arrangement Regimes**, showing the (h, ψ) space partitioned into In-Person, Hybrid, and Remote regions.
    - [ ] Plot the **Equilibrium Employment Distribution n(h, ψ)** with its marginals.
    - [ ] Create the **overlaid version** of the regimes plot, greying out regions with negative surplus to visualize the selection effect.
    - **Next steps:** Build plot scripts in `src/structural/model/plots/` (regimes.jl, distribution.jl); integrate into `main.jl` and save outputs to `/figures/baseline`.

---

#### **Phase 2: Structural Estimation**
*The goal of this phase is to formally discipline the model with data, moving from calibration to estimation.*

- [ ] **Task 2.1: Data Preparation and Proxy Creation.**
    - [ ] Process microdata (e.g., CPS) to create empirical counterparts to the model's variables.
    - [ ] **Proxy for h:** Estimate a Mincer log-wage regression with extensive controls and extract residuals.
    - [ ] **Proxy for ψ:** Merge in an external occupation-level teleworkability index.
    - [ ] **Proxy for α:** Classify workers based on survey questions on remote work.

- [ ] **Task 2.2: Implement the Simulated Method of Moments (SMM) Estimator.**
    - [ ] **Calculate Data Moments:** Compute the vector of target moments from the prepared data (unemployment rate, work arrangement shares, wage Gini, skill premium, and the "marginal remote sorting" regression coefficient β₁_hat from log(ψ) ~ log(h)).
    - [ ] **Implement the SMM Loop:** Write the code to solve the model, simulate moments, and compute the distance for a given parameter guess.
    - [ ] **Run the Estimator:** Use a numerical optimizer to find the parameter vector that minimizes the distance.
    - [ ] **Compute Standard Errors:** Use bootstrapping or asymptotic formulas.

- [ ] **Task 2.3 (Advanced): Implement the Maximum Likelihood (MLE) Estimator.**
    - [ ] Assume a parametric distribution for h.
    - [ ] Write the likelihood function that handles point identification (hybrid) and set identification (corners).
    - [ ] Use an optimizer to find the parameters that maximize the log-likelihood.
    - [ ] Compare MLE and SMM parameter estimates for robustness.

---

#### **Phase 3: Results, Validation, and Counterfactuals**
*The goal of this phase is to use the estimated model to answer the research question and explore its implications.*

- [ ] **Task 3.1: Validate the Model by Replicating Stylized Facts.**
    - [ ] Generate a simulated dataset from  estimated model.
    - [ ] Run the same wage regressions on the simulated data as were run on the real data.
    - [ ] Show that the model-generated coefficients are close to the empirical ones.

- [ ] **Task 3.2: Decompose the "Within-Occupation" Remote Wage Premium.**
    - [ ] Implement the "slicing" visualization: for a fixed ψ, compare the average h and w of remote vs. in-person workers.
    - [ ] Use the visualization to demonstrate selection on unobserved skill.

- [ ] **Task 3.3: Design and Run Counterfactual Experiments.**
    - [ ] Define a **Technology Shock** experiment (e.g., increase ν or ψ₀).
    - [ ] Define a **Preference Shock** experiment (e.g., decrease c₀ or χ).

- [ ] **Task 3.4: Analyze and Visualize Counterfactual Results.**
    - [ ] Create plots showing how key outcomes change in response to the shocks:
        - [ ] Aggregate Unemployment and Market Tightness.
        - [ ] Work Arrangement Shares.
        - [ ] Sorting Pattern (ρ).
        - [ ] Wage Inequality (Gini, Skill Premium).

---

#### **Phase 4: Final Output and Dissemination**
*The goal of this phase is to communicate  findings effectively.*

- [ ] **Task 4.1: Structure and Write the Research Paper.**
    - [ ] Draft the paper with a clear structure (Introduction, Model, Estimation, Results, Counterfactuals, Conclusion).

- [ ] **Task 4.2: Create the Final Presentation Deck.**
    - [ ] Build a new set of slides based on the completed research.
    - [ ] Focus on building intuition using the key visualizations developed in earlier phases.


# Imputation Strategy 
### 1. The Core Objective

The goal was to create a high-resolution map of remote work in the U.S. To do this, needed to **impute a detailed work-from-home (WFH) share (α)** for every worker in a large, representative dataset like the American Community Survey (ACS), which contains detailed occupation codes.

The central challenge was that the two primary datasets had complementary strengths and weaknesses:
*   **The SWAA Dataset:** Contains a rich, detailed WFH share variable (wfhcovid_fracmat) but only has coarse occupational and industrial categories.
*   **The ACS Dataset:** Contains the detailed occupational codes need but lacks a detailed WFH share variable.

Our task was to intelligently fuse these two datasets.

### 2. Strategy

The plan was to use the variables common to both datasets (demographics, coarse occupation/industry) as a "bridge" to transfer the WFH information from SWAA to the ACS.

We outlined a process to:
1.  **Train a model** on the SWAA data to learn the relationship between worker characteristics and their WFH share.
2.  **Apply that model** to the ACS data to predict a WFH share for every worker.

### 3. The Refined Method: Three-Part Model

Improved upon the initial strategy by implementing a more powerful **three-part model**. This was a crucial enhancement because the distribution of WFH is not smooth; it's heavily clustered at 0% (fully in-person) and 100% (fully remote).

model brreaks the problem down into three stages:
1.  **The Hurdle Model:** Predicts the probability that a worker does *any* remote work (P(α > 0)).
2.  **The Top Corner Model:** For those who do some remote work, it predicts the conditional probability of being *fully remote* (P(α = 1 | α > 0)).
3.  **The Interior Model:** For those who are hybrid, it predicts their specific WFH share (E[α | 0 < α < 1]).

Used the predictions from these three models in a clever simulation to assign a final imputed WFH share (alpha_final) to every worker in the ACS, creating a realistic distribution with the appropriate number of fully in-person and fully remote workers.

### 4. Validation and the "Reality Check"

After creating the initial imputation,  performed a validation step.  compared the distribution of work arrangements in  imputed ACS data against two benchmarks:
1.  The original SWAA training data.
2.  A simple "yes/no" WFH question available within the ACS itself ("ACS Validated Data").

### 5. The Final Step: Calibration to an Authoritative Target

To resolve this discrepancy and create the most credible dataset possible, we designed a **calibration** strategy. This final step adjusts the imputed distribution to match known, authoritative totals.

1.  **Define the Target:**  provided a table from an external source (like the BLS) with the "true" national shares of Fully In-Person (78.4%), Hybrid (11.5%), and Fully Remote (10.1%) workers.
2.  **Create a Propensity Score:** We combined the outputs of  three models into a single expected_alpha score for each worker, ranking them from least to most likely to work from home.
3.  **Rank-and-Assign:** We used the percentile cutoffs of this score to assign each worker to one of the three categories, ensuring the final counts precisely matched the BLS targets. For workers assigned to the "Hybrid" category, we retained their unique, model-predicted WFH share.

---

### The Final Product

For every worker in the ACS,  now have a new variable (alpha_calibrated_bls) that represents their WFH share. This variable is exceptional because it is:

*   **Individually-Informed:** The value for each person is based on a sophisticated model that accounts for their specific demographic and work characteristics.
*   **Aggregately-Calibrated:** The overall distribution of work arrangements in  dataset now perfectly matches a reliable, real-world benchmark, making  subsequent analyses highly defensible.


---

### **Strategy 1: Simulated Method of Moments (SMM)**

**The Core Idea:** Find the set of structural parameters that makes  model's aggregate outcomes (the "moments") match the same aggregate statistics calculated from real-world data. This method focuses on getting the "big picture" right.

**How it Works:**

1.  **Calculate Moments from Data:**  first process  real-world data to calculate a vector of key statistics. These are the targets. Examples include:
    *   The share of workers in Full-Remote, Hybrid, and In-Person arrangements.
    *   The average remote share (α) for hybrid workers.
    *   The overall wage Gini coefficient.
    *   The "Within-Occupation Remote Premium" (the coefficient from a wage regression with occupation fixed effects).
    *   The "Marginal Remote Sorting" coefficient (from the log(ψ) ~ log(h) regression on marginal remote workers).

2.  **Simulate Moments from the Model:**  computational loop that:
    a.  Takes a **guess** for the structural parameters (φ, ν, χ, ψ₀, κ₀, etc.).
    b.  **Solves the full general equilibrium** of the model for that parameter guess.
    c.  Uses the model's equilibrium employment and wage distributions to **calculate the same vector of moments** as did for the data.

3.  **Minimize the Distance:** The estimation routine uses a numerical optimizer to find the set of parameters that minimizes the (weighted) squared distance between the data moments and the model's simulated moments.

**Key Features:**
*   **What it Matches:** Aggregate statistics.
*   **Handling of Unobserved Skill (h):** It does not require an assumption about the distribution of h. It only requires that the *outcomes* generated by the distribution of h in the model match the aggregate outcomes in the data.
*   **Pros:** More robust to misspecification. If  assume the wrong distribution for h, SMM can still provide reasonable estimates as long as the model can match the key aggregate patterns.
*   **Cons:** Less statistically efficient, as it doesn't use all the information in the microdata (only the summary moments). It can also be computationally very intensive.

---

### **Strategy 2: Maximum Likelihood Estimation (MLE)**

**The Core Idea:** Find the set of structural parameters that makes the observed individual choices in  dataset the most likely to have occurred. This method focuses on getting the micro-level decisions right for every person.

**How it Works:**

1.  **Assume a Distribution for Unobserved Skill (h):** This is the critical first step.  must assume that h is drawn from a parametric distribution in the population (e.g., a log-normal distribution with mean μ_h and variance σ_h²). The parameters μ_h and σ_h become part of the set of parameters  estimate.

2.  **Construct the Likelihood Function:** For each worker i in  data,  use the model's structure to write down the probability of observing their specific work arrangement α_i, given their occupation ψ_i and the structural parameters.
    *   **For a Hybrid Worker (0 < α_i < 1):**  invert the model's FOC to find the **unique skill level h_i*** that would rationalize their choice. The likelihood is the probability density of that skill level, f(h_i*).
    *   **For a Corner-Solution Worker (α_i = 0 or α_i = 1):**  use the model's threshold formulas to find the **set of possible skills** consistent with their choice. The likelihood is the total probability mass of h falling into that set, calculated from the CDF of the assumed skill distribution.

3.  **Maximize the Likelihood:** The estimation routine uses a numerical optimizer to find the set of parameters that maximizes the total log-likelihood (the sum of the log-likelihoods for every worker in the sample).

**Key Features:**
*   **What it Matches:** Individual-level choices.
*   **Handling of Unobserved Skill (h):** Requires a specific parametric assumption for the distribution of h. It then estimates the parameters of this distribution.
*   **Pros:** More statistically efficient, as it uses all available microdata. It allows  to estimate the distribution of unobserved heterogeneity itself, which is a powerful result.
*   **Cons:** Less robust. The estimates are conditional on  distributional assumption for h being correct. If the true distribution is very different, the parameter estimates can be biased.

---

### Phase 2.1: Additional Counterfactual Experiments

- Experiment 2: Vary Worker Bargaining Power (ξ)  
  • Hold all other parameters constant; sweep ξ ∈ [0.2, 0.8]; recalibrate κ₀ to preserve 5% unemployment.  
  • Analyze impact on sorting (ρ), wage inequality, and regime shares.

- Experiment 3: Preference Shock (c₀, χ)  
  • Increase or decrease the in-office cost scale c₀ or curvature χ to mimic changing remote‐work preferences.  
  • Recalibrate κ₀; examine shifts in α* regions and wage‐arrangement premia.

- Experiment 4: Matching Frictions (γ₁)  
  • Switch matching elasticity γ₁ ∈ [0.3, 0.7]; recalibrate ξ via Hosios; recalibrate κ₀.  
  • Test sensitivity of market tightness θ and vacancy creation to matching technology.

- Experiment 5: Skill Distribution Shifts  
  • Replace F_h(h) with alternative distributions (e.g., more mass at high‐h).  
  • Keep primitives fixed; study changes in equilibrium sorting, unemployment, and wages.

- Experiment 6: Occupation‐Specific Technology Shocks  
  • Introduce a sectoral shock to ψ (e.g., raise ψ for a subset of firms)  
  • Observe heterogeneous effects across sectors on remote‐work adoption and wages.

---

#### **Discussion Agenda**

- [ ] Discuss progress and plan. #todiscuss  #rasmus
- [ ] Discuss the problem of internal parameters as conditions of the theoretical model. #todiscuss  #rasmus
- [ ] Discuss the calibration. #todiscuss  #rasmus
- [ ] Discuss the imputation strategy (in particular the moment matching with CPS actual and detailed that can be exploited). #todiscuss  #rasmus