# **Imputing Work-From-Home (WFH) Share from SWAA to CPS Data**

**Objective:** To create a new CPS dataset where each employed individual has an imputed WFH share (`alpha`), predicted based on their characteristics using a model trained on the richer SWAA dataset.

**Methodology:**

- **Data Preparation:** Python (`pandas`)
- **Econometric Modeling:** Stata (`fracreg`)

---

## 0: Setup & Prerequisites

This initial phase ensures your environment is ready and all necessary assets are in place before starting the project.

- **1. Create a Project Directory:**
  - Create a single folder on your computer to hold all scripts, raw data, and outputs. This keeps the project organized.
  - Example structure:

    ```bash
        /project_root/
        |-- data/
        |   |-- raw/
        |   |   |-- swaa
        |   |   |   |-- WFHdata_MonthYY.csv
        |   |   |-- acs
        |   |   |   |-- acs_data.csv.gz
        |   |-- processed/
        |-- src/
        |   |-- empirical/
        |   |   |-- imputing_wfh_swaa_to_cps
        |   |       |-- 1_prepare_data.py
        |   |       |-- 2_run_model.do    
    ```

- **2. Acquire All Necessary Files:**
  - **SWAA Data:** Download `WFHdata_MonthYY.csv` from wfhresearch.com and place it in `data/raw/swaa/`.
  - **CPS Data:** Download IPUMS ACS data extract (e.g., `acs_data.csv.gz`) and place it in `data/raw/acs/`. Ensure it contains the necessary variables: 
    - `age`, `sex`, `educ`, `race`
    - `indnaics`, `occsoc`
  - **Occupation Crosswalk:** 

---

### **Phase 1: SWAA Data Preparation (Python)**

**Goal:** To create a clean, filtered "training" dataset from the raw SWAA data.

- **1.1. Load Raw SWAA Data:**
  - In your Python script (`1_prepare_data.py`), load `WFHdata_June25.csv` into a pandas DataFrame.

- **1.2. Filter the Sample to the Target Population:**
  - **Time Window:** Filter the data to a stable, recent period. A 12-month window is recommended (e.g., `date >= 202407`).
  - **Data Quality:** Remove any observations flagged as low quality (`ilowquality != 1`).
  - **Employment Status:** Keep only currently employed individuals by filtering on `workstatus_current_new` for values `1` ("Working for pay") and `2` ("Employed and paid, but not working").

- **1.3. Create the Dependent Variable:**
  - Create a new column named `wfh_share`.
  - Calculate its value by dividing the `wfhcovid_fracmat` variable by 100.

- **1.4. Select and Finalize Columns for Modeling:**
  - Create a list of all variables required for the Stata model: `wfh_share`, `cratio100` (the weight), and all predictor variables (`occupation_clean`, `work_industry`, `education_s`, `agebin`, `female`, `race_ethnicity_s`, `censusdivision`, `haschildren`).
  - Create a new DataFrame containing only these columns.
  - Drop any rows that have missing values (`NaN`) in any of these selected columns.

- **1.5. Export the Prepared SWAA Data:**
  - Save the final, cleaned DataFrame to a new CSV file named `swaa_prepared_for_stata.csv` inside the `data_processed/` folder.
  - Use the option `index=False` when saving.

---

### **Phase 2: CPS Data Preparation & Harmonization (Python)**

**Goal:** To create a "prediction" dataset from the raw CPS data with variables that perfectly match the structure of the SWAA data.

- **2.1. Load Raw CPS Data and Crosswalks:**
  - In the same Python script, load your IPUMS CPS data (`your_ipums_cps_data.csv`).
  - Load the occupation crosswalk (`occtooccsoc18.csv`).

- **2.2. Prepare for Harmonization:**
  - Merge the occupation crosswalk with your main CPS DataFrame using the `occ` variable as the key. This adds the `soc2018` column to your CPS data, which is needed for harmonization.
  - Filter the CPS data for employed individuals, using a consistent definition as in Step 1.2.

- **2.3. Harmonize Each "Bridge" Variable:**
  - For each predictor variable from the SWAA dataset, create a corresponding column in the CPS DataFrame with the exact same name and categorical structure.
    - **`work_industry` (Target):**
      - **Source:** `indnaics` from CPS.
      - **Logic:** Write a function that maps the numeric `indnaics` codes to the 18 SWAA industry categories based on NAICS prefix (e.g., codes starting with '11' map to category 1 'Agriculture').
    - **`occupation_clean` (Target):**
      - **Source:** `soc2018` from the merged crosswalk.
      - **Logic:** Write a function that maps the `soc2018` codes to the 12 SWAA occupation categories based on SOC prefix (e.g., codes starting with '11-' or '13-' map to category 5 'Management, Business and Financial').
    - **`agebin` (Target):**
      - **Source:** `age` from CPS.
      - **Logic:** Use `pd.cut()` to slice the continuous `age` variable into bins that match SWAA's `agebin` categories (e.g., 20-29, 30-39, etc.).
    - **`education_s` (Target):**
      - **Source:** `educ` from IPUMS.
      - **Logic:** Map the IPUMS `educ` codes to the 5 simplified SWAA education categories.
    - **`female` (Target):**
      - **Source:** `sex` from CPS.
      - **Logic:** Create a binary indicator where `female = 1` if `sex == 2`, else `0`.
    - **`race_ethnicity_s` (Target):**
      - **Sources:** `race` and `hispan` from CPS.
      - **Logic:** Create the 4-category simplified race/ethnicity variable, ensuring Hispanic is prioritized.
    - **`censusdivision` & `haschildren` (Targets):**
      - **Sources:** Relevant geography and household variables from CPS.
      - **Logic:** Map to the corresponding SWAA categories.

- **2.4. Select and Finalize Columns for Prediction:**
  - Create a new DataFrame containing only the newly harmonized predictor columns. The names must exactly match those in `swaa_prepared_for_stata.csv`.
  - Drop any rows with missing values that may have resulted from the harmonization process.

- **2.5. Export the Prepared CPS Data:**
  - Save the final, harmonized DataFrame to a new CSV file named `cps_prepared_for_stata.csv` inside the `data_processed/` folder.

---

### **Phase 3: Econometric Modeling & Imputation (Stata)**

**Goal:** To use the prepared datasets to estimate the model and generate the final imputed values.

- **3.1. Import Prepared SWAA Data:**
  - In your Stata `.do` script (`2_run_model.do`), use the `import delimited` command to load `swaa_prepared_for_stata.csv`.

- **3.2. Estimate the Fractional Logit Model:**
  - Use the `fracreg logit` command to regress `wfh_share` on all the harmonized predictor variables (`i.occupation_clean`, `i.work_industry`, etc.).
  - Crucially, apply the population weights using the `[pweight=cratio100]` option.
  - Immediately after estimation, store the model results in memory using `estimates store wfh_model`.

- **3.3. Import Harmonized CPS Data:**
  - `clear` the memory and use `import delimited` to load `cps_prepared_for_stata.csv`.

- **3.4. Generate Predictions:**
  - Restore the model you just trained using `estimates restore wfh_model`.
  - Use the `predict alpha, mu` command. This applies the coefficients from the SWAA model to the CPS data, creating a new variable `alpha` containing the predicted WFH share for each individual.

- **3.5. Save the Final Imputed Dataset:**
  - Use `compress` to optimize the file size.
  - Save the final CPS DataFrame, which now includes the `alpha` column, as a Stata data file (`.dta`) named `cps_with_imputed_wfh.dta` in the `output/` folder.

---

### **Phase 4: Validation and Sanity Checks**

**Goal:** To ensure the imputed values are reasonable and the model behaved as expected.

- **4.1. Summarize the Predictions:**
  - In Stata, after creating `alpha`, run `summarize alpha, detail`. Check that the minimum value is >= 0 and the maximum is <= 1. Review the mean and median to see if they are plausible.

- **4.2. Compare Group Averages:**
  - This is the most important check. In Stata, calculate the average imputed `alpha` for each coarse occupation category in your final CPS dataset (`table occupation_clean, contents(mean alpha)`).
  - Separately, go back to the prepared SWAA data and calculate the *actual* weighted average of `wfh_share` for each `occupation_clean` category.
  - Compare the two sets of averages. They should be very close, confirming your model is correctly capturing the main patterns. Repeat for industry and education.

- **4.3. Spot-Check Individual Predictions:**
  - In the final CPS dataset, look at a few specific detailed occupations. For example, find a "Software Developer" and a "Roofer". The software developer should have a high `alpha` (e.g., >0.6), while the roofer should have an `alpha` very close to 0. This confirms the model's face validity.
