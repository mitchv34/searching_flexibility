# **Imputing Work-From-Home (WFH) Share from SWAA to ACS Data**

**Objective:** To create a new ACS dataset where each employed individual has an imputed WFH share (`alpha`), predicted based on their characteristics using a model trained on the richer SWAA dataset.

**Methodology:**

- **Data Preprocessing:** Polars-based ACS processing (see [`acs_processing_polar.md`](acs_processing_polar.md))
- **Data Preparation:** Python (`pandas`)
- **Econometric Modeling:** Stata (`fracreg`)

## **Project Structure**

The project follows this directory structure:

```
searching_flexibility/
├── data/
│   ├── raw/
│   │   ├── swaa/
│   │   │   └── WFHdata_March25.csv          # SWAA survey data
│   │   └── acs/
│   │       └── usa_00136.csv.gz             # IPUMS ACS data
│   ├── processed/
│   │   └── acs/
│   │       └── acs_136_processed_polar.csv  # Preprocessed ACS data
├── src/
│   └── empirical/
│       ├── acs_data_prep/
│       │   └── 01_wfh_acs_polar.py          # ACS preprocessing (Phase 0)
│       └── imputing_wfh_swaa_to_cps/
│           ├── 1_prepare_data.py            # Phase 1-2: Data preparation
│           ├── 2_run_model.do               # Phase 3: Stata modeling
│           ├── run_master.py                # Master coordination script
│           └── test.py                      # Testing script for functions
├── output/                                  # Final results
├── doc/
│   ├── acs_processing_polar.md             # ACS preprocessing documentation
│   └── imputing_wfh_swaa_to_cps.md         # This methodology document
└── requirements.txt                        # Python dependencies
```

## **Software Requirements**

- **Python 3.7+** with packages:
  - polars (>=0.20.0) - for ACS preprocessing
  - pandas (>=1.3.0) - for data harmonization
  - numpy (>=1.20.0)
- **Stata** (any recent version with `fracreg` command)

Install Python dependencies:
```bash
pip install -r requirements.txt
```

## **Complete Workflow Overview**

The WFH imputation methodology consists of four main phases:

**Phase 0:** ACS Data Preprocessing (Polars) → **[See `acs_processing_polar.md`](acs_processing_polar.md)**
**Phase 1:** SWAA Data Preparation (Python)
**Phase 2:** ACS-SWAA Harmonization (Python) 
**Phase 3:** Econometric Modeling & Imputation (Stata)

## **Usage Instructions**

### **Option 1: Complete Pipeline (Recommended)**

#### Step 1: ACS Preprocessing
```bash
# Preprocess raw ACS data using Polars (see acs_processing_polar.md for details)
python src/empirical/acs_data_prep/01_wfh_acs_polar.py --min_year 2019 --max_year 2023
```

#### Step 2: WFH Imputation Pipeline
```bash
# Run the complete WFH imputation methodology
python src/empirical/imputing_wfh_swaa_to_cps/run_master.py
```

### **Option 2: Individual Steps**

#### Step 1: ACS Preprocessing (if not done)
See [`acs_processing_polar.md`](acs_processing_polar.md) for detailed instructions.

#### Step 2: Data Preparation (Python)
```bash
python src/empirical/imputing_wfh_swaa_to_cps/1_prepare_data.py
```

#### Step 3: Econometric Modeling (Stata)
```stata
cd "v:\high_tech_ind\WFH\searching_flexibility"
do "src/empirical/imputing_wfh_swaa_to_cps/2_run_model.do"
```

## **Data File Configuration**

### **ACS Data Configuration**
For ACS preprocessing configuration, see [`acs_processing_polar.md`](acs_processing_polar.md).

### **WFH Imputation Configuration**
In `src/empirical/imputing_wfh_swaa_to_cps/1_prepare_data.py`, modify these global variables:

```python
# Global file name variables - modify these to use different data files
SWAA_DATA_FILE = "WFHdata_March25.csv"              # SWAA input file
ACS_PROCESSED_FILE = "acs_136_processed_polar.csv"   # Preprocessed ACS input file  
SWAA_OUTPUT_FILE = "swaa_prepared_for_stata.csv"    # SWAA output file
ACS_OUTPUT_FILE = "acs_prepared_for_stata.csv"      # ACS output file

# Global date filter variables - modify these to change the time window
DEFAULT_START_DATE = 202407  # Start date in YYYYMM format (July 2024)
DEFAULT_END_DATE = None      # End date in YYYYMM format (None = no end limit)
```

### **Expected Output Files**

#### Intermediate Files
- `data/processed/acs/acs_136_processed_polar.csv` - Preprocessed ACS data (from Phase 0)
- `data/processed/{SWAA_OUTPUT_FILE}` - Cleaned SWAA training data
- `data/processed/{ACS_OUTPUT_FILE}` - Harmonized ACS prediction data

#### Final Outputs
- `output/acs_with_imputed_wfh.dta` - **Main output**: ACS data with imputed WFH shares
- `output/acs_with_imputed_wfh.csv` - CSV version of main output
- `output/wfh_model_estimates.ster` - Saved Stata model estimates
- `output/wfh_imputation_log.log` - Detailed Stata log file

The key output variable is `alpha` in the final dataset, representing the predicted WFH share (0-1) for each individual.

---

## **Phase 0: ACS Data Preprocessing**

**Goal:** Create a high-quality, research-ready ACS dataset with enhanced variables.

**→ This phase is fully documented in [`acs_processing_polar.md`](acs_processing_polar.md)**

**Key benefits of the preprocessing:**
- **10-50x faster processing** using Polars
- **Pre-filtered working population** (35+ hours, wage > $5, employees only)
- **Enhanced occupation hierarchies** (detailed/broad/minor SOC codes)
- **Teleworkability measures** from multiple methodologies
- **Geographic enhancements** (CBSA codes, metro indicators)
- **WFH indicators** from transportation data

**Output:** `data/processed/acs/acs_136_processed_polar.csv`

---

## **Phase 1: SWAA Data Preparation (Python)**

**Goal:** Create a clean, filtered "training" dataset from the raw SWAA data.

**1.1. Load Raw SWAA Data:**
- Load the file specified in `SWAA_DATA_FILE` global variable (default: `WFHdata_March25.csv`)

**1.2. Filter the Sample to Target Population:**
- **Time Window:** Filter to stable period using `DEFAULT_START_DATE` and `DEFAULT_END_DATE`
- **Data Quality:** Remove low quality observations (`ilowquality != 1`)
- **Employment Status:** Keep employed individuals (`workstatus_current_new` in [1,2])

**1.3. Create the Dependent Variable:**
- Create `wfh_share` by dividing `wfhcovid_fracmat` by 100

**1.4. Select and Finalize Columns:**
- Keep modeling variables: `wfh_share`, `cratio100`, and predictors
- Drop rows with missing values

**1.5. Export Prepared SWAA Data:**
- Save to `data/processed/{SWAA_OUTPUT_FILE}`

---

## **Phase 2: ACS-SWAA Harmonization (Python)**

**Goal:** Create harmonized prediction dataset from preprocessed ACS data.

**2.1. Load Preprocessed ACS Data:**
- Load from `data/processed/acs/acs_136_processed_polar.csv`
- **Advantage:** Data is already filtered and enhanced from Phase 0

**2.2. Harmonize Variables to Match SWAA Structure:**
- **`work_industry`:** Map `INDNAICS` to 18 SWAA industry categories
- **`occupation_clean`:** Map `OCCSOC_detailed` to 12 SWAA occupation categories  
- **`agebin`:** Convert `AGE` to SWAA age bins
- **`education_s`:** Map `EDUC` to 5 SWAA education categories
- **`female`:** Convert `SEX` to binary indicator
- **`race_ethnicity_s`:** Combine `RACE`/`HISPAN` to 4 SWAA categories
- **`censusdivision`:** Map `CBSA20` to census divisions (if available)

**2.3. Create Unique Identifier:**
- Generate `unique_person_id` from `SAMPLE`, `SERIAL`, `PERNUM` for merging back to original data

**2.4. Export Harmonized ACS Data:**
- Save to `data/processed/{ACS_OUTPUT_FILE}`

---

## **Phase 3: Econometric Modeling & Imputation (Stata)**

**Goal:** Estimate fractional logit model and generate predictions.

/*
==============================================================================
PHASE 3.1: IMPORT PREPARED SWAA DATA
==============================================================================
*/

// Import the prepared SWAA training data
import delimited "data/processed/swaa_prepared_for_stata.csv", clear

/*
==============================================================================
PHASE 3.2: ESTIMATE THE FRACTIONAL LOGIT MODEL
==============================================================================
*/

// Estimate fractional logit model with population weights
fracreg logit wfh_share ///
    i.occupation_clean ///
    i.work_industry ///
    i.education_s ///
    i.agebin ///
    i.gender ///
    i.race_ethnicity_s ///
    i.censusdivision ///
    [pweight=cratio100], vce(robust)

estimates store wfh_model

/*
==============================================================================
PHASE 3.3: IMPORT HARMONIZED ACS DATA
==============================================================================
*/

clear
import delimited "data/processed/acs_prepared_for_stata.csv", clear

/*
==============================================================================
PHASE 3.4: GENERATE PREDICTIONS
==============================================================================
*/

estimates restore wfh_model
predict alpha

/*
==============================================================================
PHASE 3.5: SAVE THE FINAL IMPUTED DATASET
==============================================================================
*/

save "output/acs_with_imputed_wfh.dta", replace
export delimited "output/acs_with_imputed_wfh.csv", replace

---

## **Integration Benefits**

Using the Polars-preprocessed ACS data provides:

### **Performance Improvements**
- **Faster processing:** Preprocessed data loads in seconds vs. minutes
- **Lower memory usage:** Optimized data types and compression
- **Scalable to multiple years:** Batch processing capability

### **Enhanced Data Quality**
- **Consistent sample definition:** Pre-filtered working population
- **Richer variable set:** Teleworkability measures, occupation hierarchies
- **Geographic enhancements:** CBSA codes, metropolitan indicators
- **Built-in validation:** Comprehensive quality checks

### **Research Advantages**
- **Multiple teleworkability indices:** Compare across methodologies
- **Occupation hierarchies:** Analysis at different aggregation levels
- **WFH baseline measures:** Transportation-based work-from-home indicators

## **Variable Mapping Between ACS and SWAA**

| SWAA Variable | ACS Source (Preprocessed) | Harmonization Function |
|---------------|---------------------------|------------------------|
| `work_industry` | `INDNAICS` | `harmonize_naics_to_swaa_industry()` |
| `occupation_clean` | `OCCSOC_detailed` | `harmonize_soc_to_swaa_occupation()` |
| `agebin` | `AGE` | `harmonize_age_to_swaa_bins()` |
| `education_s` | `EDUC` | `harmonize_educ_to_swaa()` |
| `female` | `SEX` | `harmonize_sex_to_female()` |
| `race_ethnicity_s` | `RACE`, `HISPAN` | `harmonize_race_ethnicity_to_swaa()` |
| `censusdivision` | `CBSA20` → region | `harmonize_cbsa_to_censusdivision()` |

## **Validation Methodology**

The methodology includes several validation steps to ensure data quality and model performance:

### **Data Quality Checks**
1. **Range validation**: Ensure 0 ≤ alpha ≤ 1 for all predictions
2. **Missing data assessment**: Track and report missing values throughout the process
3. **Sample size validation**: Confirm adequate sample sizes after filtering

### **Model Performance Validation**
1. **Group comparisons**: Compare average WFH shares across occupations/industries between SWAA and CPS data
2. **Face validity checks**: Verify that high-WFH occupations (e.g., software developers) receive high alpha values
3. **Distribution analysis**: Examine the distribution of predicted values for reasonableness

### **Variable Harmonization Validation**
The project includes comprehensive mapping functions for:
- **Industry codes**: NAICS → SWAA categories (18 industry groups)
- **Occupation codes**: SOC → SWAA categories (12 occupation groups)
- **Education levels**: IPUMS → SWAA categories (5 education levels)
- **Age groups**: Continuous age → SWAA bins (5 age groups)
- **Race/ethnicity**: Detailed categories → SWAA simplified (4 categories)

### **Enhanced Validation with Preprocessed Data**
1. **Teleworkability validation:** Compare imputed WFH shares with existing teleworkability indices
2. **Transportation validation:** Compare with WFH indicators from commuting data
3. **Occupation hierarchy validation:** Verify consistency across SOC aggregation levels
4. **Geographic validation:** Check patterns across metropolitan vs. non-metropolitan areas

## **Troubleshooting**

### **ACS Preprocessing Issues**
See [`acs_processing_polar.md`](acs_processing_polar.md) for ACS-specific troubleshooting.

### **Common WFH Imputation Issues**

1. **Missing preprocessed ACS file**
   - **Problem:** `acs_136_processed_polar.csv` not found
   - **Solution:** Run Phase 0 preprocessing first (see `acs_processing_polar.md`)

2. **Variable harmonization failures**
   - **Problem:** Variables don't map correctly between datasets
   - **Solution:** Check mapping functions in `test.py` and validate crosswalks

3. **Sample size mismatches**
   - **Problem:** Unexpected sample sizes after harmonization
   - **Solution:** Review filtering criteria in both preprocessing and harmonization steps

### **Testing Individual Functions**

Use the provided `test.py` script to test individual functions before running the full pipeline:

```python
# Test SWAA data loading and filtering
# %%
# Test industry harmonization
# %%  
# Test occupation harmonization
# etc.
```

This allows you to debug issues with specific functions without running the entire process.

## **Conclusion**

The integrated methodology combining Polars-based ACS preprocessing with WFH imputation provides:
- **Significant performance improvements** through optimized data processing
- **Enhanced research capabilities** with teleworkability measures and occupation hierarchies  
- **Higher data quality** through systematic filtering and validation
- **Scalable approach** for multi-year and multi-dataset analyses

For detailed ACS preprocessing documentation, see [`acs_processing_polar.md`](acs_processing_polar.md).
     - `data/raw/acs/usa_00136.csv.gz`

2. **Stata not found**
   - **Problem**: Stata executable not found in system PATH
   - **Solution**: Install Stata and add to PATH, or run Stata script manually

3. **Variable name mismatches**
   - **Problem**: Required variables not found in SWAA data
   - **Solution**: Check variable names in actual data and update mapping functions

4. **Memory issues with large datasets**
   - **Problem**: Python or Stata runs out of memory
   - **Solution**: Process data in chunks or increase available memory

5. **Harmonization failures**
   - **Problem**: Variables don't map correctly between datasets
   - **Solution**: Review and adjust mapping functions in `test.py` before running full pipeline

### **Testing Individual Functions**

Use the provided `test.py` script to test individual functions before running the full pipeline:

```python
# Test SWAA data loading and filtering
# %%
# Test industry harmonization
# %%  
# Test occupation harmonization
# etc.
```

This allows you to debug issues with specific functions without running the entire process.
