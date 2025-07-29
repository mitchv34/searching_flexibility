/*
==============================================================================
WFH IMPUTATION PROJECT - ECONOMETRIC MODELING & IMPUTATION
==============================================================================

This Stata script implements Phase 3 of the WFH imputation methodology:
- Estimate fractional logit model on SWAA data
- Generate predictions for ACS data (preprocessed via Polars pipeline)
- Save final imputed dataset

Following the methodology outlined in doc/imputing_wfh_swaa_to_cps.md
ACS preprocessing documented in doc/acs_processing_polar.md

Author: Generated for WFH Imputation Project
Date: July 2025
==============================================================================
*/

clear all
set more off
set linesize 120

// Set working directory to project root
cd "v:\high_tech_ind\WFH\searching_flexibility"

// Create output directory if it doesn't exist
capture mkdir "output"

// Create log file
capture log close _all
capture log using "output/wfh_imputation_log.log", replace

display "{hline 80}"
display "WFH IMPUTATION - ECONOMETRIC MODELING & IMPUTATION"
display "{hline 80}"

/*
==============================================================================
PHASE 3.1: IMPORT PREPARED SWAA DATA
==============================================================================
*/

display ""
display "PHASE 3.1: IMPORTING PREPARED SWAA DATA"
display "{hline 50}"

// Import the prepared SWAA training data
// Note: The filename corresponds to SWAA_OUTPUT_FILE in the Python script
import delimited "data/processed/swaa_prepared_for_stata.csv", clear

// Display basic information about the SWAA data
describe
summarize

// Check the dependent variable
display ""
display "Dependent Variable Summary:"
summarize wfh_share, detail

// Check for any issues with the WFH share variable
count if wfh_share < 0
count if wfh_share > 1
count if missing(wfh_share)

/*
==============================================================================
PHASE 3.2: ESTIMATE THE FRACTIONAL LOGIT MODEL
==============================================================================
*/

display ""
display "PHASE 3.2: ESTIMATING FRACTIONAL LOGIT MODEL"
display "{hline 50}"

// Check which predictor variables are available
describe occupation_clean work_industry education_s agebin gender race_ethnicity_s censusdivision

// Estimate the fractional logit model with population weights
display "Estimating fractional logit model..."

// Basic model - adjust based on available variables
capture {
    fracreg logit wfh_share ///
        i.occupation_clean ///
        i.work_industry ///
        i.education_s ///
        i.agebin ///
        i.gender ///
        i.race_ethnicity_s ///
        i.censusdivision ///
        [pweight=cratio100], vce(robust)
}

// If full model fails, try simplified version
if _rc != 0 {
    display "Full model failed, trying simplified version..."
    
    // Check which variables actually exist and have variation
    foreach var of varlist occupation_clean work_industry education_s agebin gender race_ethnicity_s censusdivision {
        capture confirm variable `var'
        if _rc == 0 {
            display "Variable `var' exists"
            tab `var', missing
        }
        else {
            display "Variable `var' not found"
        }
    }
    
    // Simplified model with core variables
    fracreg logit wfh_share ///
        i.occupation_clean ///
        i.education_s ///
        i.gender ///
        [pweight=cratio100], vce(robust)
}

// Store the model results
estimates store wfh_model

// Display model results
display ""
display "Model Estimation Results:"
estimates replay wfh_model

// Save model estimates
estimates save "output/wfh_model_estimates", replace

/*
==============================================================================
PHASE 3.3: IMPORT HARMONIZED ACS DATA
==============================================================================
*/

display ""
display "PHASE 3.3: IMPORTING HARMONIZED ACS DATA"
display "{hline 50}"

// Clear memory and import ACS prediction data
// Note: The filename corresponds to ACS_OUTPUT_FILE in the Python script
clear
import delimited "data/processed/acs_prepared_for_stata.csv", clear

// Display basic information about the ACS data
describe
summarize

// Check sample size
display "ACS sample size: " _N

// Verify unique identifier exists and is unique
capture confirm string variable unique_person_id
if _rc == 0 {
    display "✓ Found unique_person_id variable"
    
    // Check for uniqueness
    duplicates report unique_person_id
    local dups = r(N) - r(unique_N)
    
    if `dups' == 0 {
        display "✓ unique_person_id is properly unique"
    }
    else {
        display "⚠ WARNING: `dups' duplicate unique_person_id values found"
    }
}
else {
    display "✗ ERROR: unique_person_id variable not found!"
    display "Cannot proceed without unique identifier for merging back to original data"
    exit 1
}

/*
==============================================================================
PHASE 3.4: GENERATE PREDICTIONS
==============================================================================
*/

display ""
display "PHASE 3.4: GENERATING PREDICTIONS"
display "{hline 50}"

// Restore the model
estimates restore wfh_model

// Generate predictions
display "Generating WFH share predictions..."
predict alpha

// Check predictions
display ""
display "Prediction Summary:"
summarize alpha, detail

// Validate predictions are in valid range [0,1]
count if alpha < 0
count if alpha > 1
count if missing(alpha)

// Display distribution of predictions
display ""
display "Distribution of Predicted WFH Shares:"
histogram alpha, frequency title("Distribution of Predicted WFH Shares") ///
    xlabel(0(0.1)1) ylabel(, format(%9.0fc))

/*
==============================================================================
PHASE 3.5: SAVE THE FINAL IMPUTED DATASET
==============================================================================
*/

display ""
display "PHASE 3.5: SAVING FINAL IMPUTED DATASET"
display "{hline 50}"

// Compress to optimize file size
compress

// Add labels for clarity
label variable alpha "Predicted WFH Share (Imputed from SWAA)"
label variable unique_person_id "Unique Person Identifier (SAMPLE_SERIAL_PERNUM)"

// Display final dataset information
describe
summarize alpha

// Verify we still have the unique identifier
display ""
display "Final dataset verification:"
display "- Total observations: " _N
display "- Unique person IDs: " 
quietly duplicates report unique_person_id
display r(unique_N)

// Save as Stata data file with unique identifier preserved
save "output/acs_with_imputed_wfh.dta", replace

// Also save as CSV for broader compatibility
export delimited "output/acs_with_imputed_wfh.csv", replace

display ""
display "Final dataset saved successfully!"
display "Stata file: output/acs_with_imputed_wfh.dta"
display "CSV file: output/acs_with_imputed_wfh.csv"
display "Both files contain unique_person_id for merging back to original ACS data"

/*
==============================================================================
PHASE 4: VALIDATION AND SANITY CHECKS
==============================================================================
*/

display ""
display "PHASE 4: VALIDATION AND SANITY CHECKS"
display "{hline 50}"

/*
PHASE 4.1: SUMMARIZE THE PREDICTIONS
*/

display ""
display "4.1 PREDICTION SUMMARY:"
display "{hline 30}"

summarize alpha, detail

// Check for valid range
local min_alpha = r(min)
local max_alpha = r(max)
local mean_alpha = r(mean)
local median_alpha = r(p50)

display ""
display "Validation Checks:"
display "- Minimum value: " %6.4f `min_alpha' " (should be >= 0)"
display "- Maximum value: " %6.4f `max_alpha' " (should be <= 1)"
display "- Mean value: " %6.4f `mean_alpha'
display "- Median value: " %6.4f `median_alpha'

if `min_alpha' < 0 {
    display "WARNING: Predictions below 0 detected!"
}
if `max_alpha' > 1 {
    display "WARNING: Predictions above 1 detected!"
}

/*
PHASE 4.2: COMPARE GROUP AVERAGES
*/

display ""
display "4.2 GROUP AVERAGES COMPARISON:"
display "{hline 45}"

// Calculate average alpha by occupation category
capture {
    display "Average WFH Share by Occupation:"
    table occupation_clean, contents(mean alpha) format(%6.4f)
}

// Calculate average alpha by industry
capture {
    display ""
    display "Average WFH Share by Industry:"
    table work_industry, contents(mean alpha) format(%6.4f)
}

// Calculate average alpha by education
capture {
    display ""
    display "Average WFH Share by Education:"
    table education_s, contents(mean alpha) format(%6.4f)
}

// Calculate average alpha by gender
capture {
    display ""
    display "Average WFH Share by Gender:"
    table gender, contents(mean alpha) format(%6.4f)
}

/*
PHASE 4.3: SPOT-CHECK INDIVIDUAL PREDICTIONS
*/

display ""
display "4.3 SPOT-CHECK INDIVIDUAL PREDICTIONS:"
display "{hline 40}"

// Display some high and low predictions for manual inspection
display ""
display "Highest WFH Share Predictions (Top 10):"
gsort -alpha
list occupation_clean work_industry education_s gender alpha in 1/10, clean

display ""
display "Lowest WFH Share Predictions (Bottom 10):"
gsort alpha
list occupation_clean work_industry education_s gender alpha in 1/10, clean

/*
FINAL SUMMARY
*/

display ""
display "{hline 80}"
display "WFH IMPUTATION COMPLETE"
display "{hline 80}"
display ""
display "Summary:"
display "- Model estimated on SWAA data"
display "- Predictions generated for ACS data (preprocessed via Polars)"
display "- Final dataset: output/acs_with_imputed_wfh.dta"
display "- Sample size: " _N
display "- Mean WFH share: " %6.4f `mean_alpha'
display "- Unique identifier preserved for merging: unique_person_id"
display ""
display "Files created:"
display "- output/wfh_model_estimates.ster (model estimates)"
display "- output/acs_with_imputed_wfh.dta (main output with unique_person_id)"
display "- output/acs_with_imputed_wfh.csv (CSV version with unique_person_id)"
display "- output/wfh_imputation_log.log (this log)"
display ""
display "NOTE: Use unique_person_id to merge imputed WFH shares back to original ACS data"
display "ACS preprocessing documented in doc/acs_processing_polar.md"

// Close log
capture log close

// Display completion message
display ""
display "Analysis complete! Check the log file for detailed results."
