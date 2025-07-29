/*
==============================================================================
WFH IMPUTATION PROJECT - THREE-PART MODEL IMPLEMENTATION
==============================================================================

This Stata script implements an advanced three-part model for WFH imputation:
- Model 1 (Hurdle): P(wfh_share > 0) - probability of any remote work
- Model 2 (Top Corner): P(wfh_share = 1 | wfh_share > 0) - probability of full remote
- Model 3 (Interior): E[wfh_share | 0 < wfh_share < 1] - hybrid work share

This approach better captures the mass points at 0 and 1 in the WFH distribution
compared to standard fractional logit models.

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
capture log using "output/wfh_three_part_model_log.log", replace

display "{hline 80}"
display "WFH IMPUTATION - THREE-PART MODEL IMPLEMENTATION"
display "{hline 80}"

/*
==============================================================================
PHASE A: ESTIMATION ON TRAINING DATA (SWAA)
==============================================================================
*/

display ""
display "PHASE A: ESTIMATION ON TRAINING DATA (SWAA)"
display "{hline 50}"

// Import the prepared SWAA training data
display "Loading SWAA training data..."
import delimited "data/processed/swaa_prepared_for_stata.csv", clear

// Display basic statistics of the training data
display ""
display "Training data summary:"
describe
summarize wfh_share, detail

display ""
display "Original WFH share distribution in SWAA data:"
count if wfh_share == 0
display "Fully in-person (wfh_share = 0): " r(N) " (" %4.1f r(N)/_N*100 "%)"

count if wfh_share == 1
display "Fully remote (wfh_share = 1): " r(N) " (" %4.1f r(N)/_N*100 "%)"

count if wfh_share > 0 & wfh_share < 1
display "Hybrid (0 < wfh_share < 1): " r(N) " (" %4.1f r(N)/_N*100 "%)"

/*
==============================================================================
STEP 1: CREATE DEPENDENT VARIABLES FOR THREE-PART MODEL
==============================================================================
*/

display ""
display "STEP 1: Creating dependent variables for three-part model..."

// 1. Create the "hurdle" variable: Is the person NOT fully in-person?
gen is_remote_any = (wfh_share > 0)
label variable is_remote_any "Binary: Any remote work (wfh_share > 0)"

// 2. Create the "top corner" variable: Is the person fully remote, given they are not fully in-person?
gen is_full_remote = (wfh_share == 1) if wfh_share > 0
label variable is_full_remote "Binary: Fully remote, conditional on any remote work"

// 3. The original wfh_share will be used for the interior model (already exists)

// Display the new variables
display ""
display "New dependent variables created:"
tabulate is_remote_any, missing
tabulate is_full_remote if wfh_share > 0, missing

/*
==============================================================================
STEP 2: ESTIMATE MODEL 1 - THE "HURDLE" MODEL
==============================================================================
*/

display ""
display "STEP 2: Estimating Model 1 - Hurdle Model P(wfh_share > 0)"
display "{hline 60}"

// Model 1: Logit for P(wfh_share > 0)
// This is estimated on the FULL sample
logit is_remote_any ///
    i.occupation_clean ///
    i.work_industry ///
    i.education_s ///
    i.agebin ///
    i.gender ///
    i.race_ethnicity_s ///
    i.censusdivision ///
    i.year ///
    [pweight=cratio100]

// Store the estimates
estimates store model_hurdle

display ""
display "Model 1 (Hurdle) estimation completed and saved as 'model_hurdle'"

/*
==============================================================================
STEP 3: ESTIMATE MODEL 2 - THE "TOP CORNER" MODEL
==============================================================================
*/

display ""
display "STEP 3: Estimating Model 2 - Top Corner Model P(wfh_share = 1 | wfh_share > 0)"
display "{hline 75}"

// Model 2: Logit for P(wfh_share = 1 | wfh_share > 0)
// This is estimated ONLY on the subsample of workers who are not fully in-person
logit is_full_remote ///
    i.occupation_clean ///
    i.work_industry ///
    i.education_s ///
    i.agebin ///
    i.gender ///
    i.race_ethnicity_s ///
    i.censusdivision ///
    i.year ///
    if wfh_share > 0 ///
    [pweight=cratio100]

// Store the estimates
estimates store model_top_corner

display ""
display "Model 2 (Top Corner) estimation completed and saved as 'model_top_corner'"

// Display sample size for this conditional model
count if wfh_share > 0
display "Sample size for Model 2 (workers with wfh_share > 0): " r(N)

/*
==============================================================================
STEP 4: ESTIMATE MODEL 3 - THE "INTERIOR" MODEL
==============================================================================
*/

display ""
display "STEP 4: Estimating Model 3 - Interior Model E[wfh_share | 0 < wfh_share < 1]"
display "{hline 75}"

// Model 3: Fractional Logit for E[wfh_share | 0 < wfh_share < 1]
// This is estimated ONLY on the subsample of hybrid workers
fracreg logit wfh_share ///
    i.occupation_clean ///
    i.work_industry ///
    i.education_s ///
    i.agebin ///
    i.gender ///
    i.race_ethnicity_s ///
    i.censusdivision ///
    i.year ///
    if wfh_share > 0 & wfh_share < 1 ///
    [pweight=cratio100]
	
// Store the estimates
estimates store model_interior

display ""
display "Model 3 (Interior) estimation completed and saved as 'model_interior'"

// Display sample size for this conditional model
count if wfh_share > 0 & wfh_share < 1
display "Sample size for Model 3 (hybrid workers): " r(N)

display ""
display "PHASE A COMPLETED: All three models estimated and saved"
display "- model_hurdle: P(wfh_share > 0)"
display "- model_top_corner: P(wfh_share = 1 | wfh_share > 0)"  
display "- model_interior: E[wfh_share | 0 < wfh_share < 1]"

/*
==============================================================================
PHASE B: IMPUTATION ON PREDICTION DATA (CPS/ACS)
==============================================================================
*/

display ""
display "PHASE B: IMPUTATION ON PREDICTION DATA (CPS/ACS)"
display "{hline 50}"

/*
==============================================================================
STEP 5: LOAD CPS DATA AND GENERATE PREDICTIONS FROM ALL THREE MODELS
==============================================================================
*/

display ""
display "STEP 5: Loading CPS/ACS data and generating predictions..."

// Load CPS/ACS data
clear
import delimited "data/processed/cps_prepared_for_stata.csv", clear

// Display basic info about prediction dataset
display ""
display "Prediction data summary:"
describe
display "Total observations for imputation: " _N

// Predict P(wfh_share > 0) from Model 1
display ""
display "Generating predictions from Model 1 (Hurdle)..."
estimates restore model_hurdle
predict p_remote_any, pr
label variable p_remote_any "Predicted P(wfh_share > 0) from hurdle model"

// Predict P(wfh_share = 1 | wfh_share > 0) from Model 2
display "Generating predictions from Model 2 (Top Corner)..."
estimates restore model_top_corner
predict p_full_remote_cond, pr
label variable p_full_remote_cond "Predicted P(wfh_share = 1 | wfh_share > 0)"

// Predict E[wfh_share | 0 < wfh_share < 1] from Model 3
display "Generating predictions from Model 3 (Interior)..."
estimates restore model_interior
predict alpha_hybrid_pred
label variable alpha_hybrid_pred "Predicted E[wfh_share | 0 < wfh_share < 1]"

display ""
display "All predictions generated successfully"
summarize p_remote_any p_full_remote_cond alpha_hybrid_pred

/*
==============================================================================
STEP 6: COMBINE PREDICTIONS TO SIMULATE THE FINAL IMPUTED VARIABLE
==============================================================================
*/

display ""
display "STEP 6: Combining predictions to simulate final imputed variable..."
display "{hline 65}"

// Set a seed for reproducibility
set seed 12345
display "Random seed set to 12345 for reproducibility"

// Generate two independent random numbers for our two-stage decision
gen u1 = runiform()
gen u2 = runiform()
label variable u1 "Random draw for hurdle decision"
label variable u2 "Random draw for top corner decision"

// Initialize the final imputed variable
gen alpha_final = .
label variable alpha_final "Final Imputed WFH Share (3-Part Model)"

display ""
display "Implementing two-stage probabilistic assignment..."

// --- Decision Stage 1: Are they fully in-person or not? ---
// If their random draw is greater than their probability of doing any remote work,
// they are assigned to be fully in-person.
replace alpha_final = 0 if u1 > p_remote_any

// Count assignments from stage 1
count if alpha_final == 0
local stage1_inperson = r(N)
display "Stage 1: " `stage1_inperson' " workers assigned to fully in-person (alpha = 0)"

// --- Decision Stage 2: If not in-person, are they hybrid or fully remote? ---
// For the remaining observations (where alpha_final is still missing), we make the second decision.
// If their second random draw is less than or equal to their conditional probability
// of being fully remote, they are assigned to be fully remote.
replace alpha_final = 1 if u2 <= p_full_remote_cond & missing(alpha_final)

// Count assignments from stage 2
count if alpha_final == 1
local stage2_fullremote = r(N)
display "Stage 2: " `stage2_fullremote' " workers assigned to fully remote (alpha = 1)"

// The rest must be hybrid. For them, we use the prediction from the fractional logit model.
replace alpha_final = alpha_hybrid_pred if missing(alpha_final)

// Count hybrid assignments
count if alpha_final > 0 & alpha_final < 1
local hybrid_count = r(N)
display "Remaining: " `hybrid_count' " workers assigned hybrid shares (0 < alpha < 1)"

// Verify no missing values
count if missing(alpha_final)
if r(N) > 0 {
    display "WARNING: " r(N) " observations have missing alpha_final values!"
}
else {
    display "SUCCESS: All observations have imputed alpha_final values"
}


/*
==============================================================================
STEP 7: VALIDATE THE FINAL IMPUTED VARIABLE
==============================================================================
*/

display ""
display "STEP 7: Validating the final imputed variable..."
display "{hline 50}"

display ""
display "VALIDATION OF FINAL IMPUTED VARIABLE (3-PART MODEL):"
display "{hline 55}"

// Detailed summary statistics
summarize alpha_final, detail

// Check the proportions
display ""
display "DISTRIBUTION BREAKDOWN:"

count if alpha_final == 0
local zeros = r(N)
display "Fully in-person (alpha = 0): " `zeros' " observations (" %4.1f `zeros'/_N*100 "%)"

count if alpha_final == 1  
local ones = r(N)
display "Fully remote (alpha = 1): " `ones' " observations (" %4.1f `ones'/_N*100 "%)"

count if alpha_final > 0 & alpha_final < 1
local hybrid = r(N)
display "Hybrid (0 < alpha < 1): " `hybrid' " observations (" %4.1f `hybrid'/_N*100 "%)"

// Additional validation checks
display ""
display "ADDITIONAL VALIDATION CHECKS:"

// Check for values outside [0,1] range
count if alpha_final < 0 | alpha_final > 1
if r(N) > 0 {
    display "WARNING: " r(N) " observations have alpha_final outside [0,1] range!"
} 
else {
    display "✓ All alpha_final values are within [0,1] range"
}

// Show distribution by key variables for face validity
display ""
display "MEAN ALPHA_FINAL BY OCCUPATION (Top 10 by sample size):"
preserve
collapse (mean) alpha_final (count) n=alpha_final, by(occupation_clean)
gsort -n
list occupation_clean alpha_final n in 1/10, clean noobs
restore

display ""
display "MEAN ALPHA_FINAL BY INDUSTRY (Top 10 by sample size):"
preserve  
collapse (mean) alpha_final (count) n=alpha_final, by(work_industry)
gsort -n
list work_industry alpha_final n in 1/10, clean noobs
restore

/*
==============================================================================
FINAL OUTPUT: SAVE THE IMPUTED DATASET
==============================================================================
*/

display ""
display "SAVING FINAL IMPUTED DATASET..."
display "{hline 40}"

// Add metadata about the imputation method
gen imputation_method = "three_part_model"
label variable imputation_method "Method used for WFH imputation"

gen imputation_date = date(c(current_date), "DMY")
format imputation_date %td
label variable imputation_date "Date of imputation"

// Save in Stata format
save "output/cps_with_imputed_wfh_three_part.dta", replace
display "✓ Saved: output/cps_with_imputed_wfh_three_part.dta"

// Save in CSV format  
export delimited "output/cps_with_imputed_wfh_three_part.csv", replace
display "✓ Saved: output/cps_with_imputed_wfh_three_part.csv"

// Save the model estimates
estimates restore model_hurdle
estimates save "output/wfh_model_hurdle.ster", replace
display "✓ Saved: output/wfh_model_hurdle.ster"

estimates restore model_top_corner  
estimates save "output/wfh_model_top_corner.ster", replace
display "✓ Saved: output/wfh_model_top_corner.ster"

estimates restore model_interior
estimates save "output/wfh_model_interior.ster", replace  
display "✓ Saved: output/wfh_model_interior.ster"

/*
==============================================================================
SUMMARY AND CONCLUSION
==============================================================================
*/

display ""
display "{hline 80}"
display "THREE-PART MODEL IMPUTATION COMPLETED SUCCESSFULLY"
display "{hline 80}"

display ""
display "SUMMARY OF RESULTS:"
display "• Total observations processed: " _N
display "• Fully in-person workers (alpha = 0): " `zeros' " (" %4.1f `zeros'/_N*100 "%)"
display "• Fully remote workers (alpha = 1): " `ones' " (" %4.1f `ones'/_N*100 "%)"
display "• Hybrid workers (0 < alpha < 1): " `hybrid' " (" %4.1f `hybrid'/_N*100 "%)"

display ""
display "OUTPUT FILES CREATED:"
display "• output/cps_with_imputed_wfh_three_part.dta (main output)"
display "• output/cps_with_imputed_wfh_three_part.csv (CSV version)"
display "• output/wfh_model_hurdle.ster (hurdle model estimates)"
display "• output/wfh_model_top_corner.ster (top corner model estimates)"
display "• output/wfh_model_interior.ster (interior model estimates)"
display "• output/wfh_three_part_model_log.log (detailed log file)"

display ""
display "The three-part model approach provides a more realistic imputation"
display "that better captures the mass points at 0 and 1 in the WFH distribution."

// Close log file
capture log close _all

display ""
display "Analysis complete. See log file for detailed results."

/*
==============================================================================
DISTRIBUTION PLOTS OF IMPUTED ALPHA
==============================================================================
*/

display ""
display "CREATING DISTRIBUTION PLOTS OF IMPUTED ALPHA..."
display "{hline 50}"

// Create histogram of alpha_final
histogram alpha_final, ///
    width(0.05) ///
    frequency ///
    title("Distribution of Imputed WFH Share (Three-Part Model)") ///
    xtitle("Imputed WFH Share (alpha_final)") ///
    ytitle("Frequency") ///
    note("Mass points at 0 and 1 show fully in-person and fully remote workers") ///
    scheme(s1color)
    
graph export "output/alpha_final_histogram.png", replace width(800) height(600)
display "✓ Saved: output/alpha_final_histogram.png"

// Create a more detailed histogram focusing on the interior (0,1)
histogram alpha_final if alpha_final > 0 & alpha_final < 1, ///
    width(0.025) ///
    frequency ///
    title("Distribution of Hybrid WFH Shares (0 < alpha < 1)") ///
    xtitle("Imputed WFH Share (alpha_final)") ///
    ytitle("Frequency") ///
    note("Distribution among hybrid workers only") ///
    scheme(s1color)
    
graph export "output/alpha_final_hybrid_only.png", replace width(800) height(600)
display "✓ Saved: output/alpha_final_hybrid_only.png"

// Create a bar chart showing the three categories
gen wfh_category = "Fully In-Person" if alpha_final == 0
replace wfh_category = "Hybrid" if alpha_final > 0 & alpha_final < 1
replace wfh_category = "Fully Remote" if alpha_final == 1

// Calculate percentages for the bar chart
preserve
contract wfh_category, freq(count) percent(pct)
sort count
graph bar pct, over(wfh_category, sort(count) descending) ///
    title("Work Arrangement Distribution (Three-Part Model)") ///
    ytitle("Percentage of Workers") ///
    ylabel(0(10)100) ///
    blabel(bar, format(%4.1f)) ///
    note("Based on imputed WFH shares from three-part model") ///
    scheme(s1color)
    
graph export "output/wfh_categories_bar.png", replace width(800) height(600)
display "✓ Saved: output/wfh_categories_bar.png"
restore

// Summary statistics table
display ""
display "SUMMARY STATISTICS OF IMPUTED ALPHA:"
display "{hline 40}"
summarize alpha_final, detail

// Show key percentiles
display ""
display "KEY PERCENTILES:"
_pctile alpha_final, p(10 25 50 75 90 95 99)
display "10th percentile: " %6.3f r(r1)
display "25th percentile: " %6.3f r(r2) 
display "50th percentile: " %6.3f r(r3)
display "75th percentile: " %6.3f r(r4)
display "90th percentile: " %6.3f r(r5)
display "95th percentile: " %6.3f r(r6)
display "99th percentile: " %6.3f r(r7)

display ""
display "DISTRIBUTION PLOTS COMPLETED"
display "✓ All plots saved to output/ directory"
