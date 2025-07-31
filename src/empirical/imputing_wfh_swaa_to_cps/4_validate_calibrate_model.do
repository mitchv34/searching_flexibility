/*
==============================================================================
WFH IMPUTATION PROJECT - VALIDATION AND CALIBRATION
==============================================================================

This Stata script performs validation and calibration of the three-part model
WFH imputation using the ACS WFH indicator as ground truth.

This script should be run AFTER the main three-part model imputation has been
completed and the results have been saved.

Prerequisites:
- output/acs_with_imputed_wfh_three_part.dta must exist
- The dataset must contain: alpha_final, p_remote_any, p_full_remote_cond, 
  alpha_hybrid_pred, and wfh variables

Author: Generated for WFH Imputation Project
Date: July 2025
==============================================================================
*/

clear all
set more off
set linesize 120

// Set working directory to project root
* global project_root "/Users/mitchv34/Work/searching_flexibility/"
global project_root "/project/high_tech_ind/WFH/searching_flexibility/"
global data_path "$project_root/data"
global code_path "$project_root/src"
global output_path "$project_root/output"


// Create output directory if it doesn't exist
capture mkdir "output"

// Create dedicated log file for validation
capture log close _all
capture log using "output/wfh_validation_calibration_log.log", replace

display "{hline 80}"
display "WFH IMPUTATION - VALIDATION AND CALIBRATION"
display "{hline 80}"

/*
==============================================================================
LOAD IMPUTED DATASET
==============================================================================
*/

display ""
display "Loading imputed dataset for validation..."
display "{hline 45}"

// Load the dataset with imputed WFH shares
use "output/acs_with_imputed_wfh_three_part.dta", clear

// Display basic info about the dataset
display ""
display "Dataset summary:"
describe
display "Total observations: " _N

// Check required variables exist
local required_vars "alpha_final p_remote_any p_full_remote_cond alpha_hybrid_pred wfh"
local missing_vars ""

foreach var of local required_vars {
    capture confirm variable `var'
    if _rc != 0 {
        local missing_vars "`missing_vars' `var'"
    }
}

if "`missing_vars'" != "" {
    display "ERROR: Missing required variables: `missing_vars'"
    display "Please ensure the three-part model has been run and saved properly."
    exit 198
}

display "✓ All required variables found"

/*
==============================================================================
VALIDATION AND CALIBRATION USING ACS WFH INDICATOR
==============================================================================
*/

display ""
display "{hline 80}"
display "VALIDATION AND CALIBRATION USING ACS WFH INDICATOR"
display "{hline 80}"

// Check if WFH validation variable exists
capture confirm variable wfh
if _rc == 0 {
    display "✓ WFH validation indicator found - proceeding with validation"
    
    // Count observations with non-missing WFH data
    count if !missing(wfh)
    local wfh_obs = r(N)
    display "Observations with WFH data available: " `wfh_obs' " of " _N " (" %4.1f `wfh_obs'/_N*100 "%)"
    
    if `wfh_obs' > 0 {
        /*
        ==============================================================================
        VALIDATION STEP 1: COMPARE MEAN IMPUTED ALPHA BY ACTUAL WFH STATUS
        ==============================================================================
        */
        
        display ""
        display "VALIDATION STEP 1: Comparing Mean Imputed Alpha by Actual WFH Status"
        display "{hline 70}"
        
        // Summary table - use tabstat for compatibility
        tabstat alpha_final, by(wfh) statistics(mean count) nototal
        
        // Statistical test
        display ""
        display "Statistical test for difference in means:"
        ttest alpha_final, by(wfh)
        
        /*
        ==============================================================================
        VALIDATION STEP 2: PLOT DISTRIBUTIONS BY WFH STATUS
        ==============================================================================
        */
        
        display ""
        display "VALIDATION STEP 2: Creating distribution plots by WFH status..."
        
        // Kernel density plot
        twoway (kdensity alpha_final if wfh==0, legend(label(1 "Did Not WFH (wfh=0)"))) ///
               (kdensity alpha_final if wfh==1, legend(label(2 "Did WFH (wfh=1)"))), ///
               title("Distribution of Imputed WFH Share by Actual WFH Status", size(medium)) ///
               xtitle("Imputed WFH Share (alpha_final)") ytitle("Density") ///
               note("Source: ACS data with 3-part model imputation") ///
               scheme(s1color)
               
        graph export "output/validation_alpha_by_wfh_status.png", replace width(800) height(600)
        display "✓ Saved: output/validation_alpha_by_wfh_status.png"
        
        /*
        ==============================================================================
        VALIDATION STEP 3: ROC ANALYSIS OF PREDICTIVE POWER
        ==============================================================================
        */
        
        display ""
        display "VALIDATION STEP 3: ROC Analysis of Hurdle Model's Predictive Power"
        display "{hline 65}"
        
        // ROC analysis using sample (faster alternative)
        display "Performing ROC analysis using 10% sample (faster for large dataset)..."
        preserve
            sample 10
            display "Sample size for ROC analysis: " _N " observations"
            roctab wfh p_remote_any, summary
            local auc_sample = r(area)
        restore
        display "✓ Finished ROC analysis on sample - AUC: " %6.4f `auc_sample'
        
        // Create ROC graph using a random sample (faster)
        display ""
        display "Creating ROC curve graph using 10% random sample..."
        preserve
            // Use 10% random sample for graphing
            sample 10
            display "Sample size for ROC graph: " _N " observations"
            
            // Create ROC graph using logistic regression
            quietly logistic wfh p_remote_any
            lroc, title("ROC Curve: Predicting Actual WFH from Model Probabilities") ///
                  subtitle("Based on 10% random sample of ACS data") ///
                  note("Sample-based analysis for computational efficiency") ///
                  scheme(s1color)
        
            graph export "output/validation_roc_curve.png", replace width(800) height(600)
            display "✓ Saved: output/validation_roc_curve.png"
        restore
        
        // Alternative: Use roctab with graph on sample
        display ""
        display "Creating additional ROC analysis with graph using sample..."
        preserve
            sample 5  // Use 5% sample for roctab graph
            display "Sample size for roctab graph: " _N " observations"
            
            roctab wfh p_remote_any, graph ///
                   title("ROC Analysis: Model Predictions vs Actual WFH") ///
                   subtitle("Based on 5% random sample") ///
                   scheme(s1color)
        
            graph export "output/validation_roc_curve_roctab.png", replace width(800) height(600)
            display "✓ Saved: output/validation_roc_curve_roctab.png"
        restore
        
        display "✓ ROC analysis complete - both summary statistics and graphs created"
        
        /*
        ==============================================================================
        CALIBRATION STEP 1: CALCULATE TARGET PROPORTION FROM ACS
        ==============================================================================
        */
        
        display ""
        display "CALIBRATION STEP 1: Calculating Target Proportion from ACS WFH Indicator"
        display "{hline 70}"
        
        quietly summarize wfh
        local target_p_remote_any = r(mean)
        display "Target proportion for any remote work (from ACS): " %6.4f `target_p_remote_any'
        
        /*
        ==============================================================================
        CALIBRATION STEP 2: FIND THE CALIBRATION CUTOFF VALUE
        ==============================================================================
        */
        
        display ""
        display "CALIBRATION STEP 2: Finding the calibration cutoff value..."
        
        // Calculate the percentile we need
        local p_cutoff = (1 - `target_p_remote_any') * 100
        display "To match target, we need to set the cutoff at the " %4.1f `p_cutoff' "th percentile."
        
        // Find the cutoff value
        quietly summarize p_remote_any, detail
        
        // Get the specific percentile (use _pctile for exact percentile)
        _pctile p_remote_any, p(`p_cutoff')
        local p_cutoff_value = r(r1)
        display "The cutoff value is p_remote_any = " %6.4f `p_cutoff_value'
        
        /*
        ==============================================================================
        CALIBRATION STEP 3: IMPLEMENT THE CALIBRATED IMPUTATION
        ==============================================================================
        */
        
        display ""
        display "CALIBRATION STEP 3: Implementing the calibrated imputation..."
        display "{hline 60}"
        
        // Set seed for reproducibility
        set seed 12345
        
        // Drop existing calibration variables if they exist
        capture drop u2_calib alpha_final_calibrated
        
        gen u2_calib = runiform()
        
        // Initialize the calibrated variable
        gen alpha_final_calibrated = .
        label variable alpha_final_calibrated "Final Imputed WFH Share (Calibrated to ACS)"
        
        // Stage 1 (Calibrated): Assign in-person based on the new cutoff
        replace alpha_final_calibrated = 0 if p_remote_any <= `p_cutoff_value'
        
        // Stage 2 (Unchanged): For the rest, decide between hybrid and full remote
        replace alpha_final_calibrated = 1 if u2_calib <= p_full_remote_cond & missing(alpha_final_calibrated)
        
        // Stage 3 (Unchanged): Assign the interior prediction to the remaining hybrid workers
        replace alpha_final_calibrated = alpha_hybrid_pred if missing(alpha_final_calibrated)
        
        /*
        ==============================================================================
        CALIBRATION STEP 4: VERIFY THE CALIBRATION
        ==============================================================================
        */
        
        display ""
        display "CALIBRATION STEP 4: Verifying the calibration..."
        display "{hline 50}"
        
        // Check the calibrated share
        count if alpha_final_calibrated > 0
        local calibrated_share = r(N) / _N
        display "Share of any remote work in calibrated data: " %6.4f `calibrated_share'
        display "Target share was: " %6.4f `target_p_remote_any'
        display "Difference: " %6.4f abs(`calibrated_share' - `target_p_remote_any')
        
        // Summary statistics
        display ""
        display "Summary of calibrated variable:"
        summarize alpha_final_calibrated, detail
        
        /*
        ==============================================================================
        ADDITIONAL VALIDATION: COMPARE ORIGINAL VS CALIBRATED
        ==============================================================================
        */
        
        display ""
        display "ADDITIONAL VALIDATION: Comparing Original vs Calibrated Imputations"
        display "{hline 65}"
        
        // Correlation between original and calibrated
        corr alpha_final alpha_final_calibrated
        
        // Distribution comparison by WFH status
        display ""
        display "Mean alpha by WFH status - Original vs Calibrated:"
        display "WFH Status | Original Mean | Calibrated Mean"
        display "{hline 40}"
        
        quietly summarize alpha_final if wfh == 0
        local orig_mean_0 = r(mean)
        quietly summarize alpha_final_calibrated if wfh == 0
        local calib_mean_0 = r(mean)
        display "WFH = 0    |    " %8.4f `orig_mean_0' "   |     " %8.4f `calib_mean_0'
        
        quietly summarize alpha_final if wfh == 1
        local orig_mean_1 = r(mean)
        quietly summarize alpha_final_calibrated if wfh == 1
        local calib_mean_1 = r(mean)
        display "WFH = 1    |    " %8.4f `orig_mean_1' "   |     " %8.4f `calib_mean_1'
        
        // Create comparison plot
        twoway (kdensity alpha_final, legend(label(1 "Original Imputation"))) ///
               (kdensity alpha_final_calibrated, legend(label(2 "Calibrated Imputation"))), ///
               title("Comparison of Original vs Calibrated Imputations", size(medium)) ///
               xtitle("Imputed WFH Share") ytitle("Density") ///
               note("Calibrated version matches ACS aggregate WFH share") ///
               scheme(s1color)
               
        graph export "output/comparison_original_vs_calibrated.png", replace width(800) height(600)
        display "✓ Saved: output/comparison_original_vs_calibrated.png"
        
        // Distribution breakdown for calibrated version
        display ""
        display "CALIBRATED DISTRIBUTION BREAKDOWN:"
        
        count if alpha_final_calibrated == 0
        local zeros_calib = r(N)
        display "Fully in-person (alpha = 0): " `zeros_calib' " observations (" %4.1f `zeros_calib'/_N*100 "%)"
        
        count if alpha_final_calibrated == 1  
        local ones_calib = r(N)
        display "Fully remote (alpha = 1): " `ones_calib' " observations (" %4.1f `ones_calib'/_N*100 "%)"
        
        count if alpha_final_calibrated > 0 & alpha_final_calibrated < 1
        local hybrid_calib = r(N)
        display "Hybrid (0 < alpha < 1): " `hybrid_calib' " observations (" %4.1f `hybrid_calib'/_N*100 "%)"
        
        /*
        ==============================================================================
        DETAILED VALIDATION ANALYSIS
        ==============================================================================
        */
        
        display ""
        display "DETAILED VALIDATION ANALYSIS"
        display "{hline 35}"
        
        // Cross-tabulation of imputed categories vs actual WFH
        display ""
        display "Cross-tabulation: Imputed categories vs Actual WFH"
        
        // Create categorical version of alpha_final for crosstab
        gen alpha_cat = "In-person" if alpha_final == 0
        replace alpha_cat = "Hybrid" if alpha_final > 0 & alpha_final < 1
        replace alpha_cat = "Full Remote" if alpha_final == 1
        
        tab alpha_cat wfh, row col
        
        // Calibrated version
        gen alpha_cat_calib = "In-person" if alpha_final_calibrated == 0
        replace alpha_cat_calib = "Hybrid" if alpha_final_calibrated > 0 & alpha_final_calibrated < 1
        replace alpha_cat_calib = "Full Remote" if alpha_final_calibrated == 1
        
        display ""
        display "Cross-tabulation: Calibrated categories vs Actual WFH"
        tab alpha_cat_calib wfh, row col
        
        // Calculate classification accuracy metrics
        display ""
        display "CLASSIFICATION ACCURACY METRICS:"
        display "{hline 40}"
        
        // For original imputation (treating alpha > 0 as "remote")
        gen imputed_remote = (alpha_final > 0)
        gen calibrated_remote = (alpha_final_calibrated > 0)
        
        // Calculate accuracy, sensitivity, specificity
        tab imputed_remote wfh, row col
        
        display ""
        display "ORIGINAL IMPUTATION CLASSIFICATION METRICS:"
        quietly tab imputed_remote wfh
        local total = r(N)
        local true_pos = r(r21)    // predicted remote, actually remote
        local false_pos = r(r12)   // predicted remote, actually not remote  
        local true_neg = r(r11)    // predicted not remote, actually not remote
        local false_neg = r(r22)   // predicted not remote, actually remote
        
        local accuracy = (`true_pos' + `true_neg') / `total'
        local sensitivity = `true_pos' / (`true_pos' + `false_neg')
        local specificity = `true_neg' / (`true_neg' + `false_pos')
        local precision = `true_pos' / (`true_pos' + `false_pos')
        
        display "• Accuracy: " %6.4f `accuracy'
        display "• Sensitivity (True Positive Rate): " %6.4f `sensitivity'
        display "• Specificity (True Negative Rate): " %6.4f `specificity'
        display "• Precision (Positive Predictive Value): " %6.4f `precision'
        
        display ""
        display "CALIBRATED IMPUTATION CLASSIFICATION METRICS:"
        quietly tab calibrated_remote wfh
        local total_c = r(N)
        local true_pos_c = r(r21)
        local false_pos_c = r(r12)
        local true_neg_c = r(r11)
        local false_neg_c = r(r22)
        
        local accuracy_c = (`true_pos_c' + `true_neg_c') / `total_c'
        local sensitivity_c = `true_pos_c' / (`true_pos_c' + `false_neg_c')
        local specificity_c = `true_neg_c' / (`true_neg_c' + `false_pos_c')
        local precision_c = `true_pos_c' / (`true_pos_c' + `false_pos_c')
        
        display "• Accuracy: " %6.4f `accuracy_c'
        display "• Sensitivity (True Positive Rate): " %6.4f `sensitivity_c'
        display "• Specificity (True Negative Rate): " %6.4f `specificity_c'
        display "• Precision (Positive Predictive Value): " %6.4f `precision_c'
        
        /*
        ==============================================================================
        SAVE RESULTS WITH VALIDATION VARIABLES
        ==============================================================================
        */
        
        display ""
        display "SAVING RESULTS WITH VALIDATION AND CALIBRATION..."
        display "{hline 55}"
        
        // Add validation metadata (drop if they exist first)
        capture drop validation_performed target_wfh_share calibration_cutoff
        
        gen validation_performed = 1
        label variable validation_performed "Whether WFH validation was performed"
        
        gen target_wfh_share = `target_p_remote_any'
        label variable target_wfh_share "Target WFH share from ACS validation data"
        
        gen calibration_cutoff = `p_cutoff_value'
        label variable calibration_cutoff "Cutoff value used for calibration"
        
        // Save updated dataset
        save "output/acs_with_imputed_wfh_validated.dta", replace
        display "✓ Saved: output/acs_with_imputed_wfh_validated.dta"
        
        export delimited "output/acs_with_imputed_wfh_validated.csv", replace
        display "✓ Saved: output/acs_with_imputed_wfh_validated.csv"
        
        /*
        ==============================================================================
        VALIDATION AND CALIBRATION SUMMARY
        ==============================================================================
        */
        
        display ""
        display "{hline 80}"
        display "VALIDATION AND CALIBRATION SUMMARY"
        display "{hline 80}"
        
        display ""
        display "VALIDATION RESULTS:"
        quietly ttest alpha_final, by(wfh)
        local mean_diff = r(mu_2) - r(mu_1)
        local p_value = r(p)
        display "• Mean difference in imputed alpha (WFH=1 vs WFH=0): " %6.4f `mean_diff'
        display "• P-value for difference: " %6.4f `p_value'
        display "• ROC Area Under Curve (AUC - from 10% sample): " %6.4f `auc_sample'
        
        display ""
        display "CALIBRATION RESULTS:"
        // Calculate original any-remote share
        count if alpha_final > 0
        local original_any_remote = r(N) / _N
        display "• Original any-remote share: " %6.4f `original_any_remote'
        display "• Target any-remote share (ACS): " %6.4f `target_p_remote_any'
        display "• Calibrated any-remote share: " %6.4f `calibrated_share'
        display "• Calibration accuracy: " %6.4f (1 - abs(`calibrated_share' - `target_p_remote_any'))
        
        display ""
        display "CLASSIFICATION PERFORMANCE:"
        display "• Original model accuracy: " %6.4f `accuracy'
        display "• Calibrated model accuracy: " %6.4f `accuracy_c'
        display "• Original model AUC: " %6.4f `auc_sample'
        
        display ""
        display "OUTPUT FILES WITH VALIDATION:"
        display "• output/acs_with_imputed_wfh_validated.dta (main output with validation)"
        display "• output/acs_with_imputed_wfh_validated.csv (CSV version)"
        display "• output/validation_alpha_by_wfh_status.png (distribution comparison)"
        display "• output/validation_roc_curve.png (ROC analysis)"
        display "• output/comparison_original_vs_calibrated.png (calibration comparison)"
        display "• output/wfh_validation_calibration_log.log (this log file)"
        
    }
    else {
        display "ERROR: No observations with valid WFH data found"
        display "Skipping validation and calibration steps"
    }
}
else {
    display "WARNING: WFH validation variable not found in dataset"
    display "Skipping validation and calibration steps"
    display "This is expected if validation data is not available"
}

/*
==============================================================================
WORK ARRANGEMENT STATISTICS - ORIGINAL VS CALIBRATED IMPUTATION - WEIGHTED
==============================================================================
*/

display ""
display "WORK ARRANGEMENT STATISTICS - ORIGINAL VS CALIBRATED IMPUTATION - WEIGHTED"
display "{hline 80}"

// Check if weights exist for ACS data
capture confirm variable perwt
if _rc == 0 {
    local weight_opt "perwt"
    display "Using ACS person weights (perwt) for validation statistics"
}
else {
    local weight_opt ""
    display "No weights available for ACS data - using unweighted statistics"
}

// Original imputation statistics (weighted)
display ""
display "ORIGINAL IMPUTATION STATISTICS (WEIGHTED):"
display "{hline 45}"

if "`weight_opt'" != "" {
    // Weighted counts using summarize
    quietly summarize `weight_opt' if alpha_final == 0
    local orig_inperson = r(sum)
    quietly summarize `weight_opt'
    local total_weighted = r(sum)
    local orig_pct_inperson = (`orig_inperson' / `total_weighted') * 100

    quietly summarize `weight_opt' if alpha_final > 0 & alpha_final < 1
    local orig_hybrid = r(sum)
    local orig_pct_hybrid = (`orig_hybrid' / `total_weighted') * 100

    quietly summarize `weight_opt' if alpha_final == 1
    local orig_remote = r(sum)
    local orig_pct_remote = (`orig_remote' / `total_weighted') * 100
}
else {
    // Unweighted counts
    count if alpha_final == 0
    local orig_inperson = r(N)
    local total_weighted = _N
    local orig_pct_inperson = (`orig_inperson' / `total_weighted') * 100

    count if alpha_final > 0 & alpha_final < 1
    local orig_hybrid = r(N)
    local orig_pct_hybrid = (`orig_hybrid' / `total_weighted') * 100

    count if alpha_final == 1
    local orig_remote = r(N)
    local orig_pct_remote = (`orig_remote' / `total_weighted') * 100
}

display "• Fully In-Person: " %12.0fc `orig_inperson' " (" %5.2f `orig_pct_inperson' "%)"
display "• Hybrid: " %19.0fc `orig_hybrid' " (" %5.2f `orig_pct_hybrid' "%)"
display "• Fully Remote: " %15.0fc `orig_remote' " (" %5.2f `orig_pct_remote' "%)"

local orig_total_remote = `orig_hybrid' + `orig_remote'
if `orig_total_remote' > 0 {
    local orig_pct_hybrid_remote = (`orig_hybrid' / `orig_total_remote') * 100
    local orig_pct_full_remote = (`orig_remote' / `orig_total_remote') * 100
    
    display "Among remote workers:"
    display "  - Hybrid: " %5.2f `orig_pct_hybrid_remote' "% | Fully Remote: " %5.2f `orig_pct_full_remote' "%"
}

// Calibrated imputation statistics (weighted)
display ""
display "CALIBRATED IMPUTATION STATISTICS (WEIGHTED):"
display "{hline 47}"

if "`weight_opt'" != "" {
    quietly summarize `weight_opt' if alpha_final_calibrated == 0
    local calib_inperson = r(sum)
    local calib_pct_inperson = (`calib_inperson' / `total_weighted') * 100

    quietly summarize `weight_opt' if alpha_final_calibrated > 0 & alpha_final_calibrated < 1
    local calib_hybrid = r(sum)
    local calib_pct_hybrid = (`calib_hybrid' / `total_weighted') * 100

    quietly summarize `weight_opt' if alpha_final_calibrated == 1
    local calib_remote = r(sum)
    local calib_pct_remote = (`calib_remote' / `total_weighted') * 100
}
else {
    count if alpha_final_calibrated == 0
    local calib_inperson = r(N)
    local calib_pct_inperson = (`calib_inperson' / `total_weighted') * 100

    count if alpha_final_calibrated > 0 & alpha_final_calibrated < 1
    local calib_hybrid = r(N)
    local calib_pct_hybrid = (`calib_hybrid' / `total_weighted') * 100

    count if alpha_final_calibrated == 1
    local calib_remote = r(N)
    local calib_pct_remote = (`calib_remote' / `total_weighted') * 100
}

display "• Fully In-Person: " %12.0fc `calib_inperson' " (" %5.2f `calib_pct_inperson' "%)"
display "• Hybrid: " %19.0fc `calib_hybrid' " (" %5.2f `calib_pct_hybrid' "%)"
display "• Fully Remote: " %15.0fc `calib_remote' " (" %5.2f `calib_pct_remote' "%)"

local calib_total_remote = `calib_hybrid' + `calib_remote'
if `calib_total_remote' > 0 {
    local calib_pct_hybrid_remote = (`calib_hybrid' / `calib_total_remote') * 100
    local calib_pct_full_remote = (`calib_remote' / `calib_total_remote') * 100
    
    display "Among remote workers:"
    display "  - Hybrid: " %5.2f `calib_pct_hybrid_remote' "% | Fully Remote: " %5.2f `calib_pct_full_remote' "%"
}

// Remote work hours comparison (weighted)
display ""
display "REMOTE WORK HOURS COMPARISON (40-hour work week, weighted):"
display "{hline 65}"

// Original hours
gen orig_remote_hours = alpha_final * 40
gen calib_remote_hours = alpha_final_calibrated * 40

if "`weight_opt'" != "" {
    quietly mean orig_remote_hours [pweight=`weight_opt']
    local orig_avg_all = _b[orig_remote_hours]
    quietly mean orig_remote_hours [pweight=`weight_opt'] if alpha_final > 0
    local orig_avg_remote = _b[orig_remote_hours]
    quietly mean orig_remote_hours [pweight=`weight_opt'] if alpha_final > 0 & alpha_final < 1
    local orig_avg_hybrid = _b[orig_remote_hours]

    // Calibrated hours
    quietly mean calib_remote_hours [pweight=`weight_opt']
    local calib_avg_all = _b[calib_remote_hours]
    quietly mean calib_remote_hours [pweight=`weight_opt'] if alpha_final_calibrated > 0
    local calib_avg_remote = _b[calib_remote_hours]
    quietly mean calib_remote_hours [pweight=`weight_opt'] if alpha_final_calibrated > 0 & alpha_final_calibrated < 1
    local calib_avg_hybrid = _b[calib_remote_hours]
}
else {
    quietly mean orig_remote_hours
    local orig_avg_all = _b[orig_remote_hours]
    quietly mean orig_remote_hours if alpha_final > 0
    local orig_avg_remote = _b[orig_remote_hours]
    quietly mean orig_remote_hours if alpha_final > 0 & alpha_final < 1
    local orig_avg_hybrid = _b[orig_remote_hours]

    // Calibrated hours
    quietly mean calib_remote_hours
    local calib_avg_all = _b[calib_remote_hours]
    quietly mean calib_remote_hours if alpha_final_calibrated > 0
    local calib_avg_remote = _b[calib_remote_hours]
    quietly mean calib_remote_hours if alpha_final_calibrated > 0 & alpha_final_calibrated < 1
    local calib_avg_hybrid = _b[calib_remote_hours]
}

display "                          Original    Calibrated    Difference"
display "Economy-wide average:     " %8.2f `orig_avg_all' "     " %8.2f `calib_avg_all' "     " %8.2f (`calib_avg_all' - `orig_avg_all')
display "Among remote workers:     " %8.2f `orig_avg_remote' "     " %8.2f `calib_avg_remote' "     " %8.2f (`calib_avg_remote' - `orig_avg_remote')
display "Among hybrid workers:     " %8.2f `orig_avg_hybrid' "     " %8.2f `calib_avg_hybrid' "     " %8.2f (`calib_avg_hybrid' - `orig_avg_hybrid')

// Percentage of time worked remotely (weighted)
display ""
display "PERCENTAGE OF TIME WORKED REMOTELY (WEIGHTED):"
display "{hline 50}"

local orig_pct_all = `orig_avg_all' / 40 * 100
local orig_pct_remote = `orig_avg_remote' / 40 * 100
local orig_pct_hybrid = `orig_avg_hybrid' / 40 * 100

local calib_pct_all = `calib_avg_all' / 40 * 100
local calib_pct_remote = `calib_avg_remote' / 40 * 100
local calib_pct_hybrid = `calib_avg_hybrid' / 40 * 100

display "                          Original    Calibrated    Difference"
display "Economy-wide:             " %7.2f `orig_pct_all' "%     " %7.2f `calib_pct_all' "%     " %7.2f (`calib_pct_all' - `orig_pct_all') "%"
display "Among remote workers:     " %7.2f `orig_pct_remote' "%     " %7.2f `calib_pct_remote' "%     " %7.2f (`calib_pct_remote' - `orig_pct_remote') "%"
display "Among hybrid workers:     " %7.2f `orig_pct_hybrid' "%     " %7.2f `calib_pct_hybrid' "%     " %7.2f (`calib_pct_hybrid' - `orig_pct_hybrid') "%"

// ACS Validation Statistics (weighted)
display ""
display "ACS VALIDATION REFERENCE (WEIGHTED):"
display "{hline 35}"

if "`weight_opt'" != "" {
    quietly summarize `weight_opt' if wfh == 1
    local acs_remote_count = r(sum)
    local acs_remote_pct = (`acs_remote_count' / `total_weighted') * 100

    quietly summarize `weight_opt' if wfh == 0
    local acs_inperson_count = r(sum)
    local acs_inperson_pct = (`acs_inperson_count' / `total_weighted') * 100
}
else {
    count if wfh == 1
    local acs_remote_count = r(N)
    local acs_remote_pct = (`acs_remote_count' / `total_weighted') * 100

    count if wfh == 0
    local acs_inperson_count = r(N)
    local acs_inperson_pct = (`acs_inperson_count' / `total_weighted') * 100
}

display "• ACS Any Remote Work: " %12.0fc `acs_remote_count' " (" %5.2f `acs_remote_pct' "%)"
display "• ACS Fully In-Person: " %12.0fc `acs_inperson_count' " (" %5.2f `acs_inperson_pct' "%)"

// Clean up temporary variables
drop orig_remote_hours calib_remote_hours

display ""
display "✓ Comparative work arrangement statistics completed (using weights where available)"

// Close log file
capture log close _all

display ""
display "{hline 80}"
display "VALIDATION AND CALIBRATION ANALYSIS COMPLETE"
display "{hline 80}"
