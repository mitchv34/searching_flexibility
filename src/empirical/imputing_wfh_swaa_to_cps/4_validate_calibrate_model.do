/*
==============================================================================
WFH IMPUTATION PROJECT - VALIDATION AND CALIBRATION
==============================================================================

This Stata script performs validation and calibration of the three-part model
WFH imputation using the CPS WFH indicator as ground truth.

This script should be run AFTER the main three-part model imputation has been
completed and the results have been saved.

Prerequisites:
- output/cps_with_imputed_wfh_three_part.dta must exist
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
use "output/cps_with_imputed_wfh_three_part.dta", clear

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
VALIDATION AND CALIBRATION USING CPS WFH INDICATOR
==============================================================================
*/

display ""
display "{hline 80}"
display "VALIDATION AND CALIBRATION USING CPS WFH INDICATOR"
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
        
        // Summary table (updated for Stata 17+ syntax)
        table wfh, statistic(mean alpha_final) statistic(count alpha_final) nformat(%6.4f)
        
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
               note("Source: CPS data with 3-part model imputation") ///
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
        
        /* ORIGINAL ROC ANALYSIS - COMMENTED OUT DUE TO PERFORMANCE WITH LARGE DATASET
        // ROC analysis (simplified for large dataset - no graph)
        display "Performing ROC analysis on full dataset... "
        roctab wfh p_remote_any, summary
        display "✓ Finished ROC analysis on full dataset"
        */
        
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
            // Use 10% random sample for graphing (still ~200k observations)
            sample 10
            display "Sample size for ROC graph: " _N " observations"
            
            // Create ROC graph using logistic regression
            quietly logistic wfh p_remote_any
            lroc, title("ROC Curve: Predicting Actual WFH from Model Probabilities") ///
                  subtitle("Based on 10% random sample of CPS data") ///
                  note("Sample-based analysis for computational efficiency") ///
                  scheme(s1color)
        
            graph export "output/validation_roc_curve.png", replace width(800) height(600)
            display "✓ Saved: output/validation_roc_curve.png"
        restore
        
        // Alternative: Use roctab with graph on sample if you prefer
        display ""
        display "Creating additional ROC analysis with graph using sample..."
        preserve
            sample 5  // Use 5% sample (~100k observations) for roctab graph
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
        CALIBRATION STEP 1: CALCULATE TARGET PROPORTION FROM CPS
        ==============================================================================
        */
        
        display ""
        display "CALIBRATION STEP 1: Calculating Target Proportion from CPS WFH Indicator"
        display "{hline 70}"
        
        quietly summarize wfh
        local target_p_remote_any = r(mean)
        display "Target proportion for any remote work (from CPS): " %6.4f `target_p_remote_any'
        
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
        label variable alpha_final_calibrated "Final Imputed WFH Share (Calibrated to CPS)"
        
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
        
        // Distribution comparison by WFH status (updated for Stata 17+ syntax)
        display ""
        display "Mean alpha by WFH status - Original vs Calibrated:"
        table wfh, statistic(mean alpha_final) statistic(mean alpha_final_calibrated) nformat(%6.4f)
        
        // Create comparison plot
        twoway (kdensity alpha_final, legend(label(1 "Original Imputation"))) ///
               (kdensity alpha_final_calibrated, legend(label(2 "Calibrated Imputation"))), ///
               title("Comparison of Original vs Calibrated Imputations", size(medium)) ///
               xtitle("Imputed WFH Share") ytitle("Density") ///
               note("Calibrated version matches CPS aggregate WFH share") ///
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
        label variable target_wfh_share "Target WFH share from CPS validation data"
        
        gen calibration_cutoff = `p_cutoff_value'
        label variable calibration_cutoff "Cutoff value used for calibration"
        
        // Save updated dataset
        save "output/cps_with_imputed_wfh_validated.dta", replace
        display "✓ Saved: output/cps_with_imputed_wfh_validated.dta"
        
        export delimited "output/cps_with_imputed_wfh_validated.csv", replace
        display "✓ Saved: output/cps_with_imputed_wfh_validated.csv"
        
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
        
        /* ORIGINAL AUC CALCULATION - COMMENTED OUT DUE TO PERFORMANCE WITH LARGE DATASET
        quietly roctab wfh p_remote_any
        local auc = r(area)
        display "• ROC Area Under Curve (AUC): " %6.4f `auc'
        */
        
        // Use sample-based AUC calculated earlier
        display "• ROC Area Under Curve (AUC - from 10% sample): " %6.4f `auc_sample'
        
        display ""
        display "CALIBRATION RESULTS:"
        // Calculate original any-remote share
        count if alpha_final > 0
        local original_any_remote = r(N) / _N
        display "• Original any-remote share: " %6.4f `original_any_remote'
        display "• Target any-remote share (CPS): " %6.4f `target_p_remote_any'
        display "• Calibrated any-remote share: " %6.4f `calibrated_share'
        display "• Calibration accuracy: " %6.4f (1 - abs(`calibrated_share' - `target_p_remote_any'))
        
        display ""
        display "CLASSIFICATION PERFORMANCE:"
        display "• Original model accuracy: " %6.4f `accuracy'
        display "• Calibrated model accuracy: " %6.4f `accuracy_c'
        display "• Original model AUC: " %6.4f `auc'
        
        display ""
        display "OUTPUT FILES WITH VALIDATION:"
        display "• output/cps_with_imputed_wfh_validated.dta (main output with validation)"
        display "• output/cps_with_imputed_wfh_validated.csv (CSV version)"
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

// Close log file
capture log close _all

display ""
display "{hline 80}"
display "VALIDATION AND CALIBRATION ANALYSIS COMPLETE"
display "{hline 80}"
