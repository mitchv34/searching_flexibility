@echo off
REM WFH Imputation Project - Windows Batch Runner
REM This file runs the complete WFH imputation pipeline

echo ============================================================
echo WFH IMPUTATION PROJECT - BATCH RUNNER
echo ============================================================
echo.

echo Current directory: %CD%
echo.

REM Change to the project root directory
cd /d "v:\high_tech_ind\WFH\searching_flexibility"

echo Changed to project directory: %CD%
echo.

echo TIP: To test individual functions first, run:
echo      python src\empirical\imputing_wfh_swaa_to_cps\test.py
echo.

REM Run the master Python script
echo Running master script...
python "src\empirical\imputing_wfh_swaa_to_cps\run_master.py"

echo.
echo ============================================================
echo BATCH RUNNER COMPLETE
echo ============================================================
echo.
echo Check the output folder for results:
echo - output\cps_with_imputed_wfh.dta (main output)
echo - output\wfh_imputation_log.log (detailed log)
echo.
echo For debugging, use: python src\empirical\imputing_wfh_swaa_to_cps\test.py
echo.

pause
