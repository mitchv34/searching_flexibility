# Data Processing & Moment Construction Documentation

This document records the end-to-end data preparation and transformation steps used to create the empirical moments for model calibration (steady states 2019 & 2024). It is the single source of truth for all manipulations applied to raw CPS, BLS, and FRED data prior to estimation.

## 1. CPS Pipeline (`wfh_cps_polar.py`)

### 1.1 Raw Input
- Source: CPS monthly microdata extract ( file e.g. `cps_00038.csv.gz`) stored under `data/raw/cps/`.
- Years processed: `min_year` (default 2013) through 2025 (cap enforced regardless of input `max_year`).
- Key raw variables used: YEAR, MONTH, MISH, WTFINL, AGE, SEX, RACE, HISPAN, EDUC, UHRSWORK1, OCC, IND, EARNWEEK2, HOURWAGE2, TELWRKHR, TELWRKPAY.

### 1.2 Filtering & Basic Cleaning
- Hours: Currently NO hard exclusion beyond what upstream extraction includes (previous 35+ hour filter relaxed to better align with BLS shares). Hours logic retained in function if needed.
- Wage non-response cleanup: Drop rows with placeholder 999 / 9999 sentinel codes when constructing wages (logic in `filter_cps_data`).
- Class of worker: Keep all classes except codes 00 (NIU) and 99 (missing); implemented by passing `class_of_worker="all"` which expands to an allowlist.
- Hispanic recode: Convert HISPAN to binary (1 if >0 & !=9, 0 otherwise), dropping HISPAN==9.

### 1.3 Industry Codes
- Strip whitespace; remove unemployed code 9920, zero code "0", and military industry codes 9890–9899.
- Map CPS IND → NAICS using crosswalk `data/aux_and_croswalks/cw_ind_naics.csv` column `definitive` (handles 2025 reclassification harmonization).

### 1.4 Occupation Codes → SOC Hierarchy
- Construct hierarchical SOC aggregator from `soc_structure_2018.xlsx` (Detailed → Broad → Minor → Major).
- CPS OCC mapped to SOC using `cps_occ_soc_cw.csv`. For CPS codes mapping to multiple SOC codes, aggregate to the narrowest common prefix (detailed, broad, minor, else major).
- Rows with unresolved mappings dropped.
- Derived columns: `OCCSOC_detailed`, `OCCSOC_broad`, `OCCSOC_minor` plus classification group tag.

### 1.5 Teleworkability Index
- Merge custom occupation WFH index (`wfh_estimates.csv`).
- Compute average teleworkability for detailed, broad, and minor SOC groupings: `TELEWORKABLE_OCCSOC_{detailed,broad,minor}`.

### 1.6 Work Arrangement (WFH) Measures
- Use TELWRKHR (telework hours) & TELWRKPAY (Yes/No) for 2022+.
  1. Drop TELWRKPAY == "0" (NIU).
  2. If TELWRKPAY == "2" (No) but TELWRKHR > 0, set TELWRKHR = 0.
  3. ALPHA = TELWRKHR / UHRSWORK1, clamped to [0,1].
  4. Dummies: FULL_REMOTE (ALPHA==1), FULL_INPERSON (ALPHA==0), HYBRID (0<ALPHA<1), WFH = 1(ALPHA>0).

### 1.7 Wage Construction
- Inputs considered: HOURWAGE (legacy), HOURWAGE2 (new), EARNWEEK / EARNWEEK2, UHRSWORK1.
- Two temporary wage candidates built (WAGE_TEMP1, WAGE_TEMP2); final `WAGE` uses priority: HOURWAGE2, else HOURWAGE, else EARNWEEK2/UHRSWORK1, else EARNWEEK/UHRSWORK1 (subject to sentinels & hours validity). Final selection consolidated into a single `WAGE` variable.
- Source provenance flag created later (`SOURCE_WAGE_FLAG`).

### 1.8 Wage Non-Response Reweighting
- Function `reweight_for_wage_nonresponse` generates adjusted weights `WTFINL_ADJ` for wage sample.
- Stratification cells: YEAR, SEX, EDUC, RACE, AGE_GROUP (derived). Adjustment factor = TARGET_POP / OBSERVED_POP, capped at 5.0.
- Year-level ratio calibration ensures sum of adjusted weights equals original target within each YEAR.
- Export wage sample renames `WTFINL_ADJ` back to `WTFINL`, preserving original as `WTFINL_ORIG`.

### 1.9 Top-Coding Identification & Adjustment (NEW)
- Executed before reweighting via `add_topcode_and_real_wages`.
- Known hourly wage top-code thresholds (HOURWAGE2) for MISH {4,8} from Apr 2023–Mar 2024 hard-coded (user-provided table).
- Weekly earnings (EARNWEEK2) fixed top-code 2884.61 through March 2024; after March 2024 dynamic.
- Post-March 2024 (and months lacking explicit thresholds): dynamic detection: if WAGE equals monthly max and appears in ≥5 observations, flagged top-coded.
- Combined flags → `WAGE_TOPCODED_FLAG` (1 if any rule triggers).
- Adjustment: multiply top-coded `WAGE` by 1.3 (standard CPS factor). Raw pre-adjustment stored in `WAGE_RAW`.
- Source flag: `SOURCE_WAGE_FLAG` (2=HOURWAGE2, 1=HOURWAGE, 3=EARNWEEK2 derived, 0=other/NA).

### 1.10 CPI Deflation
- CPI series: FRED `CPIAUCSL` (monthly) fetched lazily & cached.
- Base year: 2019 annual average (CPI_BASE), forming CPI_INDEX = CPI / CPI_BASE.
- Real wage: `WAGE_REAL = WAGE / CPI_INDEX`; real log wage: `LOG_WAGE_REAL = ln(WAGE_REAL)`.
- CPI columns added: CPI, CPI_INDEX.

### 1.11 Exports
- Two CSVs under `data/processed/cps/`:
  - `<file>_universe.csv`: target pre-wage filtering sample (original weights).
  - `<file>_wage_reweighted.csv`: wage sample with adjusted weights and enriched variables (WAGE_RAW, WAGE, WAGE_REAL, LOG_WAGE_REAL, WAGE_TOPCODED_FLAG, SOURCE_WAGE_FLAG, telework indicators, occupation & industry mappings, teleworkability indices).

## 2. BLS Telework Shares (Work Arrangement Moments)
- Series pulled via BLS API (requires key in `api_keys.json` under key `bls`).
- Series used: 
  - `LNU0201B4DD` (PERSON_TELEWORK_SOME_PCT) – interpreted as *some hours but not all* (hybrid share).
  - `LNU0201B54F` (PERSON_TELEWORK_ALL_PCT) – fully remote share.
- Annual shares computed as simple (unweighted) means of monthly percentages / 100.
- In-person share = 1 − (HYBRID_SHARE + REMOTE_SHARE), clamped at ≥0.
- Saved: `monthly_bls_telework.csv`, `quarterly_bls_telework.csv`, `yearly_bls_telework.csv` in `data/moments/`.

## 3. FRED Series
- Productivity: `OPHNFB` (Nonfarm Business Sector: Real Output per Hour). Annual averages (2019, 2024) used for moment targeting (function implemented; may be toggled on in `construct_all_moments`).
- Labor Market Tightness (if needed): Job openings `JTSJOL` & unemployment level `UNEMPLOY`; monthly merge to compute θ = V / U; aggregated to quarterly and annual.

## 4. Wage Distribution Moments
- Constructed in `data_moments.py` via `construct_wage_distribution_moments` using the reweighted wage file (`*_wage_reweighted.csv`).
- Statistic definitions (per year y):
  - Mean log wage: μ_y = Σ w_i log(w_i^real) / Σ w_i.
  - Variance log wage: σ²_y = Σ w_i (log(w_i^real) − μ_y)^2 / Σ w_i.
- Optional diagnostics (future enhancement): Effective sample size N_eff = (Σ w)^2 / Σ w²; weighted top-coded share = Σ w_i * 1(WAGE_TOPCODED_FLAG) / Σ w_i.

## 5. Compensating Differential (Optional / To Finalize)
- Regression (if enabled): log(real wage) on HYBRID + FULL_INPERSON + demographics + education fixed effects; baseline = FULL_REMOTE.
- Current implementation uses OLS (can be upgraded to WLS with weights `WTFINL`).

## 6. Firm Efficiency Moments (Placeholders)
- Not computable with CPS alone (requires matched employer-employee data & firm efficiency metric ψ). Placeholders remain; estimation routines should exclude or treat as future data expansion.

## 7. Reproducibility & Caching
- CPI fetched on demand; consider persisting under `data/aux_and_croswalks/cpi_cache.csv` if API stability is a concern.
- BLS & FRED pulls produce CSV snapshots in `data/moments/` to freeze target moment values for replication.

## 8. Known Limitations / Next Steps
- Telework series interpretation assumes PERSON_TELEWORK_SOME_PCT excludes fully remote workers (pure hybrid). Verify BLS metadata; if it includes “some or all” adjust hybrid share = SOME − ALL.
- Top-code dynamic rule threshold (≥5 obs) is heuristic; can refine using weighted tail mass (e.g., top 3% weighted cutoff) once distribution diagnostics available.
- No explicit trimming of implausible wages (e.g., WAGE_REAL > $500/hr); consider adding defensive bounds.
- Demographic controls in compensating differential regression minimal; expand to race, Hispanic, region, occupation & industry fixed effects for robustness.

## 9. Variable Glossary (New / Key)
- WAGE_RAW: Constructed nominal hourly wage prior to top-code adjustment.
- WAGE: Nominal hourly wage after top-code multiplier (1.3 if flagged).
- WAGE_TOPCODED_FLAG: 1 if observation matched explicit or dynamic top-code rule.
- SOURCE_WAGE_FLAG: 2 (HOURWAGE2), 1 (HOURWAGE), 3 (EARNWEEK2 derived), 0 other.
- CPI, CPI_INDEX: Monthly CPI level and relative index vs 2019 average.
- WAGE_REAL, LOG_WAGE_REAL: Real hourly wage & log real wage (2019 base year).
- HYBRID_SHARE / REMOTE_SHARE / IN_PERSON_SHARE: Aggregated BLS arrangement shares.

---
For questions or updates, modify this file in tandem with any logic changes in `wfh_cps_polar.py` or `data_moments.py` to maintain consistency.
