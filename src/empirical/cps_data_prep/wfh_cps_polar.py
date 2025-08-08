# %% Import packages and define constants
from ast import Return
import os
from pathlib import Path
import polars as pl
import pandas as pd
import numpy as np
import logging
import argparse
import time

# Set up logging
current_dir = os.path.dirname(os.path.abspath(__file__))
script_name = os.path.basename(__file__)
log_file = os.path.join(current_dir, f'{script_name.replace(".py", ".log")}')
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent.parent
DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'cps')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'cps')

CPS_OCC_SOC_CW = os.path.join(BASE_DIR, 'data', 'aux_and_croswalks', 'cps_occ_soc_cw.csv')
CPS_IND_NAICS_CW = os.path.join(BASE_DIR, 'data', 'aux_and_croswalks', 'cw_ind_naics.csv')
SOC_AGGREGATOR = os.path.join(BASE_DIR, 'data', 'aux_and_croswalks', 'soc_structure_2018.xlsx')
STATE_CENSUSDIV = os.path.join(BASE_DIR, 'data', 'aux_and_croswalks', "state_census_divisions.csv")
WFH_INDEX = os.path.join(BASE_DIR, 'data', 'results', 'wfh_estimates.csv')

COLS_TO_EXPORT = [
    "CPSIDP",                      # Unique identifier for each person (from CPS)
    "YEAR",                        # Survey year
    "MONTH",                       # Survey month
    "WTFINL",                      # Final person weight
    "AGE",                         # Age
    "SEX",                         # Sex
    "RACE",                        # Race
    "HISPAN",                      # Hispanic indicator
    "EDUC",                        # Education
    "CLASSWKR",                    # Class of worker
    "WAGE",                        # Hourly wage
    "IND",                         # Industry code (mapped to NAICS)
    "OCCSOC_DETAILED",             # Detailed SOC occupation code (mapped from census OCC)
    "OCCSOC_BROAD",                # Broad SOC occupation code (mapped from census OCC)
    "OCCSOC_MINOR",                # Minor SOC occupation code (mapped from census OCC)
    "TELEWORKABLE_OCCSOC_DETAILED",# Teleworkablilty index of the occupation (detailed SOC)
    "TELEWORKABLE_OCCSOC_BROAD",   # Teleworkablilty index of the occupation (broad SOC)
    "TELEWORKABLE_OCCSOC_MINOR",   # Teleworkablilty index of the occupation (minor SOC)
    "STATE_FIPS",                  # State FIPS code
    "STATE_NAME",                  # State name
    "CENSUSDIV",                   # Census division
    "ALPHA",                       # Share of hours worked from home
    "WFH",                         # Work-from-home dummy (1 if ALPHA > 0, else 0)
    "FULL_INPERSON",               # Dummy: fully in-person (ALPHA == 0)
    "FULL_REMOTE",                 # Dummy: fully remote (ALPHA == 1)
    "HYBRID"                       # Dummy: hybrid (0 < ALPHA < 1)
]

# Ensure directories exist
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# %% Define functions

def read_cps_data(path, min_year=None, max_year=None):
    """
    Read CPS data from a CSV file using Polars and filter by year.
    """
    logging.info(f"Reading CPS data from {path}")
    # Define data types (Polars types)
    dtypes = {
        "YEAR": pl.Int64,
        "SERIAL": pl.Int64,
        "MONTH": pl.Int64,
        "HWTFINL": pl.Float64,
        "CPSID": pl.Utf8,
        "ASECFLAG": pl.Int64,
        "HFLAG": pl.Int64,
        "ASECWTH": pl.Float64,
        "STATEFIP": pl.Utf8,
        "PERNUM": pl.Int64,
        "WTFINL": pl.Float64,
        "CPSIDP": pl.Utf8,
        "CPSIDV": pl.Int64,
        "ASECWT": pl.Float64,
        "AGE": pl.Int64,
        "SEX": pl.Int64,
        "RACE": pl.Utf8,
        "HISPAN": pl.Int64,
        "OCC": pl.Utf8,
        "IND": pl.Utf8,
        "CLASSWKR": pl.Utf8,
        "UHRSWORK1": pl.Float64,
        "EDUC": pl.Utf8,
        "INCTOT": pl.Float64,
        "INCWAGE": pl.Float64,
        "TELWRKHR": pl.Float64
    }
    data = pl.read_csv(path, schema_overrides=dtypes)
    logging.info(f"Data shape after reading: {data.shape}")

    if min_year is not None:
        data = data.filter(pl.col("YEAR") >= min_year)
        logging.info(f"Filtered data to min_year {min_year}. Shape is now: {data.shape}")
    if max_year is not None:
        data = data.filter(pl.col("YEAR") <= max_year)
        logging.info(f"Filtered data to max_year {max_year}. Shape is now: {data.shape}")
    return data


def filter_cps_data(data, **kwargs):
    """
    Filter CPS data based on working hours, wage, and class of worker.
    """
    logging.info("Starting filter_cps_data function.")
    hours_worked_lim = kwargs.get("hours_worked_lim", 35)
    if "UHRSWORK1" in data.columns:
        initial_shape = data.shape
        data = data.filter(pl.col("UHRSWORK1") > hours_worked_lim)
        logging.info(f"Filtered by UHRSWORK1 > {hours_worked_lim}. Rows reduced from {initial_shape[0]} to {data.shape[0]}.")
    else:
        logging.warning("Column 'UHRSWORK1' not found. Skipping filter based on minimum hours worked.")
    
    if all(col in data.columns for col in ["HOURWAGE2", "EARNWEEK2", "UHRSWORK1"]):
        # 1. Drop rows where (UHRSWORK1 == 999 or 997) and HOURWAGE2 == 999.99 and EARNWEEK2 == 999999.99
        initial_shape = data.shape
        data = data.filter(
            ~(
                ((pl.col("UHRSWORK1") == 999) | (pl.col("UHRSWORK1") == 997)) &
                (pl.col("HOURWAGE2") == 999.99) &
                (pl.col("EARNWEEK2") == 999999.99)
            )
        )
        logging.info(f"Dropped rows with (UHRSWORK1 == 999 or 997) and HOURWAGE2 == 999.99 and EARNWEEK2 == 999999.99 (No Wage Info). Rows reduced from {initial_shape[0]} to {data.shape[0]}.")

        # 2. Define WAGE:
        #    - If HOURWAGE2 != 999.99, use HOURWAGE2
        #    - Else if (UHRSWORK1 == 999 or 997) and EARNWEEK2 != 999999.99, use EARNWEEK2 / UHRSWORK1
        data = data.with_columns(
            pl.when(pl.col("HOURWAGE2") != 999.99)
            .then(pl.col("HOURWAGE2"))
            .when(
                ((pl.col("UHRSWORK1") == 999) | (pl.col("UHRSWORK1") == 997)) &
                (pl.col("EARNWEEK2") != 999999.99)
            )
            .then(pl.col("EARNWEEK2") / pl.col("UHRSWORK1"))
            .otherwise(None)
            .alias("WAGE")
        )
        logging.info("Created 'WAGE' variable using HOURWAGE2 or EARNWEEK2/UHRSWORK1 as appropriate.")

        # 3. Remove wages not above wage_lim
        wage_lim = kwargs.get("wage_lim", 5)
        initial_shape = data.shape
        data = data.filter(pl.col("WAGE") > wage_lim)
        logging.info(f"Filtered by WAGE > {wage_lim}. Rows reduced from {initial_shape[0]} to {data.shape[0]}.")
    else:
        logging.warning("One or more of HOURWAGE2, EARNWEEK2, UHRSWORK1 not found. Skipping wage construction and filtering.")
        
    # CPS Class of Worker codes: 20=Works for wages/salary, 21=Wage/salary private, 22=Private for profit,
    # 23=Private nonprofit, 24=Wage/salary government, 25=Federal government employee, 
    # 27=State government employee, 28=Local government employee
    class_of_worker = kwargs.get("class_of_worker", ['20', '21', '22', '23', '24', '25', '27', '28'])
    if "CLASSWKR" in data.columns:
        initial_shape = data.shape
        data = data.filter(pl.col("CLASSWKR").is_in(class_of_worker))
        logging.info(f"Filtered by CLASSWKR in {class_of_worker}. Rows reduced from {initial_shape[0]} to {data.shape[0]}.")
    else:
        logging.warning("Column 'CLASSWKR' not found. Skipping filter based on class of worker.")

    # Handle HISPAN variable - convert to binary Hispanic indicator
    if "HISPAN" in data.columns:
        initial_shape = data.shape
        # Ensure HISPAN is numeric
        data = data.with_columns(pl.col("HISPAN").cast(pl.Int64))
        # Filter out missing values (code 9 or other missing codes)
        data = data.filter(pl.col("HISPAN") != 9)
        logging.info(f"Removed rows with HISPAN==9 (missing). Rows reduced from {initial_shape[0]} to {data.shape[0]}.")
        # Convert to binary (0 if 0, 1 if 1-8)
        data = data.with_columns(
            (pl.col("HISPAN") > 0).cast(pl.Int64).alias("HISPAN")
        )
        logging.info("Converted HISPAN to binary indicator (0=not Hispanic, 1=Hispanic).")
    else:
        logging.warning("Column 'HISPAN' not found. Skipping Hispanic indicator conversion.")

    return data


def modify_industry_codes(data):
    """
    Modify industry codes by stripping whitespace, removing unwanted codes,
    and mapping to NAICS codes using the crosswalk.
    """
    logging.info("Starting modify_industry_codes function.")
    
    # Strip whitespace from IND column
    data = data.with_columns(pl.col("IND").str.strip_chars())
    initial_shape = data.shape
    
    # Remove unemployed codes first as requested
    data = data.filter(pl.col("IND") != "9920")
    logging.info(f"Removed unemployed codes (IND=='9920'). Rows reduced from {initial_shape[0]} to {data.shape[0]}.")
    
    # Remove zeros
    prev_shape = data.shape[0]
    data = data.filter(pl.col("IND") != "0")
    logging.info(f"Removed rows with IND=='0'. Rows reduced from {prev_shape} to {data.shape[0]}.")

    # Remove military industry codes
    prev_shape = data.shape[0]
    military_codes = ["9890", "9891", "9892", "9893", "9894", "9895", "9896", "9897", "9898", "9899"]
    data = data.filter(~pl.col("IND").is_in(military_codes))
    logging.info(f"Removed military industry codes. Rows reduced from {prev_shape} to {data.shape[0]}.")

    # Load industry crosswalk
    logging.info(f"Loading industry crosswalk from {CPS_IND_NAICS_CW}")
    ind_naics_cw = pl.read_csv(CPS_IND_NAICS_CW)
    
    # Create mapping dictionary from CPS industry codes to NAICS codes
    ind_to_naics_dict = dict(zip(
        ind_naics_cw["IND"].cast(pl.Utf8).to_list(),
        ind_naics_cw["definitive"].cast(pl.Utf8).to_list() # column definitive contains the NAICS codes mappings from either pre 2025 and post 2025 re-classification
    ))
    
    # Log before mapping
    initial_ind_codes = data["IND"].n_unique()
    logging.info(f"Found {initial_ind_codes} unique industry codes before mapping.")
    
    # Map IND codes to NAICS codes, replacing the original IND column
    data = data.with_columns(
        pl.col("IND").map_elements(
            lambda x: convert_using_cw(x, ind_to_naics_dict, True),
            return_dtype=pl.Utf8
        ).alias("IND_mapped")
    )
    
    # Replace the original IND column with the mapped values
    data = data.drop("IND")
    data = data.rename({"IND_mapped": "IND"})
    
    # Log mapping results
    final_ind_codes = data["IND"].n_unique()
    logging.info(f"After mapping to NAICS, now have {final_ind_codes} unique industry codes.")
    
    return data


def create_aggregator(path):
    """
    Create an aggregator for SOC (Standard Occupational Classification) data.
    (Uses pandas to read Excel then converts to Polars.)
    """
    logging.info(f"Creating SOC aggregator from file: {path}")
    soc_2018_struct = pl.from_pandas(pd.read_excel(path, skiprows=7))
    group_soc_data = pl.DataFrame()
    major_occ = soc_2018_struct.select("Major Group").unique().to_series().to_list()

    for mo in major_occ:
        if not isinstance(mo, str):
            continue
        minor_group = (
            soc_2018_struct
                    .filter(pl.col("Minor Group").str.starts_with(mo[:2]))
                    .select("Minor Group")
                    .unique()
                    .to_series()
                    .to_list()
                )
        for mg in minor_group:
            broad_group = (
                soc_2018_struct
                    .filter(pl.col("Broad Group").str.starts_with(mg[:4]))
                    .select("Broad Group")
                    .unique()
                    .to_series()
                    .to_list()
            )
            for bg in broad_group:
                detailed_occupation = (
                    soc_2018_struct
                        .filter(pl.col("Detailed Occupation").str.starts_with(bg[:6]))
                        .select("Detailed Occupation")
                        .unique()
                        .to_series()
                        .to_list()
                )
                if len(detailed_occupation) == 0:
                    continue
                new_df = pl.DataFrame({
                    "Detailed Occupation": detailed_occupation,
                    "Broad Group": [bg] * len(detailed_occupation),
                    "Minor Group": [mg] * len(detailed_occupation),
                    "Major Group": [mo] * len(detailed_occupation)
                })
                group_soc_data = pl.concat([group_soc_data, new_df], how="vertical")
    logging.info(f"SOC aggregator created with {group_soc_data.shape[0]} rows.")
    return group_soc_data


def convert_using_cw(code, cw, keep_original=True, return_type = str):
    """
    Convert a code using a code-to-value dictionary.
    """
    if code not in cw:
        final_code =  str(code) if keep_original else ""
    else:
        final_code = str(cw[code])

    if return_type == float:
        if final_code == "":
            return np.nan
        else:
            return float(final_code)
    else:
        return final_code

def modify_occupation_codes(data, aggregator, cps_occ_soc_cw, occ_col="OCC", threshold=2):
    """
    Modify the occupation codes for CPS data.
    CPS uses OCC codes which are 4-digit codes that need to be mapped to SOC codes using a crosswalk.
    """
    logging.info("Starting modify_occupation_codes function.")
    logging.info(f"Data shape before occupation code modification: {data.shape}")

    # Group the crosswalk data by CPS occupation code to identify codes with multiple SOC mappings
    cps_to_soc_mappings = cps_occ_soc_cw.group_by(
        occ_col).agg(pl.col("SOC").alias("SOC"))

    # Create a dictionary to store the aggregated mappings
    cps_to_soc_dict = {}
    multiple_mappings_count = 0
    
    # Process each CPS code and determine the appropriate aggregation level for multiple mappings
    for row in cps_to_soc_mappings.iter_rows(named=True):
        cps_code = str(row[occ_col])
        soc_codes = row["SOC"]

        # If there's only one SOC code, use it directly
        if len(soc_codes) == 1:
            cps_to_soc_dict[cps_code] = str(soc_codes[0])
            continue
            
        # For multiple mappings, find the common prefix at the appropriate level
        multiple_mappings_count += 1
        
        # Check for common prefixes at different levels
        # Try detailed level first (full code)
        if all(code == soc_codes[0] for code in soc_codes):
            cps_to_soc_dict[cps_code] = str(soc_codes[0])
            continue
            
        # Try broad level (xx-xxxx)
        broad_prefixes = [code[:6] for code in soc_codes if len(code) >= 6]
        if len(set(broad_prefixes)) == 1 and len(broad_prefixes) == len(soc_codes):
            aggregated_code = broad_prefixes[0] + "00"
            cps_to_soc_dict[cps_code] = aggregated_code
            # logging.info(f"CPS code {cps_code} maps to multiple SOC codes: {soc_codes}. Aggregated to broad level: {aggregated_code}")
            continue
            
        # Try minor level (xx-xx)
        minor_prefixes = [code[:4] for code in soc_codes if len(code) >= 4]
        if len(set(minor_prefixes)) == 1 and len(minor_prefixes) == len(soc_codes):
            aggregated_code = minor_prefixes[0] + "00"
            cps_to_soc_dict[cps_code] = aggregated_code
            # logging.info(f"CPS code {cps_code} maps to multiple SOC codes: {soc_codes}. Aggregated to minor level: {aggregated_code}")
            continue
            
        # Default to major level (xx)
        major_prefixes = [code[:2] for code in soc_codes if len(code) >= 2]
        if len(set(major_prefixes)) == 1 and len(major_prefixes) == len(soc_codes):
            aggregated_code = major_prefixes[0] + "-0000"
            cps_to_soc_dict[cps_code] = aggregated_code
            # logging.info(f"CPS code {cps_code} maps to multiple SOC codes: {soc_codes}. Aggregated to major level: {aggregated_code}")
            continue
            
        # If no common prefix, log a warning and use the first code
        # logging.warning(f"CPS code {cps_code} maps to multiple SOC codes with no common prefix: {soc_codes}. Using first code.")
        cps_to_soc_dict[cps_code] = str(soc_codes[0])
    
    logging.info(f"Created crosswalk dictionary with {len(cps_to_soc_dict)} mappings.")
    logging.info(f"Found {multiple_mappings_count} CPS codes with multiple SOC mappings.")

    # Apply the crosswalk to map OCC codes to SOC codes
    # First strip whitespace from OCC codes
    data = data.with_columns(pl.col(occ_col).str.strip_chars())

    # Log number of OCC codes before mapping
    occ_codes_before = data[occ_col].n_unique()
    logging.info(f"Found {occ_codes_before} unique occupation codes before mapping.")

    #! For some reason when constructing the crosswalk some OCC codes have an extra "0" at the end, so we remove it (codes should be 7 characters long XX-XXXX)
    #> TODD: Fix the crosswalk to avoid this issue in the future
    for key in list(cps_to_soc_dict.keys()):
        cps_to_soc_dict[key] = cps_to_soc_dict[key][:7]  # Ensure all codes are 7 characters long

    # Map OCC codes to SOC codes
    data = data.with_columns(
        pl.col(occ_col).map_elements(
            lambda x: convert_using_cw(x, cps_to_soc_dict, True),
            return_dtype=pl.Utf8
        ).alias(occ_col)
    )

    # Log the results of the mapping
    mapped_count = data.filter(pl.col(occ_col).str.contains("-")).shape[0]
    total_count = data.shape[0]
    logging.info(f"Successfully mapped {mapped_count} out of {total_count} occupation codes ({mapped_count/total_count:.2%}).")
    
    # Identify and log unconverted occupation codes
    unconverted_codes = data.filter(~pl.col(occ_col).str.contains("-"))
    
    # Extract lists from aggregator for membership tests
    detailed_list = aggregator["Detailed Occupation"].to_list()
    broad_list = aggregator["Broad Group"].to_list()
    minor_list = aggregator["Minor Group"].to_list()
    major_list = aggregator["Major Group"].to_list()

    # Initialize a group classification column
    data = data.with_columns(pl.lit(None).cast(pl.String).alias(occ_col + "_group"))
    data = data.with_columns(
        pl.when(pl.col(occ_col).is_in(detailed_list))
        .then( pl.lit( "detailed" ) )
        .when(pl.col(occ_col).is_in(broad_list))
        .then( pl.lit( "broad" ))
        .when(pl.col(occ_col).is_in(minor_list))
        .then( pl.lit( "minor" ) )
        .when(pl.col(occ_col).is_in(major_list))
        .then( pl.lit( "major" ) )
        .otherwise( pl.lit( "none" ) )
        .cast(pl.String)  # Explicitly cast to string
        .alias(occ_col + "SOC_group")
    )
    
    # Drop rows with unclassified occupation codes
    before_drop = data.shape[0]
    data = data.filter(pl.col(occ_col + "SOC_group") != "none")

    logging.info(f"Dropped rows with unclassified {occ_col}. Rows reduced from {before_drop} to {data.shape[0]}.")

    # Create mapping dictionaries for occupation codes
    soc_2018_dict_broad       = dict(zip(aggregator["Detailed Occupation"], aggregator["Broad Group"]))
    soc_2018_dict_minor       = dict(zip(aggregator["Detailed Occupation"], aggregator["Minor Group"]))
    soc_2018_dict_broad_minor = dict(zip(aggregator["Broad Group"], aggregator["Minor Group"]))

    data = data.with_columns(
        pl.when(pl.col(occ_col + "SOC_group") == "detailed")
        .then(pl.col(occ_col))
        .otherwise(pl.lit(""))
        .alias(occ_col + "SOC_detailed")
    )

    data = data.with_columns(
        pl.when(pl.col(occ_col + "SOC_group") == "broad")
        .then(pl.col(occ_col))
        .when(pl.col(occ_col + "SOC_group") == "detailed")
        .then(pl.col(occ_col).replace_strict(soc_2018_dict_broad, default=pl.lit("")))
        .otherwise(pl.lit(""))
        .alias(occ_col + "SOC_broad")
    )

    data = data.with_columns(
        pl.when(pl.col(occ_col + "SOC_group") == "minor")
        .then(pl.col(occ_col))
        .when(pl.col(occ_col + "SOC_group") == "broad")
        .then(pl.col(occ_col).replace_strict(soc_2018_dict_broad_minor, default=pl.lit("")))
        .when(pl.col(occ_col + "SOC_group") == "detailed")
        .then(pl.col(occ_col).replace_strict(soc_2018_dict_minor, default=pl.lit("")))
        .otherwise(pl.lit(""))
        .alias(occ_col + "SOC_minor")
    )

    logging.info(f"modify_occupation_codes complete. Data shape is now: {data.shape}")
    return data


def add_state_and_censusdiv(data, state_censusdiv):
    """
    Add state identifiers and census division information using STATEFIP.
    """
    logging.info("Starting add_state_and_censusdiv function.")
    
    # Ensure STATEFIP is properly formatted as 2-digit string
    data = data.with_columns(pl.col("STATEFIP").str.zfill(2).alias("state_fips"))
    
    # Load state census divisions data with all columns as strings
    state_censusdiv_data = pl.read_csv(state_censusdiv, schema_overrides={
        "state_name": pl.Utf8,
        "state_fips": pl.Utf8,
        "censusdiv_name": pl.Utf8,
        "censusdiv": pl.Utf8,
        "censusreg": pl.Utf8,
        "censusreg_name": pl.Utf8
    })
    logging.info(f"Read state census divisions from {state_censusdiv} with {state_censusdiv_data.shape[0]} rows.")
    
    # Create mapping from state_fips to censusdiv
    state_to_censusdiv_dict = dict(zip(
        state_censusdiv_data["state_fips"].to_list(),
        state_censusdiv_data["censusdiv"].to_list()
    ))
    
    # Create mapping from state_fips to state_name
    state_to_name_dict = dict(zip(
        state_censusdiv_data["state_fips"].to_list(),
        state_censusdiv_data["state_name"].to_list()
    ))
    
    # Map state_fips to censusdiv and state_name
    data = data.with_columns([
        pl.col("state_fips").map_elements(
            lambda x: convert_using_cw(x, state_to_censusdiv_dict, False),
            return_dtype=pl.Utf8
        ).alias("censusdiv"),
        pl.col("state_fips").map_elements(
            lambda x: convert_using_cw(x, state_to_name_dict, False),
            return_dtype=pl.Utf8
        ).alias("state_name")
    ])
    
    logging.info("Added state_fips, state_name, and censusdiv columns.")
    return data


def telework_index(data, soc_aggregator, how="my_index", occ_col="OCCSOC", path_to_remote_work_index=WFH_INDEX):
    """
    Calculate the telework index for different occupation codes.
    """
    logging.info("Starting telework_index function.")
    if how == "my_index":
        logging.info(f"Using my_index method. Loading telework index from {path_to_remote_work_index}.")
        wfh_index_df = pl.read_csv(path_to_remote_work_index)
        # Average the remote work index for each occupation code
        clasification = wfh_index_df.group_by("OCC_CODE").agg(pl.mean("ESTIMATE_WFH_ABLE"))
        # Merge with the SOC aggregator (joining on Detailed Occupation)
        clasification = clasification.join(soc_aggregator, left_on="OCC_CODE", right_on="Detailed Occupation", how="inner")
        clasification = clasification.rename({"ESTIMATE_WFH_ABLE": "TELEWORKABLE"})
    elif how == "dn_index":
        logging.info("Using dn_index method. Loading telework classification data.")
        clasification = pl.read_csv("https://raw.githubusercontent.com/jdingel/DingelNeiman-workathome/master/occ_onet_scores/output/occupations_workathome.csv")
        # Rename columns to uppercase
        for col in clasification.columns:
            clasification = clasification.rename({col: col.upper()})
        # Split ONETSOCCODE into OCC_CODE and ONET_DETAIL
        clasification = (
            clasification.with_columns(
                pl.col("ONETSOCCODE").str.split(".").alias("split_col")).with_columns(
                    [
                        pl.col("split_col").arr.get(0).alias("OCC_CODE"),
                        pl.col("split_col").arr.get(1).alias("ONET_DETAIL")
                    ]
                ).drop("split_col")
        )
        clasification = clasification.filter(pl.col("ONET_DETAIL") == "00")
        clasification = clasification.select(["OCC_CODE", "TITLE", "TELEWORKABLE"])
        logging.info(f"Filtered telework data to ONET_DETAIL=='00'. Shape: {clasification.shape}")
    else:
        raise ValueError("Invalid value for 'how' parameter. Must be 'my_index' or 'dn_index'.")

    logging.info("Loaded and formatted telework classification data.")
    
    # For detailed, broad, and minor levels, we average TELEWORKABLE grouping by the respective level
    data = data.join(clasification, left_on="OCCSOC_detailed", right_on="OCC_CODE", how="left")
    data = data.rename({"TELEWORKABLE": "TELEWORKABLE_OCCSOC_detailed"})
    data = data.join(clasification.group_by("Broad Group").agg(pl.mean("TELEWORKABLE")), left_on="OCCSOC_broad", right_on="Broad Group", how="left")
    data = data.rename({"TELEWORKABLE": "TELEWORKABLE_OCCSOC_broad"})
    data = data.join(clasification.group_by("Minor Group").agg(pl.mean("TELEWORKABLE")), left_on="OCCSOC_minor", right_on="Minor Group", how="left")
    data = data.rename({"TELEWORKABLE": "TELEWORKABLE_OCCSOC_minor"})

    logging.info("Assigned telework indices to detailed, broad, and minor occupation codes.")
    return data


def remote_work(data):
    """
    Calculate the work-from-home (WFH) index and related variables for each record.
    For CPS data, we use TELWRKHR (telework hours) and TELWRKPAY if available.
    """
    logging.info("Starting remote_work function.")

    # # --- TELWRKHR detailed summary by year and TELWRKPAY category (2022 and after) ---
    #Note: This is just my justification to set TELWRKHR to 0 if TELWRKPAY == "2" (NO) and TELWRKHR != 0
    # if "TELWRKHR" in data.columns and "TELWRKPAY" in data.columns:
    #     # Focus on 2022 and after
    #     df = data.filter(pl.col("YEAR") >= 2022)
    #     # Define categories for TELWRKPAY
    #     telwrkpay_map = {"0": "NIU", "1": "YES", "2": "NO"}
    #     for year in sorted(df["YEAR"].unique()):
    #         print(f"\nYear: {year}")
    #         df_year = df.filter(pl.col("YEAR") == year)
    #         for cat_code, cat_name in telwrkpay_map.items():
    #             df_cat = df_year.filter(pl.col("TELWRKPAY") == cat_code)
    #             total = df_cat.height
    #             if total == 0:
    #                 print(f"  {cat_name}: No data")
    #                 continue
    #             nulls = df_cat.filter(pl.col("TELWRKHR").is_null()).height
    #             eq_0 = df_cat.filter(pl.col("TELWRKHR") == 0).height
    #             eq_999 = df_cat.filter(pl.col("TELWRKHR") == 999).height
    #             between = df_cat.filter((pl.col("TELWRKHR") > 0) & (pl.col("TELWRKHR") < 999)).height
    #             print(f"  {cat_name}:")
    #             print(f"    Nulls:   {nulls/total:.1%} ({nulls})")
    #             print(f"    == 0:    {eq_0/total:.1%} ({eq_0})")
    #             print(f"    == 999:  {eq_999/total:.1%} ({eq_999})")
    #             print(f"    0 < x < 999: {between/total:.1%} ({between})")
    # else:
    #     print("TELWRKHR or TELWRKPAY column not found in data.")

    if "TELWRKHR" in data.columns and "TELWRKPAY" in data.columns and "UHRSWORK1" in data.columns:
        # 1. Drop all NIU (TELWRKPAY == "0")
        shape_before = data.shape
        data = data.filter(pl.col("TELWRKPAY") != "0")
        logging.info(f"Dropped NIU (TELWRKPAY == '0'). Rows reduced from {shape_before[0]} to {data.shape[0]}.")

        # 2. For TELWRKPAY == "2" (NO) and TELWRKHR != 0, set TELWRKHR to 0
        data = data.with_columns(
            pl.when((pl.col("TELWRKPAY") == "2") & (pl.col("TELWRKHR") != 0))
            .then(0)
            .otherwise(pl.col("TELWRKHR"))
            .alias("TELWRKHR")
        )

        # 3. Create WFH: 1 if TELWRKHR > 0, else 0
        data = data.with_columns(
            (pl.col("TELWRKHR") > 0).cast(pl.Int64).alias("WFH")
        )

        # 4. Calculate ALPHA = TELWRKHR / UHRSWORK1
        data = data.with_columns(
            (pl.col("TELWRKHR") / pl.col("UHRSWORK1")).alias("ALPHA")
        )

        # 5. Ensure 0 <= ALPHA <= 1, set ALPHA > 1 to 1
        data = data.with_columns(
            pl.when(pl.col("ALPHA") > 1)
            .then(1.0)
            .otherwise(
                pl.when(pl.col("ALPHA") < 0)
                .then(0.0)
                .otherwise(pl.col("ALPHA"))
            )
            .alias("ALPHA")
        )

        # 6. Create dummies for FULL_INPERSON (ALPHA == 0), FULL_REMOTE (ALPHA == 1), HYBRID (0 < ALPHA < 1)
        data = data.with_columns([
            (pl.col("ALPHA") == 0).cast(pl.Int64).alias("FULL_INPERSON"),
            (pl.col("ALPHA") == 1).cast(pl.Int64).alias("FULL_REMOTE"),
            ((pl.col("ALPHA") > 0) & (pl.col("ALPHA") < 1)).cast(pl.Int64).alias("HYBRID"),
        ])

        logging.info("WFH, ALPHA, FULL_INPERSON, FULL_REMOTE, HYBRID variables created.")
    else:
        # If required columns are not available, create placeholders
        logging.warning("Required columns for remote_work not found. Creating placeholder columns for WFH, ALPHA, FULL_INPERSON, FULL_REMOTE, HYBRID.")

    if all(col in data.columns for col in ["WFH", "FULL_INPERSON", "FULL_REMOTE", "HYBRID"]):
        print("\nYearly statistics for WFH dummies:")
        stats = (
            data.group_by("YEAR")
            .agg([
                pl.col("WFH").mean().alias("WFH_MEAN"),
                pl.col("FULL_INPERSON").mean().alias("FULL_INPERSON_MEAN"),
                pl.col("FULL_REMOTE").mean().alias("FULL_REMOTE_MEAN"),
                pl.col("HYBRID").mean().alias("HYBRID_MEAN"),
                pl.count().alias("N")
            ])
            .sort("YEAR")
        )
        print(stats)

    return data


def save_data(data, path):
    """
    Save the data to a CSV file.
    """
    logging.info(f"Saving data to {path}. Final data shape: {data.shape}")
    # Before saving make sure that all variables are uppercase   
    data = data.rename({col: col.upper() for col in data.columns})
    data.write_csv(path)
    logging.info("Data saved successfully.")

# %%
# Function to run the entire data processing pipeline
def cps_data_proc(min_year=2013, max_year=None, data_file_name="cps_00037.csv.gz", return_data=False):
    """
    Process CPS (Current Population Survey) data through the entire pipeline.

    Parameters:
        min_year (int): Minimum year to include in data processing (default: 2013)
        max_year (int): Maximum year to include in data processing (default: None)
        data_file_name (str): Name of the CPS data file to process (default: "cps_00032.csv.gz")
        return_data (bool): Whether to return the processed data (default: False)
        
    Returns:
        polars.DataFrame: Processed data if return_data is True, otherwise None
    """
    # Set default values if empty strings or None are provided
    if min_year is None or min_year == "":
        min_year = 2013

    if max_year is None or max_year == "":
        max_year = None

    if data_file_name is None or data_file_name == "":
        data_file_name = "cps_00037.csv.gz"

    # Record the start time for performance tracking
    start_time = time.time()
    logging.info("CPS processing script started.")

    # Construct the full path to the input data file
    PATH_CPS_DATA = os.path.join(DATA_DIR, data_file_name)
    logging.info(f"Processing file {data_file_name} from {PATH_CPS_DATA}")

    # Step 1: Read raw CPS data and SOC occupational code structure
    data = read_cps_data(PATH_CPS_DATA, min_year=min_year, max_year=max_year)

    soc_aggregator = create_aggregator(SOC_AGGREGATOR)

    # Step 2: Filter data based on work hours, wage, and employment class
    data = filter_cps_data(data, hours_worked_lim=35, wage_lim=5, class_of_worker=['20', '21', '22', '23', '24', '25', '27', '28'])

    # Step 3: Clean and standardize industry codes
    data = modify_industry_codes(data)

    # Step 4: Standardize and classify occupation codes at different levels
    # Read a crosswalk for CPS OCC codes to SOC codes
    if os.path.exists(CPS_OCC_SOC_CW):
        logging.info(f"Loading CPS OCC to SOC crosswalk from {CPS_OCC_SOC_CW}.")
        cps_occ_soc_cw = pl.read_csv(CPS_OCC_SOC_CW)   
    data = modify_occupation_codes(data, soc_aggregator, cps_occ_soc_cw, occ_col="OCC")

    # Step 5: Add state identifiers and census division information
    data = add_state_and_censusdiv(data, STATE_CENSUSDIV)

    # Step 6: Calculate telework capability indices for each occupation
    data = telework_index(data, soc_aggregator, occ_col="OCCSOC")

    # # Step 7: Create work-from-home indicator based on telework hours
    data = remote_work(data)

    # Step 8: Save the processed data, selecting only the specified columns
    output_path = os.path.join(PROCESSED_DATA_DIR, data_file_name.replace(".csv.gz", ".csv"))
    
    # Reanme all columns to uppercase
    data.columns = [col.upper() for col in data.columns]

    export_data = data.select(COLS_TO_EXPORT)
    n_obs = export_data.height
    years = export_data["YEAR"].unique().to_list()
    min_year_export = min(years) if years else None
    max_year_export = max(years) if years else None
    n_workers = export_data["CPSIDP"].n_unique() if "CPSIDP" in export_data.columns else None

    logging.info(f"Export summary: Observations remaining: {n_obs:,}")
    logging.info(f"Export summary: Time frame remaining: {min_year_export} to {max_year_export}")
    logging.info(f"Export summary: Unique workers remaining (CPSIDP): {n_workers:,}")

    save_data(export_data, output_path)

    # Calculate and log the total processing time
    end_time = time.time()
    total_time = end_time - start_time
    logging.info(f"CPS processing script finished successfully in {total_time:.2f} seconds.")

    # Return the processed data if requested
    if return_data:
        data_return  = data.select(COLS_TO_EXPORT)
        data_return = data_return.rename({col: col.upper() for col in data_return.columns})
        return data_return


# %% Main
if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='Process CPS data.')
    parser.add_argument('--min_year', type=int, default=2013, help='Minimum year to filter the data')
    parser.add_argument('--max_year', type=int, default=None, help='Maximum year to filter the data')
    parser.add_argument('--data_file_name', type=str, default="cps_00037.csv.gz", help='Name of the CPS data file to process')

    args = parser.parse_args()

    # Run the data processing pipeline
    cps_data_proc(min_year=args.min_year, max_year=args.max_year, data_file_name=args.data_file_name)
