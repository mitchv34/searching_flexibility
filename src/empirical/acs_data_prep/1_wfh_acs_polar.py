# %% Import packages and define constants
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
DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'acs')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'acs')

SOC_AGGREGATOR = os.path.join(BASE_DIR, 'data', 'aux_and_croswalks', 'soc_structure_2018.xlsx')
PUMA_CROSSWALK = os.path.join(BASE_DIR, 'data', 'aux_and_croswalks', "puma_to_cbsa.csv")
STATE_CENSUSDIV = os.path.join(BASE_DIR, 'data', 'aux_and_croswalks', "state_census_divisions.csv")
WFH_INDEX = os.path.join(BASE_DIR, 'data', 'results', 'wfh_estimates.csv')

COLS_TO_EXPORT = [
    "unique_person_id",  # Unique identifier for each person
    'YEAR', 'PERWT', 'AGE', 'SEX', 'RACE', 'HISPAN', 'RACED', 'EDUC', 'EDUCD', 'CLASSWKRD','WAGE', 'INDNAICS', 'cbsa20', 'WFH',
    'OCCSOC_detailed', 'OCCSOC_broad', 'OCCSOC_minor',
    'TELEWORKABLE_OCCSOC_detailed', 'TELEWORKABLE_OCCSOC_broad', 'TELEWORKABLE_OCCSOC_minor',
    'state_fips', 'state_name', 'censusdiv'
]

# Ensure directories exist
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# %% Define functions

def read_acs_data(path, min_year=None, max_year=None):
    """
    Read ACS data from a CSV file using Polars and filter by year.
    """
    logging.info(f"Reading ACS data from {path}")
    # Define data types (Polars types)
    dtypes = {
        "YEAR": pl.Int64,
        "STATEFIP": pl.Utf8,
        "PUMA": pl.Utf8,
        "PERWT": pl.Float64,
        "AGE": pl.Int64,
        "SEX": pl.Int64,
        "RACE": pl.Utf8,
        "HISPAN": pl.Int64,
        "RACED": pl.Utf8,
        "EDUC": pl.Utf8,
        "EDUCD": pl.Utf8,
        "CLASSWKR": pl.Utf8,
        "CLASSWKRD": pl.Utf8,
        "INDNAICS": pl.Utf8,
        "OCCSOC": pl.Utf8,
        "TRANWORK": pl.Utf8,
        "TRANTIME": pl.Float64,
        "INCWAGE": pl.Float64,
        "UHRSWORK": pl.Float64,
        "INCTOT": pl.Float64
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


def filter_acs_data(data, **kwargs):
    """
    Filter ACS data based on working hours, wage, and class of worker.
    """
    logging.info("Starting filter_acs_data function.")
    hours_worked_lim = kwargs.get("hours_worked_lim", 35)
    if "UHRSWORK" in data.columns:
        initial_shape = data.shape
        data = data.filter(pl.col("UHRSWORK") > hours_worked_lim)
        logging.info(f"Filtered by UHRSWORK > {hours_worked_lim}. Rows reduced from {initial_shape[0]} to {data.shape[0]}.")
    else:
        logging.warning("Column 'UHRSWORK' not found. Skipping filter based on minimum hours worked.")

    if "INCWAGE" in data.columns and "UHRSWORK" in data.columns:
        data = data.with_columns(
                WAGE=pl.col("INCWAGE") / (pl.col("UHRSWORK") * 52 )
        )
        logging.info("Calculated hourly wage and stored in 'WAGE' column.")
        data = data.drop(["INCWAGE", "UHRSWORK"])
        logging.info("Dropped columns 'INCWAGE' and 'UHRSWORK'.")

    wage_lim = kwargs.get("wage_lim", 5)
    if "WAGE" in data.columns:
        initial_shape = data.shape
        data = data.filter(pl.col("WAGE") > wage_lim)
        logging.info(f"Filtered by WAGE > {wage_lim}. Rows reduced from {initial_shape[0]} to {data.shape[0]}.")
    else:
        logging.warning("Column 'WAGE' not found. Skipping filter based on minimum wage.")

    class_of_worker = kwargs.get("class_of_worker", ['2'])
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
        # Filter out missing values (code 9)
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
    Modify industry codes by stripping whitespace and removing unwanted codes.
    """
    logging.info("Starting modify_industry_codes function.")
    data = data.with_columns(pl.col("INDNAICS").str.strip_chars())
    initial_shape = data.shape
    data = data.filter(pl.col("INDNAICS") != "0")
    logging.info(f"Removed rows with INDNAICS=='0'. Rows reduced from {initial_shape[0]} to {data.shape[0]}.")

    military_codes = ["928110P1", "928110P2", "928110P3", "928110P4", "928110P5", "928110P6", "928110P7"]
    data = data.filter(~pl.col("INDNAICS").is_in(military_codes))
    logging.info("Removed military industry codes.")

    data = data.filter(pl.col("INDNAICS") != "999920")
    logging.info("Removed unemployed codes (INDNAICS=='999920').")
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
        minor_group = (soc_2018_struct
                       .filter(pl.col("Minor Group").str.starts_with(mo[:2]))
                       .select("Minor Group")
                       .unique()
                       .to_series()
                       .to_list())
        for mg in minor_group:
            broad_group = (soc_2018_struct
                           .filter(pl.col("Broad Group").str.starts_with(mg[:4]))
                           .select("Broad Group")
                           .unique()
                           .to_series()
                           .to_list())
            for bg in broad_group:
                detailed_occupation = (soc_2018_struct
                                       .filter(pl.col("Detailed Occupation").str.starts_with(bg[:6]))
                                       .select("Detailed Occupation")
                                       .unique()
                                       .to_series()
                                       .to_list())
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
# %%
def modify_occupation_codes(data, aggregator, occ_col="OCCSOC", threshold=2):
    """
    Modify the occupation codes.
    """
    logging.info("Starting modify_occupation_codes function.")
    data = data.filter(pl.col(occ_col).is_not_null())
    data = data.filter(pl.col(occ_col) != "nan")

    # Check if codes are already in "XX-XXXX" format
    if data.filter(~pl.col(occ_col).str.contains("-")).is_empty():
        logging.info(f"Column {occ_col} already formatted.")
    else:
        data = data.with_columns(
            (pl.col(occ_col).str.slice(0, 2) + "-" + pl.col(occ_col).str.slice(2)).alias(occ_col)
        )
        logging.info(f"Formatted {occ_col} to standard 'XX-XXXX' format.")

    before_filter = data.shape[0]
    data = data.filter(
        (pl.col(occ_col).str.count_matches("X", literal=True) + 
        pl.col(occ_col).str.count_matches("Y", literal=True)) <= threshold
    )

    logging.info(f"Removed rows with more than {threshold} X/Y characters. Rows reduced from {before_filter} to {data.shape[0]}.")

    data = data.with_columns(pl.col(occ_col).str.replace("X", "0").str.replace("Y", "0"))
    logging.info("Replaced X and Y in occupation codes with 0.")

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
        .alias(occ_col + "_group")
    )
    # Drop rows with unclassified occupation codes
    before_drop = data.shape[0]
    data = data.filter(pl.col(occ_col + "_group") != "none")

    logging.info(f"Dropped rows with unclassified {occ_col}. Rows reduced from {before_drop} to {data.shape[0]}.")

    # Create mapping dictionaries for occupation codes
    # Create dictionaries for mapping
    soc_2018_dict_broad       = dict(zip(aggregator["Detailed Occupation"], aggregator["Broad Group"]))
    soc_2018_dict_minor       = dict(zip(aggregator["Detailed Occupation"], aggregator["Minor Group"]))
    soc_2018_dict_broad_minor = dict(zip(aggregator["Broad Group"], aggregator["Minor Group"]))

    data = data.with_columns(
        pl.when(pl.col(occ_col + "_group") == "detailed")
        .then(pl.col(occ_col))
        .otherwise(pl.lit(""))
        .alias(occ_col + "_detailed")
    )

    data = data.with_columns(
        pl.when(pl.col(occ_col + "_group") == "broad")
        .then(pl.col(occ_col))
        .when(pl.col(occ_col + "_group") == "detailed")
        .then(pl.col(occ_col).replace_strict(soc_2018_dict_broad, default=pl.lit("")))
        .otherwise(pl.lit(""))
        .alias(occ_col + "_broad")
    )

    data = data.with_columns(
        pl.when(pl.col(occ_col + "_group") == "minor")
        .then(pl.col(occ_col))
        .when(pl.col(occ_col + "_group") == "broad")
        .then(pl.col(occ_col).replace_strict(soc_2018_dict_broad_minor, default=pl.lit("")))
        .when(pl.col(occ_col + "_group") == "detailed")
        .then(pl.col(occ_col).replace_strict(soc_2018_dict_minor, default=pl.lit("")))
        .otherwise(pl.lit(""))
        .alias(occ_col + "_minor")
    )

    logging.info(f"modify_occupation_codes complete. Data shape is now: {data.shape}")
    return data


# %%
def aggregate_puma_to_cbsa(data, puma_crosswalk):
    """
    Aggregate PUMA codes to CBSA codes.
    """
    logging.info("Starting aggregate_puma_to_cbsa function.")
    data = data.with_columns(pl.col("STATEFIP").str.zfill(2))
    data = data.with_columns(pl.col("PUMA").str.zfill(5))
    data = data.with_columns((pl.col("STATEFIP") + "-" + pl.col("PUMA")).alias("state_puma"))
    data = data.drop(["STATEFIP", "PUMA"])
    logging.info("Formatted and combined STATEFIP and PUMA into state_puma.")

    puma_to_cbsa = pl.read_csv(puma_crosswalk, schema_overrides={"state_puma": pl.Utf8, "cbsa20": pl.Utf8})
    logging.info(f"Read PUMA crosswalk from {puma_crosswalk} with {puma_to_cbsa.shape[0]} rows.")
    puma_to_cbsa_dict = dict(zip(puma_to_cbsa["state_puma"].to_list(), puma_to_cbsa["cbsa20"].to_list()))
    data = data.with_columns(
        pl.col("state_puma").map_elements(
            lambda x: convert_using_cw(x, puma_to_cbsa_dict, False), return_dtype=pl.Utf8
            )
        .alias("cbsa20")
    )
    logging.info("Aggregated state_puma to CBSA codes.")
    return data

# %%
def add_state_and_censusdiv(data, puma_crosswalk, state_censusdiv):
    """
    Add state and census division columns using cbsa20 to map back to state_puma,
    then extract state and map to census division.
    """
    logging.info("Starting add_state_and_censusdiv function.")
    
    # Load the PUMA crosswalk to create reverse mapping from cbsa20 to state_puma
    puma_to_cbsa = pl.read_csv(puma_crosswalk, schema_overrides={
        "state_puma": pl.Utf8, 
        "cbsa20": pl.Utf8
    })
    logging.info(f"Read PUMA crosswalk from {puma_crosswalk} with {puma_to_cbsa.shape[0]} rows.")
    
    # Create mapping from cbsa20 to state_puma (taking the first occurrence for each cbsa20)
    cbsa_to_state_puma = puma_to_cbsa.group_by("cbsa20").agg(
        pl.col("state_puma").first()
    )
    cbsa_to_state_puma_dict = dict(zip(
        cbsa_to_state_puma["cbsa20"].to_list(), 
        cbsa_to_state_puma["state_puma"].to_list()
    ))
    
    # Map cbsa20 back to state_puma
    data = data.with_columns(
        pl.col("cbsa20").map_elements(
            lambda x: convert_using_cw(x, cbsa_to_state_puma_dict, False), 
            return_dtype=pl.Utf8
        ).alias("state_puma_mapped")
    )
    
    # Extract state code (first 2 characters) from state_puma
    data = data.with_columns(
        pl.col("state_puma_mapped").str.slice(0, 2).alias("state_fips")
    )
    
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
    
    # Drop the temporary state_puma_mapped column
    data = data.drop("state_puma_mapped")
    
    logging.info("Added state_fips, state_name, and censusdiv columns.")
    return data

# %%
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
        clasification = (clasification.with_columns(pl.col("ONETSOCCODE").str.split(".").alias("split_col"))
                         .with_columns([
                             pl.col("split_col").arr.get(0).alias("OCC_CODE"),
                             pl.col("split_col").arr.get(1).alias("ONET_DETAIL")
                         ])
                         .drop("split_col"))
        clasification = clasification.filter(pl.col("ONET_DETAIL") == "00")
        clasification = clasification.select(["OCC_CODE", "TITLE", "TELEWORKABLE"])
        logging.info(f"Filtered telework data to ONET_DETAIL=='00'. Shape: {clasification.shape}")
    else:
        raise ValueError("Invalid value for 'how' parameter. Must be 'my_index' or 'dn_index'.")

    logging.info("Loaded and formatted telework classification data.")
    # telework_dict = dict(zip(clasification["OCC_CODE"].to_list(), clasification["TELEWORKABLE"].to_list()))
    # data = data.with_columns(
    #     pl.col("OCCSOC").map_elements(lambda x: convert_using_cw(x, telework_dict, False, return_type=float), return_dtype=pl.Float64)
    #     .alias("TELEWORKABLE_OCCSOC")
    # )    
    # For detailed, broad, and minor levels, we average TELEWORKABLE_OCCSOC grouping by the respective level
    data = data.join(clasification, left_on="OCCSOC_detailed", right_on="OCC_CODE", how="left")
    data = data.rename({"TELEWORKABLE": "TELEWORKABLE_OCCSOC_detailed"})
    data = data.join(clasification.group_by("Broad Group").agg(pl.mean("TELEWORKABLE")), left_on="OCCSOC_broad", right_on="Broad Group", how="left")
    data = data.rename({"TELEWORKABLE": "TELEWORKABLE_OCCSOC_broad"})
    data = data.join(clasification.group_by("Minor Group").agg(pl.mean("TELEWORKABLE")), left_on="OCCSOC_minor", right_on="Minor Group", how="left")
    data = data.rename({"TELEWORKABLE": "TELEWORKABLE_OCCSOC_minor"})

    logging.info("Assigned telework indices to detailed, broad, and minor occupation codes.")
    return data


def wfh_index(data):
    """
    Calculate the work-from-home (WFH) index for each record.
    """
    logging.info("Starting wfh_index function.")
    data = data.with_columns((pl.col("TRANWORK") == "80").alias("WFH"))
    logging.info("WFH index calculated based on TRANWORK column.")
    return data


def save_data(data, path):
    """
    Save the data to a CSV file.
    """
    logging.info(f"Saving data to {path}. Final data shape: {data.shape}")
    # Before saving make sure that all variales are uppercase   
    data = data.rename({col: col.upper() for col in data.columns})
    data.write_csv(path)
    logging.info("Data saved successfully.")


# Function to run the entire data processing pipeline
def acs_data_proc(min_year=2013, max_year=None, data_file_name="usa_00137.csv.gz", return_data=False):
    """
    Process ACS (American Community Survey) data through the entire pipeline.
    
    Parameters:
        min_year (int): Minimum year to include in data processing (default: 2013)
        max_year (int): Maximum year to include in data processing (default: None)
        data_file_name (str): Name of the ACS data file to process (default: "usa_00137.csv.gz")
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
        data_file_name = "usa_00137.csv.gz"

    # Record the start time for performance tracking
    start_time = time.time()
    logging.info("Script started.")
    
    # Construct the full path to the input data file
    PATH_ACS_DATA = os.path.join(DATA_DIR, data_file_name)
    logging.info(f"Processing file {data_file_name} from {PATH_ACS_DATA}")

    # Step 1: Read raw ACS data and SOC occupational code structure
    data = read_acs_data(PATH_ACS_DATA, min_year=min_year)
    
    soc_aggregator = create_aggregator(SOC_AGGREGATOR)

    # CREATE UNIQUE IDENTIFIER AT THE VERY BEGINNING
    # This must be the first transformation after loading the data
    data = data.with_columns([
        # Create unique person identifier using SAMPLE + SERIAL + PERNUM
        pl.concat_str([
            pl.col("SAMPLE").cast(pl.Utf8),
            pl.lit("_"),
            pl.col("SERIAL").cast(pl.Utf8), 
            pl.lit("_"),
            pl.col("PERNUM").cast(pl.Utf8)
        ]).alias("unique_person_id")
    ])
    
    logging.info(f"✓ Created unique_person_id for {data.height:,} observations")
    
    # Verify uniqueness of the identifier
    unique_count = data.select(pl.col("unique_person_id")).n_unique()
    total_count = data.height
    
    if unique_count == total_count:
        logging.info(f"✓ Unique identifier verified: {unique_count:,} unique IDs for {total_count:,} observations")
    else:
        logging.info(f"⚠ WARNING: Only {unique_count:,} unique IDs for {total_count:,} observations")
        
    # ENSURE unique_person_id IS PRESERVED IN ALL SUBSEQUENT OPERATIONS
    # Add it to the list of columns to keep in all select operations
    
    # Step 2: Filter data based on work hours, wage, and employment class
    data = filter_acs_data(data, hours_worked_lim=35, wage_lim=5, class_of_worker=['2'])

    # Step 3: Clean and standardize industry codes
    data = modify_industry_codes(data)

    # Step 4: Standardize and classify occupation codes at different levels
    data = modify_occupation_codes(data, soc_aggregator)

    # Step 5: Convert PUMA (Public Use Microdata Area) codes to CBSA (Core Based Statistical Area)
    data = aggregate_puma_to_cbsa(data, PUMA_CROSSWALK)

    # Step 6: Add state identifiers and census division information
    data = add_state_and_censusdiv(data, PUMA_CROSSWALK, STATE_CENSUSDIV)

    # Step 7: Calculate telework capability indices for each occupation
    data = telework_index(data, soc_aggregator)

    # Step 8: Create work-from-home indicator based on transportation data
    data = wfh_index(data)

    # Step 9: Save the processed data, selecting only the specified columns
    output_path = os.path.join(PROCESSED_DATA_DIR, data_file_name.replace(".csv.gz", ".csv"))
    save_data(data.select(COLS_TO_EXPORT), output_path)

    # Calculate and log the total processing time
    end_time = time.time()
    total_time = end_time - start_time
    logging.info(f"Script finished successfully in {total_time:.2f} seconds.")

    # Return the processed data if requested
    if return_data:
        data_return  = data.select(COLS_TO_EXPORT)
        data_return = data_return.rename({col: col.upper() for col in data_return.columns})
        return data_return



# %% Main
if __name__ == "__main__":
    # # Parse arguments
    # parser = argparse.ArgumentParser(description='Process ACS data.')
    # parser.add_argument('--min_year', type=int, default="", help='Minimum year to filter the data')
    # parser.add_argument('--max_year', type=int, default="", help='Maximum year to filter the data')
    # parser.add_argument('--data_file_name', type=str, default="", help='Name of the ACS data file to process (default: "usa_00137.csv.gz")')

    # args = parser.parse_args()

    # # Set default values if empty strings or None are provided
    # if args.min_year is None or args.min_year == "":
    #     args.min_year = 2013
    # if args.max_year is None or args.max_year == "":
    #     args.max_year = None
    # if args.data_file_name is None or args.data_file_name == "":
    #     args.data_file_name = "usa_00137.csv.gz"

    # # Run the data processing pipeline
    # acs_data_proc(min_year=args.min_year, max_year=args.max_year, data_file_name=args.data_file_name)
    acs_data_proc()

