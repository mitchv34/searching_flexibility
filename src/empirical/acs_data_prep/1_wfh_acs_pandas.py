# %% Import packages and define constants
import os
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
# BASE_DIR = '../../../../'
BASE_DIR = '.'
DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'acs')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'acs')

SOC_AGGREGATOR = os.path.join(BASE_DIR, 'data', 'aux_and_croswalks', 'soc_structure_2018.xlsx')
PUMA_CROSSWALK = os.path.join(BASE_DIR, 'data', 'aux_and_croswalks', "puma_to_cbsa.csv")
WFH_INDEX = os.path.join(BASE_DIR, 'data', 'results', 'wfh_estimates.csv')
SKILL_VECTORS = os.path.join(BASE_DIR, 'data', 'results', 'skill_vectors.csv')

COLS_TO_EXPORT = [
    'YEAR', 'PERWT', 'AGE', 'RACE', 'RACED', 'EDUC', 'EDUCD', 'CLASSWKRD','WAGE', 'INDNAICS', 'cbsa20', 'WFH',
    'OCCSOC_detailed', 'OCCSOC_broad', 'OCCSOC_minor',
    'TELEWORKABLE_OCCSOC_detailed', 'TELEWORKABLE_OCCSOC_broad', 'TELEWORKABLE_OCCSOC_minor',
    'SKILL_MECHANICAL_OCCSOC_detailed', 'SKILL_COGNITIVE_OCCSOC_detailed', 'SKILL_SOCIAL_OCCSOC_detailed',
    'SKILL_MECHANICAL_OCCSOC_broad', 'SKILL_COGNITIVE_OCCSOC_broad', 'SKILL_SOCIAL_OCCSOC_broad',
    'SKILL_MECHANICAL_OCCSOC_minor', 'SKILL_COGNITIVE_OCCSOC_minor', 'SKILL_SOCIAL_OCCSOC_minor'
]

# Ensure directories exist
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# %% Define functions
#? Read ACS data
def read_acs_data(path, min_year=None, max_year=None):
    """
    Read ACS data from a CSV file and filter it based on the specified minimum and maximum years.
    """
    logging.info(f"Reading ACS data from {path}")
    
    # Define data types
    dtypes = {
        "YEAR": int,
        "STATEFIP" : str,
        "PUMA" : str,
        "PERWT" : float,
        "RACE" : str,
        "RACED" : str,
        "EDUC" : str,
        "EDUCD" : str,
        "CLASSWKR": str,
        "CLASSWKRD": str,
        "INDNAICS": str,
        "OCCSOC" : str,
        "TRANWORK": str,
        "TRANTIME": float,
        "INCWAGE": float,
        "UHRSWORK": float,
        "INCTOT": float
    }
    
    data = pd.read_csv(path, compression='gzip', low_memory=False, dtype=dtypes)
    
    logging.info(f"Data shape after reading: {data.shape}")

    # Filter data
    if min_year is not None:
        data = data[data['YEAR'] >= min_year]
        logging.info(f"Filtered data to min_year {min_year}. Shape is now: {data.shape}")
    if max_year is not None:
        data = data[data['YEAR'] <= max_year]
        logging.info(f"Filtered data to max_year {max_year}. Shape is now: {data.shape}")

    return data

#? Filter ACS data
def filter_acs_data(data, **kwargs):
    """
    Filter ACS (American Community Survey) data based on specified criteria.
    """
    logging.info("Starting filter_acs_data function.")
    
    # Minimum hours worked
    if ("UHRSWORK" in data.columns) or ("hours_worked_lim" in kwargs):
        if "hours_worked_lim" not in kwargs:
            kwargs["hours_worked_lim"] = 35
        if "UHRSWORK" not in data.columns:
            logging.warning("Column 'UHRSWORK' not found. Skipping filter based on minimum hours worked.")
        else:
            initial_shape = data.shape
            data = data[data['UHRSWORK'] > kwargs["hours_worked_lim"]]
            logging.info(f"Filtered by UHRSWORK > {kwargs['hours_worked_lim']}. Rows reduced from {initial_shape[0]} to {data.shape[0]}.")

    # Calculate wage per hour if data available
    if 'INCWAGE' in data.columns and 'UHRSWORK' in data.columns:
        data.loc[data.index, 'WAGE'] = data.INCWAGE / (data.UHRSWORK * 52)
        logging.info("Calculated hourly wage and stored in 'WAGE' column.")
        # Drop the original wage and hours worked columns
        data = data.drop(columns=['INCWAGE', 'UHRSWORK'])
        logging.info("Dropped columns 'INCWAGE' and 'UHRSWORK'.")

    # Minimum wage filter
    if ("WAGE" in data.columns) or "wage_lim" in kwargs:
        if "wage_lim" not in kwargs:
            kwargs["wage_lim"] = 5
        if "WAGE" not in data.columns:
            logging.warning("Column 'WAGE' not found. Skipping filter based on minimum wage.")
        else:
            initial_shape = data.shape
            data = data[data['WAGE'] > kwargs["wage_lim"]]
            logging.info(f"Filtered by WAGE > {kwargs['wage_lim']}. Rows reduced from {initial_shape[0]} to {data.shape[0]}.")

    # Class of worker filter
    if ("CLASSWKR" in data.columns) or "class_of_worker" in kwargs:
        if "class_of_worker" not in kwargs:
            kwargs["class_of_worker"] = ['2']  # Work for wages
        if "CLASSWKR" not in data.columns:
            logging.warning("Column 'CLASSWKR' not found. Skipping filter based on class of worker.")
        else:
            initial_shape = data.shape
            data = data[data['CLASSWKR'].isin(kwargs["class_of_worker"])]
            logging.info(f"Filtered by CLASSWKR in {kwargs['class_of_worker']}. Rows reduced from {initial_shape[0]} to {data.shape[0]}.")

    return data

#? Modify the industry codes
def modify_industry_codes(data):
    """
    Modifies the industry codes in the given data.
    """
    logging.info("Starting modify_industry_codes function.")
    
    # Strip whitespace and remove unwanted codes
    data.INDNAICS = data.INDNAICS.str.strip()
    initial_shape = data.shape
    data = data[data.INDNAICS != "0"]
    logging.info(f"Removed rows with INDNAICS=='0'. Rows reduced from {initial_shape[0]} to {data.shape[0]}.")
    
    # Remove military industry codes
    military_codes = ["928110P1", "928110P2", "928110P3", "928110P4", "928110P5", "928110P6", "928110P7"]
    data = data[~data.INDNAICS.isin(military_codes)]
    logging.info("Removed military industry codes.")
    
    # Remove unemployed codes
    data = data[data.INDNAICS != "999920"]
    logging.info("Removed unemployed codes (INDNAICS=='999920').")
    
    return data

# ? Create aggregator for occupation codes
def create_aggregator(path):
    """
    Create an aggregator for SOC (Standard Occupational Classification) data.
    """
    logging.info(f"Creating SOC aggregator from file: {path}")
    soc_2018_struct = pd.read_excel(path, skiprows=7)
    group_soc_data = pd.DataFrame()
    major_occ = soc_2018_struct['Major Group'].unique().tolist()

    for mo in major_occ:
        if not isinstance(mo, str):
            continue
        minor_group = soc_2018_struct.loc[soc_2018_struct["Minor Group"].str.startswith(mo[:2]) == True, "Minor Group"].unique().tolist()
        for mg in minor_group:
            broad_group = soc_2018_struct.loc[soc_2018_struct["Broad Group"].str.startswith(mg[:4]) == True, "Broad Group"].unique().tolist()
            for bg in broad_group:
                detailed_occupation = soc_2018_struct.loc[soc_2018_struct["Detailed Occupation"].str.startswith(bg[:6]) == True, "Detailed Occupation"].unique().tolist()
                bg_list = [bg] * len(detailed_occupation)
                mg_list = [mg] * len(detailed_occupation)
                mo_list = [mo] * len(detailed_occupation)
                group_soc_data = pd.concat(
                    [group_soc_data,
                     pd.DataFrame({
                         "Detailed Occupation": detailed_occupation,
                         "Broad Group": bg_list,
                         "Minor Group": mg_list,
                         "Major Group": mo_list
                     })]
                )
    logging.info(f"SOC aggregator created with {group_soc_data.shape[0]} rows.")
    return group_soc_data

# ? Convert using crosswalk
def convert_using_cw(code, cw, keep_original=True):
    """
    Convert a code using a code-to-value dictionary.
    """
    if code not in cw.keys():
        if keep_original:
            return code
        else:
            return np.nan
    return cw[code]

# ? Modify the occupation codes
def modify_occupation_codes(data, aggregator, occ_col="OCCSOC", threshold=2):
    """
    Modify the occupation codes in the given data DataFrame.
    """
    logging.info("Starting modify_occupation_codes function.")
    initial_shape = data.shape

    data.dropna(subset=[occ_col], inplace=True)
    data = data[data[occ_col] != "nan"]

    # Format the occupation codes if needed
    if data[occ_col].str.contains("-").all():
        logging.info(f"Column {occ_col} already formatted.")
    else:
        data[occ_col] = data[occ_col].apply(lambda x: "-".join([x[:2], x[2:]]))
        logging.info(f"Formatted {occ_col} to standard 'XX-XXXX' format.")

    # Remove rows with too many 'X' or 'Y' characters
    before_filter = data.shape[0]
    data = data[data[occ_col].str.count('X') + data[occ_col].str.count('Y') <= threshold]
    logging.info(f"Removed rows with more than {threshold} X/Y characters. Rows reduced from {before_filter} to {data.shape[0]}.")

    # Replace "X" and "Y" with "0"
    data[occ_col] = data[occ_col].str.replace("X", "0").str.replace("Y", "0")
    logging.info("Replaced X and Y in occupation codes with 0.")

    # Classify occupation codes based on the aggregator
    data[occ_col + "_group"] = np.nan
    data.loc[data[occ_col].isin(aggregator["Detailed Occupation"]), occ_col + "_group"] = "detailed"
    data.loc[data[occ_col].isin(aggregator["Broad Group"]), occ_col + "_group"] = "broad"
    data.loc[data[occ_col].isin(aggregator["Minor Group"]), occ_col + "_group"] = "minor"
    data.loc[data[occ_col].isin(aggregator["Major Group"]), occ_col + "_group"] = "major"
    # Drop rows where group remains NaN
    before_drop = data.shape[0]
    data = data[data[occ_col + "_group"].notna()]
    logging.info(f"Dropped rows with unclassified {occ_col}. Rows reduced from {before_drop} to {data.shape[0]}.")

    # Create dictionaries for mapping
    soc_2018_dict_broad       = dict(zip(aggregator["Detailed Occupation"], aggregator["Broad Group"]))
    soc_2018_dict_minor       = dict(zip(aggregator["Detailed Occupation"], aggregator["Minor Group"]))
    soc_2018_dict_broad_minor = dict(zip(aggregator["Broad Group"], aggregator["Minor Group"]))

    # Create Detailed, Broad, and Minor columns
    data[occ_col + "_detailed"] = np.nan
    data[occ_col + "_broad"]    = np.nan
    data[occ_col + "_minor"]    = np.nan

    # Fill detailed and broad columns based on group classification
    data.loc[data[occ_col + "_group"] == "detailed", occ_col + "_detailed"] = data[occ_col]
    data.loc[data[occ_col + "_group"] == "broad", occ_col + "_broad"] = data[occ_col]
    data.loc[data[occ_col + "_group"] == "detailed", occ_col + "_broad"] = data.loc[data[occ_col + "_group"] == "detailed", occ_col].apply(
        lambda x: convert_using_cw(x, soc_2018_dict_broad)
    )
    data[occ_col + "_minor"] = data[occ_col + "_broad"].apply(lambda x: convert_using_cw(x, soc_2018_dict_broad_minor))
    data.loc[data[occ_col + "_group"] == "minor", occ_col + "_minor"] = data[occ_col]

    logging.info(f"modify_occupation_codes complete. Data shape is now: {data.shape}")
    return data

# ? Aggregate PUMA codes to CBSA codes
def aggregate_puma_to_cbsa(data, puma_crosswalk):
    """
    Aggregates PUMA (Public Use Microdata Area) codes to CBSA (Core Based Statistical Area) codes.
    """
    logging.info("Starting aggregate_puma_to_cbsa function.")
    # Pad the state and PUMA codes with leading zeros
    data['STATEFIP'] = data['STATEFIP'].str.zfill(2)
    data['PUMA'] = data['PUMA'].str.zfill(5)
    data['state_puma'] = data['STATEFIP'] + '-' + data['PUMA']
    data = data.drop(columns=['STATEFIP', 'PUMA'])
    logging.info("Formatted and combined STATEFIP and PUMA into state_puma.")

    # Read the PUMA to CBSA crosswalk
    puma_to_cbsa = pd.read_csv(puma_crosswalk, dtype={'state_puma': str, 'cbsa20': str})
    logging.info(f"Read PUMA crosswalk from {puma_crosswalk} with {puma_to_cbsa.shape[0]} rows.")
    # Create crosswalk dictionary
    puma_to_cbsa_dict = dict(zip(puma_to_cbsa['state_puma'], puma_to_cbsa['cbsa20']))
    # Map each PUMA to the CBSA 
    data['cbsa20'] = data['state_puma'].apply(lambda x: convert_using_cw(x, puma_to_cbsa_dict, False))
    logging.info("Aggregated state_puma to CBSA codes.")

    return data

# ? Telework Index
def telework_index(data, soc_aggregator, how="my_index", occ_col="OCCSOC", path_to_remote_work_index=WFH_INDEX):
    """
    Calculate the telework index for different occupation codes.
    """
    logging.info("Starting telework_index function.")
    if how == "my_index":
        logging.info(f"Using my_index method. Loading telework index from {path_to_remote_work_index}.")
        wfh_index = pd.read_csv(path_to_remote_work_index)
        # Average the remote work index for each occupation code
        clasification = wfh_index.groupby('OCC_CODE')['ESTIMATE_WFH_ABLE'].mean().reset_index()
        clasification = clasification.merge(soc_aggregator, left_on = "OCC_CODE", right_on="Detailed Occupation", how="inner")
        clasification.rename(columns = {"ESTIMATE_WFH_ABLE":"TELEWORKABLE"}, inplace=True)
    elif how == "dn_index":   
        # Load external classification data
        logging.info("Using dn_index method. Loading telework classification data.")
        clasification = pd.read_csv("https://raw.githubusercontent.com/jdingel/DingelNeiman-workathome/master/occ_onet_scores/output/occupations_workathome.csv")
        clasification.rename(columns=lambda x: x.upper(), inplace=True)
        # Split the ONETSOCCODE to get OCC_CODE and ONET_DETAIL
        clasification[['OCC_CODE', 'ONET_DETAIL']] = clasification['ONETSOCCODE'].str.split(".", expand=True)
        clasification = clasification[clasification['ONET_DETAIL'] == "00"]
        clasification = clasification[['OCC_CODE', 'TITLE', 'TELEWORKABLE']]
        logging.info(f"Filtered telework data to ONET_DETAIL=='00'. Shape: {clasification.shape}")
    else:
        raise ValueError("Invalid value for 'how' parameter. Must be 'my_index' or 'dn_index'.")
    
    logging.info("Loaded and formatted telework classification data.")
    

    # Create mapping from OCC_CODE to TELEWORKABLE
    telework_dict = dict(zip(clasification['OCC_CODE'], clasification['TELEWORKABLE']))
    data['TELEWORKABLE_OCCSOC'] = data['OCCSOC'].apply(lambda x: convert_using_cw(x, telework_dict, False))
    logging.info("Mapped TELEWORKABLE values to data based on OCCSOC.")

    # Create dictionaries for mapping detailed to broad and minor groups
    soc_2018_dict_broad       = dict(zip(soc_aggregator["Detailed Occupation"], soc_aggregator["Broad Group"]))
    soc_2018_dict_minor       = dict(zip(soc_aggregator["Detailed Occupation"], soc_aggregator["Minor Group"]))
    soc_2018_dict_broad_minor = dict(zip(soc_aggregator["Broad Group"], soc_aggregator["Minor Group"]))
    
    # Map telework values at different aggregation levels
    clasification["BROAD"] = clasification["OCC_CODE"].apply(lambda x: convert_using_cw(x, soc_2018_dict_broad))
    clasification["MINOR"] = clasification["OCC_CODE"].apply(lambda x: convert_using_cw(x, soc_2018_dict_minor))
    logging.info("Created BROAD and MINOR occupation codes for telework classification.")

    # Group and average TELEWORKABLE values for BROAD and MINOR categories
    clasification_BROAD = clasification.groupby("BROAD")["TELEWORKABLE"].mean().reset_index()   
    clasification_MINOR = clasification.groupby("MINOR")["TELEWORKABLE"].mean().reset_index()  
    logging.info("Calculated average TELEWORKABLE values for BROAD and MINOR groups.")

    # Create mapping dictionaries for telework values
    telework_dict_detailed = dict(zip(clasification['OCC_CODE'], clasification['TELEWORKABLE']))
    telework_dict_broad = dict(zip(clasification_BROAD['BROAD'], clasification_BROAD['TELEWORKABLE']))
    telework_dict_minor = dict(zip(clasification_MINOR['MINOR'], clasification_MINOR['TELEWORKABLE']))

    # Map the telework values to the data for detailed, broad, and minor groups
    data['TELEWORKABLE_' + occ_col + '_detailed'] = data[occ_col + '_detailed'].apply(lambda x: convert_using_cw(x, telework_dict_detailed, False))
    data['TELEWORKABLE_' + occ_col + '_broad']    = data[occ_col + '_broad'].apply(lambda x: convert_using_cw(x, telework_dict_broad, False))
    data['TELEWORKABLE_' + occ_col + '_minor']    = data[occ_col + '_minor'].apply(lambda x: convert_using_cw(x, telework_dict_minor, False))
    logging.info("Assigned telework indices to detailed, broad, and minor occupation codes.")

    return data

# ? Skill Vectors
def skill_vectors(data, soc_aggregator, occ_col="OCCSOC", path_to_skill_vectors=SKILL_VECTORS):
    """
    Add skill vectors (mechanical, cognitive, social) to the data for different occupation codes.
    """
    logging.info("Starting skill_vectors function.")
    
    # Load skill vectors data
    logging.info(f"Loading skill vectors from {path_to_skill_vectors}")
    skill_data = pd.read_csv(path_to_skill_vectors)
    
    # Extract the SOC code from ONET_SOC_CODE (remove the .00 suffix)
    skill_data['OCC_CODE'] = skill_data['ONET_SOC_CODE'].str.split('.').str[0]
    
    # Group by OCC_CODE and calculate mean for each skill dimension
    # This handles cases where multiple ONET codes map to the same SOC code
    classification = skill_data.groupby('OCC_CODE')[['mechanical', 'cognitive', 'social']].mean().reset_index()
    
    # Create dictionaries for mapping detailed codes to skill scores
    mechanical_dict_detailed = dict(zip(classification['OCC_CODE'], classification['mechanical']))
    cognitive_dict_detailed = dict(zip(classification['OCC_CODE'], classification['cognitive']))
    social_dict_detailed = dict(zip(classification['OCC_CODE'], classification['social']))
    
    # Map the skill scores to detailed occupation codes
    data['SKILL_MECHANICAL_' + occ_col + '_detailed'] = data[occ_col + '_detailed'].apply(lambda x: convert_using_cw(x, mechanical_dict_detailed, False))
    data['SKILL_COGNITIVE_' + occ_col + '_detailed'] = data[occ_col + '_detailed'].apply(lambda x: convert_using_cw(x, cognitive_dict_detailed, False))
    data['SKILL_SOCIAL_' + occ_col + '_detailed'] = data[occ_col + '_detailed'].apply(lambda x: convert_using_cw(x, social_dict_detailed, False))
    logging.info("Mapped skill vectors to detailed occupation codes.")
    
    # Create dictionaries for mapping detailed to broad and minor groups
    soc_2018_dict_broad = dict(zip(soc_aggregator["Detailed Occupation"], soc_aggregator["Broad Group"]))
    soc_2018_dict_minor = dict(zip(soc_aggregator["Detailed Occupation"], soc_aggregator["Minor Group"]))
    
    # Add broad and minor group codes to the classification data
    classification["BROAD"] = classification["OCC_CODE"].apply(lambda x: convert_using_cw(x, soc_2018_dict_broad))
    classification["MINOR"] = classification["OCC_CODE"].apply(lambda x: convert_using_cw(x, soc_2018_dict_minor))
    logging.info("Created BROAD and MINOR occupation codes for skill vector classification.")
    
    # Group and average skill values for BROAD category
    classification_BROAD_mechanical = classification.groupby("BROAD")["mechanical"].mean().reset_index()
    classification_BROAD_cognitive = classification.groupby("BROAD")["cognitive"].mean().reset_index()
    classification_BROAD_social = classification.groupby("BROAD")["social"].mean().reset_index()
    
    # Group and average skill values for MINOR category
    classification_MINOR_mechanical = classification.groupby("MINOR")["mechanical"].mean().reset_index()
    classification_MINOR_cognitive = classification.groupby("MINOR")["cognitive"].mean().reset_index()
    classification_MINOR_social = classification.groupby("MINOR")["social"].mean().reset_index()
    logging.info("Calculated average skill values for BROAD and MINOR groups.")
    
    # Create mapping dictionaries for skill values at broad level
    mechanical_dict_broad = dict(zip(classification_BROAD_mechanical['BROAD'], classification_BROAD_mechanical['mechanical']))
    cognitive_dict_broad = dict(zip(classification_BROAD_cognitive['BROAD'], classification_BROAD_cognitive['cognitive']))
    social_dict_broad = dict(zip(classification_BROAD_social['BROAD'], classification_BROAD_social['social']))
    
    # Create mapping dictionaries for skill values at minor level
    mechanical_dict_minor = dict(zip(classification_MINOR_mechanical['MINOR'], classification_MINOR_mechanical['mechanical']))
    cognitive_dict_minor = dict(zip(classification_MINOR_cognitive['MINOR'], classification_MINOR_cognitive['cognitive']))
    social_dict_minor = dict(zip(classification_MINOR_social['MINOR'], classification_MINOR_social['social']))
    
    # Map the skill values to the data for broad occupation groups
    data['SKILL_MECHANICAL_' + occ_col + '_broad'] = data[occ_col + '_broad'].apply(lambda x: convert_using_cw(x, mechanical_dict_broad, False))
    data['SKILL_COGNITIVE_' + occ_col + '_broad'] = data[occ_col + '_broad'].apply(lambda x: convert_using_cw(x, cognitive_dict_broad, False))
    data['SKILL_SOCIAL_' + occ_col + '_broad'] = data[occ_col + '_broad'].apply(lambda x: convert_using_cw(x, social_dict_broad, False))
    
    # Map the skill values to the data for minor occupation groups
    data['SKILL_MECHANICAL_' + occ_col + '_minor'] = data[occ_col + '_minor'].apply(lambda x: convert_using_cw(x, mechanical_dict_minor, False))
    data['SKILL_COGNITIVE_' + occ_col + '_minor'] = data[occ_col + '_minor'].apply(lambda x: convert_using_cw(x, cognitive_dict_minor, False))
    data['SKILL_SOCIAL_' + occ_col + '_minor'] = data[occ_col + '_minor'].apply(lambda x: convert_using_cw(x, social_dict_minor, False))
    logging.info("Assigned skill vectors to broad and minor occupation codes.")

    return data

# ? Add WFH index to the data (based on TRANWORK) 
def wfh_index(data):  
    """
    Calculate the work-from-home (WFH) index for each record.
    """
    logging.info("Starting wfh_index function.")
    data.loc[:, 'WFH'] = data.TRANWORK == "80"
    logging.info("WFH index calculated based on TRANWORK column.")
    return data

# ? Save the data
def save_data(data, path):
    """
    Save the data to a CSV file.
    """
    logging.info(f"Saving data to {path}. Final data shape: {data.shape}")
    data.to_csv(path, index=False)
    logging.info("Data saved successfully.")
    

# %% Main
if __name__ == "__main__":
    start_time = time.time()
    logging.info("Script started.")
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Process ACS data.')
    parser.add_argument('--min_year', type=int, default=2013, help='Minimum year to filter the data')
    parser.add_argument('--max_year', type=int, default=None, help='Maximum year to filter the data')
    parser.add_argument('--data_file_number', type=int, default=136, help='Number of the ACS data file (default: 136)')

    args = parser.parse_args()

    PATH_ACS_DATA = os.path.join(DATA_DIR, f'usa_00{args.data_file_number}.csv.gz')
    logging.info(f"Processing file number {args.data_file_number} from {PATH_ACS_DATA}")

    # Read the ACS data and crosswalks
    data = read_acs_data(PATH_ACS_DATA, min_year=2013)
    soc_aggregator = create_aggregator(SOC_AGGREGATOR)

    # Filter the ACS data
    data = filter_acs_data(data, hours_worked_lim=35, wage_lim=5, class_of_worker=['2'])
    # Modify the industry codes
    data = modify_industry_codes(data)
    # Modify the occupation codes
    data = modify_occupation_codes(data, soc_aggregator)
    # Aggregate PUMA codes to CBSA codes
    data = aggregate_puma_to_cbsa(data, PUMA_CROSSWALK)
    # Calculate the telework index for each individual's occupation
    data = telework_index(data, soc_aggregator)
    # Add skill vectors to the data
    data = skill_vectors(data, soc_aggregator)
    # Calculate the WFH index for each worker
    data = wfh_index(data)
    # Save the processed data
    output_path = os.path.join(PROCESSED_DATA_DIR, f'acs_{args.data_file_number}_processed.csv')
    # Select the relevant variables before exporting
    save_data(data[COLS_TO_EXPORT], output_path)

    end_time = time.time()
    total_time = end_time - start_time
    logging.info(f"Script finished successfully in {total_time:.2f} seconds.")

# %%
# Example analysis (commented out)
# average_wfh_by_year = data.groupby('YEAR')['WFH'].apply(lambda x: (x * data.loc[x.index, 'PERWT']).sum() / data.loc[x.index, 'PERWT'].sum())
# average_telewk_by_year = data.groupby('YEAR')['TELEWORKABLE_OCCSOC_minor'].apply(lambda x: (x * data.loc[x.index, 'PERWT']).sum() / data.loc[x.index, 'PERWT'].sum())
# %%
