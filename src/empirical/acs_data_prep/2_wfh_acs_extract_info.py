# %% Import packages and define constants
import os
import pandas as pd
import numpy as np
import logging
import argparse

# Set up logging
current_dir = os.path.dirname(os.path.abspath(__file__))
script_name = os.path.basename(__file__)
log_file = os.path.join(current_dir, f'{script_name.replace(".py", ".log")}')
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Base paths
# BASE_DIR = '../../../../'
BASE_DIR = '.'
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'acs')

SOC_AGGREGATOR = os.path.join(BASE_DIR, 'data', 'aux_and_croswalks', 'soc_structure_2018.xlsx')
PUMA_CROSSWALK = os.path.join(BASE_DIR, 'data', 'aux_and_croswalks', "puma_to_cbsa.csv")

# %% Define functions
#? Read ACS data (proc)
def read_acs_data(path, min_year=None, max_year=None):
    """
    Read ACS data from a CSV file and filter it based on the specified minimum and maximum years.
    """
    # logging.info(f"Reading ACS data from {path}")
    
    # # Define data types
    dtypes = {
        "YEAR": int,
        "PERWT" : float,
        "RACE" : str,
        'AGE' : int,
        "EDUC" : str,
        "CLASSWKR": str,
        "CLASSWKRD": str,
        "INDNAICS": str,
        'OCCSOC_group' : str,
        'OCCSOC_detailed' : str,
        'OCCSOC_broad' : str,
        'OCCSOC_minor' :str,
        'WAGE' : float,
        'WFH' : bool
    }
    # Read data
    data = pd.read_csv(path, low_memory=False, dtype=dtypes, usecols=dtypes.keys())

    # Filter data
    if min_year is not None:
        data = data[data['YEAR'] >= min_year]
        logging.info(f"Filtered data to min_year {min_year}. Shape is now: {data.shape}")
    if max_year is not None:
        data = data[data['YEAR'] <= max_year]
        logging.info(f"Filtered data to max_year {max_year}. Shape is now: {data.shape}")

    return data

def compute_wfh_index(data, by = ["YEAR"]):
    """
    Compute the work from home index.
    """
    logging.info(f"Computing the work from home index by {', '.join(by)}")
    # Compute the work from home index
    wfh_index = data.groupby(by)[["WFH", "PERWT"]].apply(
        lambda x: pd.Series({
            "WFH_INDEX": np.average(x["WFH"], weights=x["PERWT"]),
            "TOTAL_WEIGHT": x["PERWT"].sum(),
        })
    ).reset_index()
    return wfh_index

# %% Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process ACS data.')
    parser.add_argument('--data_file_number', type=int, default=136, help='Number of the ACS data file (default: 136)')

    args = parser.parse_args()

    acs_data = read_acs_data(os.path.join(DATA_DIR, f'acs_{args.data_file_number}_processed.csv'))
    # Compute the work from home index by:
    by =[
            ["YEAR"], # -  year
            ["YEAR", "INDNAICS"], # - year, industry
            ["YEAR", "AGE"], # - year, age
            ["YEAR", "EDUC"], # - year, education
            ["YEAR", "RACE"], # - year
            ["YEAR", "OCCSOC_minor"], # - year, occupation (minor)
    ]
    for by_ in by:
        wfh_index = compute_wfh_index(acs_data, by=by_)
        file_name =  os.path.join(DATA_DIR, f"acs_{args.data_file_number}_{'_'.join(by_)}")
        wfh_index.to_csv(f"{file_name}.csv", index=False)
        logging.info(f"Saved work from home index to {os.path.join(DATA_DIR, 'wfh_index.csv')}")


# %%
