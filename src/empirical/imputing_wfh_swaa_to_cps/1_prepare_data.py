"""
Data Preparation Script for WFH Imputation Project

This script implements Phase 1 and Phase 2 of the WFH imputation methodology:
- Phase 1: SWAA Data Preparation 
- Phase 2: CPS Data Preparation & Harmonization

Following the methodology outlined in doc/imputing_wfh_swaa_to_cps.md
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import importlib.util
# Note: polars is imported locally in functions to handle optional dependency

# Add rich for colored output
try:
    from rich.console import Console
    from rich.text import Text
    console = Console()
    
    def print_success(message):
        console.print(f"✓ {message}", style="bold green")
    
    def print_error(message):
        console.print(f"✗ {message}", style="bold red")
    
    def print_warning(message):
        console.print(f"⚠ {message}", style="bold yellow")
    
    def print_info(message):
        console.print(message, style="cyan")
    
    def print_header(message):
        console.print(message, style="bold blue")
        
except ImportError:
    console = None
    
    def print_success(message):
        print(f"✓ {message}")
    
    def print_error(message):
        print(f"✗ {message}")
    
    def print_warning(message):
        print(f"⚠ {message}")
    
    def print_info(message):
        print(message)
    
    def print_header(message):
        print(message)

# Set up paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")

# Global file name variables - modify these to use different data files
SWAA_DATA_FILE = "WFHdata_March25.csv"
CPS_DATA_FILE = "usa_00137.csv.gz"
SWAA_OUTPUT_FILE = "swaa_prepared_for_stata.csv"
CPS_OUTPUT_FILE = "cps_prepared_for_stata.csv"

# Global date filter variables - modify these to change the time window
# Note: Year will be extracted from YYYYMM format for filtering to match ACS data
DEFAULT_START_DATE = 202001  # Start date in YYYYMM format (January 2020 -> filter by year 2020)
DEFAULT_END_DATE = None      # End date in YYYYMM format (None = no end limit)

print(f"Project root: {PROJECT_ROOT}")
print(f"Raw data path: {DATA_RAW}")
print(f"Processed data path: {DATA_PROCESSED}")
print(f"SWAA data file: {SWAA_DATA_FILE}")
print(f"CPS data file: {CPS_DATA_FILE}")
print(f"Default date range: {DEFAULT_START_DATE} (year: {DEFAULT_START_DATE // 100}) to {'present' if DEFAULT_END_DATE is None else f'{DEFAULT_END_DATE} (year: {DEFAULT_END_DATE // 100})'}")

def create_date_filter(year, month):
    """
    Helper function to create date filter in YYYYMM format
    
    Note: For SWAA data filtering, the year will be extracted from this YYYYMM format
    to match with ACS data which only contains year information.
    
    Parameters:
    -----------
    year : int
        Year (e.g., 2024)
    month : int
        Month (1-12)
    
    Returns:
    --------
    int
        Date in YYYYMM format (year component will be used for filtering)
    
    Example:
    --------
    >>> create_date_filter(2024, 7)  # July 2024
    202407
    >>> # When used in filtering, year 2024 will be extracted for year-based comparison
    """
    return year * 100 + month

def load_swaa_data():
    """
    Phase 1.1: Load Raw SWAA Data
    """
    swaa_file = os.path.join(DATA_RAW, "swaa", SWAA_DATA_FILE)
    print(f"Loading SWAA data from: {swaa_file}")

    if not os.path.exists(swaa_file):
        raise FileNotFoundError(f"SWAA data file not found: {swaa_file}")
    
    df = pd.read_csv(swaa_file, low_memory=False)
    print(f"SWAA data loaded. Shape: {df.shape}")
    return df

def filter_swaa_sample(df, start_date=202407, end_date=None):
    """
    Phase 1.2: Filter the Sample to the Target Population
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input SWAA dataframe
    start_date : int, optional
        Start date for filtering in YYYYMM format (default: 202407)
        Will extract year for year-based filtering to match ACS data
    end_date : int, optional
        End date for filtering in YYYYMM format (default: None, no upper limit)
        Will extract year for year-based filtering to match ACS data
    
    Returns:
    --------
    pandas.DataFrame
        Filtered dataframe with year column extracted from date
    """
    print("Filtering SWAA sample...")
    print(f"Initial sample size: {len(df)}")
    print(f"Date filter: {start_date} to {end_date if end_date else 'present'}")
    
    # Time Window - Convert YYYYMM date to year and filter by year to match ACS
    if 'date' in df.columns:
        print("Processing date column...")
        
        # Convert date to string first (in case it's numeric)
        df['date_str'] = df['date'].astype(str)
        
        # Extract year from YYYYMM format
        # Handle different date formats that might exist
        def extract_year_from_date(date_str):
            try:
                # Remove any non-numeric characters and ensure we have at least 4 digits
                date_clean = ''.join(filter(str.isdigit, str(date_str)))
                if len(date_clean) >= 4:
                    return int(date_clean[:4])  # First 4 digits are the year
                else:
                    return np.nan
            except (ValueError, TypeError):
                return np.nan
        
        # Extract year from the date
        df['year'] = df['date_str'].apply(extract_year_from_date)
        
        # Show year distribution
        if 'year' in df.columns and df['year'].notna().any():
            year_range = f"{df['year'].min():.0f} to {df['year'].max():.0f}"
            print(f"Date range extracted: {year_range}")
            print(f"Year distribution: {df['year'].value_counts().sort_index().head(10).to_dict()}")
        
        # Convert start_date and end_date to years for filtering
        start_year = start_date // 100 if start_date is not None else None
        end_year = end_date // 100 if end_date is not None else None
        
        print(f"Filtering by years: {start_year} to {end_year if end_year else 'present'}")
        
        # Apply start year filter
        if start_year is not None:
            initial_count = len(df)
            df = df[df['year'] >= start_year]
            print(f"After start year filter (>= {start_year}): {len(df)} (removed {initial_count - len(df)})")
        
        # Apply end year filter if specified
        if end_year is not None:
            initial_count = len(df)
            df = df[df['year'] <= end_year]
            print(f"After end year filter (<= {end_year}): {len(df)} (removed {initial_count - len(df)})")
        
        # Keep the year column for potential use in analysis
        print(f"Added 'year' column extracted from 'date' for year-based filtering")
        
    else:
        print("Warning: 'date' column not found - skipping date filter")
    
    # Data Quality - Remove low quality observations
    if 'ilowquality' in df.columns:
        initial_count = len(df)
        df = df[df['ilowquality'] != 1]
        print(f"After quality filter: {len(df)} (removed {initial_count - len(df)} low quality observations)")
    
    # Employment Status - Keep only currently employed
    if 'workstatus_current_new' in df.columns:
        initial_count = len(df)
        df = df[df['workstatus_current_new'].isin([1, 2])]  # 1: Working for pay, 2: Employed but not working
        print(f"After employment filter: {len(df)} (removed {initial_count - len(df)} non-employed observations)")
    
    return df

def create_dependent_variable(df):
    """
    Phase 1.3: Create the Dependent Variable
    """
    print("Creating dependent variable...")
    
    if 'wfhcovid_fracmat' not in df.columns:
        raise ValueError("Required column 'wfhcovid_fracmat' not found in SWAA data")
    
    # Create wfh_share by dividing wfhcovid_fracmat by 100
    df['wfh_share'] = df['wfhcovid_fracmat'] / 100
    
    print(f"WFH share summary:")
    print(df['wfh_share'].describe())
    
    return df

def prepare_swaa_for_modeling(df):
    """
    Phase 1.4: Select and Finalize Columns for Modeling
    """
    print("Preparing SWAA data for modeling...")
    
    # Define required variables for Stata model
    required_vars = [
        'wfh_share',           # Dependent variable
        'cratio100',           # Weight
        'occupation_clean',    # Predictor variables
        'work_industry',
        'education_s',
        'agebin',
        'gender',
        'race_ethnicity_s',
        'censusdivision',
    ]
    
    # Add year column if it exists (extracted from date filtering)
    if 'year' in df.columns:
        required_vars.append('year')
        print("Including 'year' variable extracted from date column")
    
    # Check which variables exist
    missing_vars = [var for var in required_vars if var not in df.columns]
    if missing_vars:
        print(f"Warning: Missing variables in SWAA data: {missing_vars}")
        print(f"Available columns: {list(df.columns)}")
        # Use only available variables
        required_vars = [var for var in required_vars if var in df.columns]
    
    # Select only required columns
    df_model = df[required_vars].copy()
    print(f"Selected {len(required_vars)} variables for modeling")
    
    # Drop rows with missing values
    initial_rows = len(df_model)
    df_model = df_model.dropna()
    final_rows = len(df_model)
    
    print(f"Dropped {initial_rows - final_rows} rows with missing values")
    print(f"Final SWAA sample size: {final_rows}")
    
    return df_model

def export_swaa_data(df):
    """
    Phase 1.5: Export the Prepared SWAA Data
    """
    output_file = os.path.join(DATA_PROCESSED, SWAA_OUTPUT_FILE)
    df.to_csv(output_file, index=False)
    print(f"SWAA data exported to: {output_file}")

def load_cps_data():
    """
    Phase 2.1: Load ACS Data - First check for preprocessed, then run preprocessing if not found
    
    This function:
    1. Checks for preprocessed ACS data in data/processed/acs/
    2. If not found, runs the Polars preprocessing pipeline directly
    3. Returns the processed ACS data ready for harmonization
    """
    # Generate processed file path based on raw file name
    processed_file_path = os.path.join(DATA_PROCESSED, "acs", CPS_DATA_FILE.replace(".csv.gz", ".csv") )

    print_info(f"Looking for preprocessed ACS data: {processed_file_path}")
    
    # Check if preprocessed file exists
    if os.path.exists(processed_file_path):
        print_success("Found preprocessed ACS data, loading...")
        df = pd.read_csv(processed_file_path)
        print_info(f"Preprocessed ACS data loaded. Shape: {df.shape}")
        print_info(f"Available columns: {list(df.columns)}")
        return df
    
    else:
        print_error("Preprocessed ACS data not found")
        print_info("Running preprocessing pipeline directly...")

        try:
            
            # Construct the path to the Polars script
            module_path = os.path.join(PROJECT_ROOT, "src", "empirical", "acs_data_prep", "1_wfh_acs_polar.py")
            if not os.path.exists(module_path):
                raise FileNotFoundError(f"Polars preprocessing script not found: {module_path}")
            # Import the Polars preprocessing function using importlib
            spec = importlib.util.spec_from_file_location("acs_polar_module", module_path)
            acs_polar_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(acs_polar_module)
            
            print_success("Successfully imported acs_data_proc function using importlib")
            
            # Run the processing function directly
            print_info("Running Polars preprocessing function...")
            processed_data = acs_polar_module.acs_data_proc(
                min_year=None,  # Run with default values
                max_year=None,
                data_file_name=CPS_DATA_FILE,
                return_data=True # Return processed data and save to file
            )
            
            # Convert to pandas if it's a Polars DataFrame
            if hasattr(processed_data, 'to_pandas'):
                df = processed_data.to_pandas()
            else:
                df = processed_data
                
            print_success(f"Preprocessing completed. Final shape: {df.shape}")
            return df
            
            
        except ImportError as e:
            print_error(f"Error importing preprocessing function: {e}")
            raise

# Harmonization functions for preprocessed (Polars) data
def harmonize_industry_from_naics(naics_code):
    """
    Map NAICS industry codes (from Polars preprocessing) to SWAA industry categories
    """
    if pd.isna(naics_code) or naics_code == "":
        return np.nan
    
    try:
        naics = str(naics_code)
        
        # Convert NAICS to SWAA industry categories
        
        # Category 1: Agriculture (NAICS 11)
        if naics.startswith('11'):
            return 1 
        
        # Category 2: Arts & Entertainment (NAICS 71)
        elif naics.startswith('71'):
            return 2
            
        # Category 3: Finance & Insurance (NAICS 52)
        elif naics.startswith('52'):
            return 3
            
        # Category 4: Construction (NAICS 23)
        elif naics.startswith('23'):
            return 4
            
        # Category 5: Education (NAICS 61)
        elif naics.startswith('61'):
            return 5
            
        # Category 6: Health Care & Social Assistance (NAICS 62)
        elif naics.startswith('62'):
            return 6
            
        # Category 7: Hospitality & Food Services (NAICS 72)
        elif naics.startswith('72'):
            return 7
            
        # Category 8: Information (NAICS 51)
        elif naics.startswith('51'):
            return 8
            
        # Category 9: Manufacturing (NAICS 31-33)
        elif naics.startswith(('31', '32', '33')):
            return 9
            
        # Category 10: Mining (NAICS 21)
        elif naics.startswith('21'):
            return 10
            
        # Category 11: Professional & Business Services (NAICS 54-56)
        elif naics.startswith(('54', '55', '56')):
            return 11
            
        # Category 12: Real Estate (NAICS 53)
        elif naics.startswith('53'):
            return 12
            
        # Category 13: Retail Trade (NAICS 44-45)
        elif naics.startswith(('44', '45')):
            return 13
            
        # Category 14: Transportation and Warehousing (NAICS 48-49)
        elif naics.startswith(('48', '49')):
            return 14
            
        # Category 15: Utilities (NAICS 22)
        elif naics.startswith('22'):
            return 15
            
        # Category 16: Wholesale Trade (NAICS 42)
        elif naics.startswith('42'):
            return 16
            
        # Category 17: Government (NAICS 92 - Public Administration)
        elif naics.startswith('92'):
            return 17
            
        # Category 18: Other (NAICS 81 - Other Services)
        else:
            # This will catch NAICS 81 and any other codes not specified above.
            return 18
        
    except (ValueError, TypeError):
        return np.nan

def harmonize_occupation_from_soc(soc_code):
    """
    Map SOC occupation codes (from Polars preprocessing) to SWAA occupation categories
    """
    if pd.isna(soc_code) or soc_code == "":
        return np.nan
    
    try:
        soc = str(soc_code)
        
        # Map SOC major groups to SWAA categories (simplified)
        if soc.startswith('55'):        # Armed Forces / Military Specific
            return 1
        elif soc.startswith('47'):      # Construction and Extraction
            return 2
        elif soc.startswith('45'):      # Farming, Fishing, and Forestry
            return 3
        elif soc.startswith('49'):      # Installation, Maintenance, and Repair
            return 4
        elif soc.startswith(('11', '13')): # Management, Business, and Financial Operations
            # 11: Management
            # 13: Business and Financial Operations
            return 5
        elif soc.startswith('43'):      # Office and Administrative Support
            return 6
        elif soc.startswith('51'):      # Production
            return 7
        elif soc.startswith(('15', '17', '19', '21', '23', '25', '27', '29')):
            # Professional and Related Occupations (a broad grouping)
            # Includes:
            # 15: Computer and Mathematical
            # 17: Architecture and Engineering
            # 19: Life, Physical, and Social Science
            # 21: Community and Social Service
            # 23: Legal
            # 25: Educational Instruction and Library
            # 27: Arts, Design, Entertainment, Sports, and Media
            # 29: Healthcare Practitioners and Technical
            return 8
        elif soc.startswith('41'):      # Sales and related
            return 9
        elif soc.startswith(('31', '33', '35', '37', '39')):
            # Service Occupations (a broad grouping)
            # Includes:
            # 31: Healthcare Support
            # 33: Protective Service
            # 35: Food Preparation and Serving
            # 37: Building and Grounds Cleaning
            # 39: Personal Care and Service
            return 10
        elif soc.startswith('53'):      # Transportation and material moving
            return 11
        else:                           # Other or unknown
            return 12
            
    except (ValueError, TypeError):
        return np.nan

def harmonize_education_from_ipums(educ_code):
    """
    Convert IPUMS education codes (EDUCD) to SWAA education categories
    """
    if pd.isna(educ_code):
        return np.nan
    
    try:
        educ = int(educ_code)
        
        # Map IPUMS EDUCD codes to SWAA categories
        if educ <= 61:          # Less than high school
            return 1
        elif educ <= 64:        # High school
            return 2
        elif educ < 100:       # Some college
            return 3
        elif educ <= 113:       # Bachelor's degree
            return 4
        else:                   # Graduate degree
            return 5
            
    except (ValueError, TypeError):
        return np.nan

def harmonize_race_from_ipums(race_code, ethnicity_code=None):
    """
    Convert IPUMS race codes (RACED) to SWAA race/ethnicity categories 
    inputs:
    - race_code: IPUMS race code (RACED)
    - ethnicity_code: IPUMS ethnicity code (HISPAN: 1 for Hispanic, 0 for non-Hispanic)
    """
    if pd.isna(race_code):
        return np.nan
    
    try:
        race = int(race_code)
        
        # If ethnicity code is provided, use it
        if ethnicity_code is not None:
            ethnicity = int(ethnicity_code)
            if ethnicity == 1:
                return 2  # Hispanic/Latino
            else:
                # Map race codes
                if race == 1:
                    return 4  # White
                elif race == 2:
                    return 1  # Black or African American
                else:
                    return 3  # Other (Asian, Native American, etc.)

    except (ValueError, TypeError):
        return np.nan

def harmonize_age(age):
    """
    Convert numeric age to SWAA age bin categories
    
    Age bin categories:
    1: Under 20
    2: 20-29
    3: 30-39
    4: 40-49
    5: 50-64
    6: 65+
    
    Parameters:
    -----------
    age : int or float
        Age value
    
    Returns:
    --------
    int
        Age bin category (1-6) or np.nan if age is missing
    """
    if pd.isna(age):
        return np.nan
    
    try:
        age = int(age)
        
        if age < 20:
            return 1  # Under 20
        elif age < 30:
            return 2  # 20-29
        elif age < 40:
            return 3  # 30-39
        elif age < 50:
            return 4  # 40-49
        elif age < 65:
            return 5  # 50-64
        else:
            return 6  # 65+
            
    except (ValueError, TypeError):
        return np.nan


def harmonize_cps_variables(df):
    """
    Phase 2.3: Harmonize Each "Bridge" Variable
    
    This function detects whether the data is preprocessed (from Polars) or raw,
    and applies the appropriate harmonization strategy.
    """
    print_info("Harmonizing CPS/ACS variables...")
    
    df_harmonized = df.copy()
    
    # PRESERVE UNIQUE IDENTIFIER if it exists
    if 'UNIQUE_PERSON_ID' in df.columns:
        print_success("Found UNIQUE_PERSON_ID - will preserve throughout harmonization")
    else:
        print_warning("UNIQUE_PERSON_ID not found - creating backup identifier")
        # Create backup identifier if the original isn't available
        if all(col in df.columns for col in ['SAMPLE', 'SERIAL', 'PERNUM']):
            df_harmonized['UNIQUE_PERSON_ID'] = (
                df['SAMPLE'].astype(str) + '_' + 
                df['SERIAL'].astype(str) + '_' + 
                df['PERNUM'].astype(str)
            )
            print_success("Created backup UNIQUE_PERSON_ID from SAMPLE+SERIAL+PERNUM")
        else:
            # Last resort - use index as identifier
            df_harmonized['UNIQUE_PERSON_ID'] = df_harmonized.index.astype(str)
            print_warning("Created UNIQUE_PERSON_ID from row index (last resort)")

    # Industry: Use INDNAICS (already cleaned by Polars)
    try:
        if 'INDNAICS' in df.columns:
            df_harmonized['work_industry'] = df['INDNAICS'].apply(harmonize_industry_from_naics)
            print_success(f"Successfully harmonized 'work_industry' from INDNAICS")
        else:
            print_warning(f"Failed to create 'work_industry': INDNAICS column not found")
    except Exception as e:
        print_warning(f"Failed to create 'work_industry': {e}")
    
    # Occupation: Use SOC hierarchy (prefer detailed, fall back to broad)
    try:
        if 'OCCSOC_DETAILED' in df.columns:
            df_harmonized['occupation_clean'] = df['OCCSOC_DETAILED'].apply(harmonize_occupation_from_soc)
            print_success(f"Successfully harmonized 'occupation_clean' from OCCSOC_DETAILED")
        elif 'OCCSOC_BROAD' in df.columns:
            df_harmonized['occupation_clean'] = df['OCCSOC_BROAD'].apply(harmonize_occupation_from_soc)
            print_success(f"Successfully harmonized 'occupation_clean' from OCCSOC_BROAD")
        else:
            print_warning(f"Failed to create 'occupation_clean': No occupation columns found")
    except Exception as e:
        print_warning(f"Failed to create 'occupation_clean': {e}")
    
    # Age: Use AGE column
    try:
        if 'AGE' in df.columns:
            df_harmonized['agebin'] = df['AGE'].apply(harmonize_age)
            print_success(f"Successfully harmonized 'agebin' from AGE")
        else:
            print_warning(f"Failed to create 'agebin': AGE column not found")
    except Exception as e:
        print_warning(f"Failed to create 'agebin': {e}")
    
    # Education: Use EDUCD if available, otherwise EDUC
    try:
        educ_col = 'EDUCD' if 'EDUCD' in df.columns else 'EDUC'
        if educ_col in df.columns:
            df_harmonized['education_s'] = df[educ_col].apply(harmonize_education_from_ipums)
            print_success(f"Successfully harmonized 'education_s' from {educ_col}")
        else:
            print_warning(f"Failed to create 'education_s': No education columns found")
    except Exception as e:
        print_warning(f"Failed to create 'education_s': {e}")
    
    # Gender: Use SEX column - create 'gender' to match SWAA
    try:
        if 'SEX' in df.columns:
            df_harmonized['gender'] = df['SEX'].astype(int)  # 1=Male, 2=Female
            print_success(f"Successfully harmonized 'gender' from SEX")
        else:
            print_warning(f"Failed to create 'gender': SEX column not found")
    except Exception as e:
        print_warning(f"Failed to create 'gender': {e}")
    
    # Race/Ethnicity: Use RACED and HISPAN if available
    try:
        if 'RACED' in df.columns and 'HISPAN' in df.columns:
            df_harmonized['race_ethnicity_s'] = df.apply(
                lambda row: harmonize_race_from_ipums(row['RACED'], row['HISPAN']), axis=1
            )
            print_success(f"Successfully harmonized 'race_ethnicity_s' from RACED and HISPAN")
        elif 'RACED' in df.columns:
            df_harmonized['race_ethnicity_s'] = df['RACED'].apply(harmonize_race_from_ipums)
            print_success(f"Successfully harmonized 'race_ethnicity_s' from RACED")
        elif 'RACE' in df.columns:
            df_harmonized['race_ethnicity_s'] = df['RACE'].apply(harmonize_race_from_ipums)
            print_success(f"Successfully harmonized 'race_ethnicity_s' from RACE")
        else:
            df_harmonized['race_ethnicity_s'] = 1  # Placeholder
            print_warning(f"Used placeholder value for 'race_ethnicity_s': No race columns found")
    except Exception as e:
        print_warning(f"Failed to create 'race_ethnicity_s': {e}")
    
    # Census division
    try:
        if 'CENSUSDIV' in df.columns:
            df_harmonized.rename(columns={'CENSUSDIV': 'censusdivision'}, inplace=True)
            print_success(f"Successfully renamed 'CENSUSDIV' to 'censusdivision'")
        else:
            print_warning(f"Failed to create 'censusdivision': CENSUSDIV column not found")
    except Exception as e:
        print_warning(f"Failed to create 'censusdivision': {e}")

    # Year: Use YEAR column from ACS data
    try:
        if 'YEAR' in df.columns:
            df_harmonized['year'] = df['YEAR'].astype(int)
            print_success(f"Successfully harmonized 'year' from YEAR")
        else:
            print_warning(f"Failed to create 'year': YEAR column not found")
    except Exception as e:
        print_warning(f"Failed to create 'year': {e}")

    # Final check of harmonized variables
    harmonized_vars = ['work_industry', 'occupation_clean', 'agebin', 'education_s', 
                    'gender', 'race_ethnicity_s', 'censusdivision', 'year']
    created_vars = [var for var in harmonized_vars if var in df_harmonized.columns]
    
    print_info(f"Harmonization completed. Created {len(created_vars)} of {len(harmonized_vars)} variables.")

    return df_harmonized

def prepare_cps_for_prediction(df):
    """
    Phase 2.4: Select and Finalize Columns for Prediction
    """
    print_info("Preparing CPS data for prediction...")
    
    # Make sure the identifier is preserved (lowercase for consistency)
    df.rename(columns={'UNIQUE_PERSON_ID': 'unique_person_id'}, inplace=True, errors='ignore')

    # Variables that must match SWAA exactly
    predictor_vars = [
        'unique_person_id',    # ALWAYS INCLUDE THE UNIQUE IDENTIFIER
        'occupation_clean',
        'work_industry',
        'education_s',
        'agebin',
        'gender',
        'race_ethnicity_s',
        'censusdivision',
        'year'                 # Include year for temporal matching
    ]
    
    # Check which variables exist
    missing_vars = [var for var in predictor_vars if var not in df.columns]
    if missing_vars:
        print_warning(f"Missing harmonized variables: {missing_vars}")
        # Remove missing vars except unique_person_id which we must have
        predictor_vars = [var for var in predictor_vars if var in df.columns or var == 'unique_person_id']
    
    # Ensure unique_person_id is always included
    if 'unique_person_id' not in predictor_vars and 'unique_person_id' in df.columns:
        predictor_vars.insert(0, 'unique_person_id')
    
    # Select only predictor columns
    df_pred = df[predictor_vars].copy()
    
    # Drop rows with missing values (EXCEPT for unique_person_id)
    initial_rows = len(df_pred)
    
    # Check for missing values in non-ID columns
    non_id_cols = [col for col in df_pred.columns if col != 'unique_person_id']
    df_pred = df_pred.dropna(subset=non_id_cols)
    
    final_rows = len(df_pred)
    
    print_info(f"Dropped {initial_rows - final_rows} rows with missing values")
    print_info(f"Final CPS sample size: {final_rows}")
    
    # Verify unique_person_id is still unique
    if 'unique_person_id' in df_pred.columns:
        unique_count = df_pred['unique_person_id'].nunique()
        total_count = len(df_pred)
        if unique_count == total_count:
            print_success(f"Unique identifier preserved: {unique_count:,} unique IDs")
        else:
            print_warning(f"Unique identifier issue: {unique_count:,} unique IDs for {total_count:,} rows")
    
    return df_pred

def export_cps_data(df):
    """
    Phase 2.5: Export the Prepared CPS Data
    """
    output_file = os.path.join(DATA_PROCESSED, CPS_OUTPUT_FILE)
    df.to_csv(output_file, index=False)
    print(f"CPS data exported to: {output_file}")

def main():
    """
    Main execution function
    """
    print_header("="*60)
    print_header("WFH IMPUTATION DATA PREPARATION")
    print_header("="*60)
    
    # Phase 1: SWAA Data Preparation
    print_header("\nPHASE 1: SWAA DATA PREPARATION")
    print_header("-" * 40)
    
    swaa_model_df = None
    try:
        # Load and process SWAA data
        swaa_df = load_swaa_data()
        swaa_df = filter_swaa_sample(swaa_df, start_date=DEFAULT_START_DATE, end_date=DEFAULT_END_DATE)
        swaa_df = create_dependent_variable(swaa_df)
        swaa_model_df = prepare_swaa_for_modeling(swaa_df)
        
        print_success("Phase 1 completed successfully!")
        
    except Exception as e:
        print_error(f"Phase 1 failed: {e}")
        print_info("Continuing to Phase 2...")
    
    # Phase 2: CPS Data Preparation & Harmonization
    print_header("\nPHASE 2: CPS DATA PREPARATION & HARMONIZATION")
    print_header("-" * 50)
    
    cps_pred_df = None
    try:
        # Load and process CPS data
        cps_df = load_cps_data()
        
        # Filter for employed individuals (implement employment filter as needed)
        # cps_df = filter_employed_individuals(cps_df)
        
        cps_harmonized = harmonize_cps_variables(cps_df)
        cps_pred_df = prepare_cps_for_prediction(cps_harmonized)
        
        print_success("Phase 2 completed successfully!")
        
    except Exception as e:
        print_error(f"Phase 2 failed: {e}")
    
    # Phase 3: Year Alignment - Ensure both datasets have the same years
    print_header("\nPHASE 3: YEAR ALIGNMENT")
    print_header("-" * 30)
    
    if swaa_model_df is not None and cps_pred_df is not None:
        try:
            # Check if both datasets have year columns
            swaa_has_year = 'year' in swaa_model_df.columns and swaa_model_df['year'].notna().any()
            cps_has_year = 'year' in cps_pred_df.columns and cps_pred_df['year'].notna().any()
            
            if swaa_has_year and cps_has_year:
                # Get year ranges for both datasets
                swaa_years = set(swaa_model_df['year'].dropna().astype(int))
                cps_years = set(cps_pred_df['year'].dropna().astype(int))
                
                print_info(f"SWAA years: {sorted(swaa_years)}")
                print_info(f"CPS years: {sorted(cps_years)}")
                
                # Find common years
                common_years = swaa_years.intersection(cps_years)
                
                if common_years:
                    print_info(f"Common years found: {sorted(common_years)}")
                    
                    # Filter both datasets to common years
                    swaa_initial_size = len(swaa_model_df)
                    cps_initial_size = len(cps_pred_df)
                    
                    swaa_model_df = swaa_model_df[swaa_model_df['year'].isin(common_years)]
                    cps_pred_df = cps_pred_df[cps_pred_df['year'].isin(common_years)]
                    
                    swaa_final_size = len(swaa_model_df)
                    cps_final_size = len(cps_pred_df)
                    
                    print_success(f"SWAA filtered: {swaa_final_size:,} rows (removed {swaa_initial_size - swaa_final_size:,})")
                    print_success(f"CPS filtered: {cps_final_size:,} rows (removed {cps_initial_size - cps_final_size:,})")
                    print_success(f"Both datasets now contain years: {sorted(common_years)}")
                    
                else:
                    print_warning("No common years found between SWAA and CPS datasets!")
                    print_warning("Proceeding with original datasets...")
                    
            elif not swaa_has_year:
                print_warning("SWAA dataset missing 'year' column - skipping year alignment")
            elif not cps_has_year:
                print_warning("CPS dataset missing 'year' column - skipping year alignment")
            else:
                print_warning("Both datasets missing 'year' columns - skipping year alignment")
                
        except Exception as e:
            print_error(f"Year alignment failed: {e}")
            print_warning("Proceeding with original datasets...")
    
    # Export final datasets
    print_header("\nEXPORTING FINAL DATASETS")
    print_header("-" * 30)
    
    if swaa_model_df is not None:
        try:
            export_swaa_data(swaa_model_df)
        except Exception as e:
            print_error(f"Failed to export SWAA data: {e}")
    
    if cps_pred_df is not None:
        try:
            export_cps_data(cps_pred_df)
        except Exception as e:
            print_error(f"Failed to export CPS data: {e}")
    
    print_info("\nData preparation complete!")
    print_info(f"Processed files saved to: {DATA_PROCESSED}")

if __name__ == "__main__":
    main()