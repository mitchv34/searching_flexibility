"""
Script to pull FRED job openings and unemployment data,
combine them and create monthly, quarterly, and yearly datasets.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

# Output directory for saving datasets (two levels up from script location)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(SCRIPT_DIR)), 'data', 'moments')

def pull_fred_data(series_id, start_date='2013-01-01', end_date=None):
    """
    Pull data from FRED API for a given series ID.
    
    Args:
        series_id (str): FRED series ID
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format (if None, uses current date)
    
    Returns:
        pandas.DataFrame: DataFrame with year, month, value columns
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    url = f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={start_date}&coed={end_date}'
    try:
        df = pd.read_csv(url)
        df.columns = ['date', 'value']
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df = df.dropna(subset=['value'])
        df = df[['year', 'month', 'value']].reset_index(drop=True)
        return df
    except Exception as e:
        try:
            import pandas_datareader.data as web
            df = web.get_data_fred(series_id, start=start_date, end=end_date)
            df = df.reset_index()
            df['year'] = df['DATE'].dt.year
            df['month'] = df['DATE'].dt.month
            df = df.rename(columns={series_id: 'value'})
            df = df[['year', 'month', 'value']].reset_index(drop=True)
            return df
        except ImportError:
            raise Exception(f"Could not retrieve FRED data. Error: {str(e)}")

def create_quarterly_data(df):
    """
    Create quarterly averages from monthly data.
    
    Args:
        df (pandas.DataFrame): Monthly data with year, month, and value columns
    
    Returns:
        pandas.DataFrame: Quarterly data
    """
    df_copy = df.copy()
    df_copy['quarter'] = ((df_copy['month'] - 1) // 3) + 1
    
    quarterly = df_copy.groupby(['year', 'quarter']).agg({
        'V': 'mean',
        'U': 'mean'
    }).reset_index()
    
    quarterly['theta'] = quarterly['V'] / quarterly['U']
    
    return quarterly

def create_yearly_data(df):
    """
    Create yearly averages from monthly data.
    
    Args:
        df (pandas.DataFrame): Monthly data with year, month, and value columns
    
    Returns:
        pandas.DataFrame: Yearly data
    """
    yearly = df.groupby('year').agg({
        'V': 'mean',
        'U': 'mean'
    }).reset_index()
    
    yearly['theta'] = yearly['V'] / yearly['U']
    
    return yearly

def main():
    """
    Main function to pull data and create datasets.
    """
    print("Pulling FRED job openings data (JTSJOL)...")
    try:
        fred_vacancies = pull_fred_data('JTSJOL', start_date='2013-01-01')
        print(f"Successfully pulled {len(fred_vacancies)} observations for vacancies from FRED")
    except Exception as e:
        print(f"Error pulling FRED vacancies data: {str(e)}")
        return

    print("Pulling FRED unemployment data (UNEMPLOY)...")
    try:
        fred_unemployed = pull_fred_data('UNEMPLOY', start_date='2013-01-01')
        print(f"Successfully pulled {len(fred_unemployed)} observations for unemployment from FRED")
    except Exception as e:
        print(f"Error pulling FRED unemployment data: {str(e)}")
        return

    print("Merging datasets...")
    fred_vacancies = fred_vacancies.rename(columns={'value': 'V'})
    fred_unemployed = fred_unemployed.rename(columns={'value': 'U'})
    monthly_data = pd.merge(fred_vacancies, fred_unemployed, on=['year', 'month'], how='inner')
    monthly_data['theta'] = monthly_data['V'] / monthly_data['U']
    print(f"Created monthly dataset with {len(monthly_data)} observations")

    print("Creating quarterly dataset...")
    quarterly_data = create_quarterly_data(monthly_data)
    print(f"Created quarterly dataset with {len(quarterly_data)} observations")

    print("Creating yearly dataset...")
    yearly_data = create_yearly_data(monthly_data)
    print(f"Created yearly dataset with {len(yearly_data)} observations")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Saving datasets...")
    monthly_data.to_csv(os.path.join(OUTPUT_DIR, 'monthly_data.csv'), index=False)
    quarterly_data.to_csv(os.path.join(OUTPUT_DIR, 'quarterly_data.csv'), index=False)
    yearly_data.to_csv(os.path.join(OUTPUT_DIR, 'yearly_data.csv'), index=False)
    print("Datasets saved successfully!")

    print("\n=== SUMMARY STATISTICS ===")
    print("\nMonthly Data:")
    print(monthly_data.describe())
    print("\nQuarterly Data:")
    print(quarterly_data.describe())
    print("\nYearly Data:")
    print(yearly_data)
    print(f"\nData range: {monthly_data['year'].min()}-{monthly_data['month'].min():02d} to {monthly_data['year'].max()}-{monthly_data['month'].max():02d}")

if __name__ == "__main__":
    main()