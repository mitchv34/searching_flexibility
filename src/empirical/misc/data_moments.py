"""
Script to pull BLS job openings and FRED unemployment data,
combine them and create monthly, quarterly, and yearly datasets.
"""

import pandas as pd
import requests
import numpy as np
from datetime import datetime
import os

def pull_bls_data(series_id, start_year=2013, end_year=None):
    """
    Pull data from BLS API for a given series ID.
    
    Args:
        series_id (str): BLS series ID
        start_year (int): Starting year
        end_year (int): Ending year (if None, uses current year)
    
    Returns:
        pandas.DataFrame: DataFrame with year, month, value columns
    """
    if end_year is None:
        end_year = datetime.now().year
    
    # BLS API endpoint
    url = 'https://api.bls.gov/publicAPI/v2/timeseries/data/'
    
    headers = {'Content-type': 'application/json'}
    
    # Prepare the request data
    data = {
        'seriesid': [series_id],
        'startyear': str(start_year),
        'endyear': str(end_year)
    }
    
    # Make the request
    response = requests.post(url, json=data, headers=headers)
    
    if response.status_code == 200:
        json_data = response.json()
        
        if json_data['status'] == 'REQUEST_SUCCEEDED':
            series_data = json_data['Results']['series'][0]['data']
            
            # Convert to DataFrame
            df_list = []
            for item in series_data:
                if item['period'] != 'M13':  # Exclude annual averages
                    year = int(item['year'])
                    month = int(item['period'][1:])  # Remove 'M' prefix
                    value = float(item['value']) if item['value'] != '' else np.nan
                    df_list.append({'year': year, 'month': month, 'value': value})
            
            df = pd.DataFrame(df_list)
            df = df.sort_values(['year', 'month']).reset_index(drop=True)
            return df
        else:
            raise Exception(f"BLS API request failed: {json_data['message']}")
    else:
        raise Exception(f"HTTP request failed with status code: {response.status_code}")

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
    
    # FRED API endpoint (using public access without API key for basic data)
    url = f'https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=12&width=1318&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id={series_id}&scale=left&cosd={start_date}&coed={end_date}&line_color=%234572a7&link_values=false&line_style=solid&mark_type=none&mw=3&lw=2&ost=-99999&oet=99999&mma=0&fml=a&fq=Monthly&fam=avg&fgst=lin&fgsnd=2020-02-01&line_index=1&transformation=lin&vintage_date={end_date}&revision_date={end_date}&nd={start_date}'
    
    try:
        # Read data directly from FRED CSV endpoint
        df = pd.read_csv(url)
        df.columns = ['date', 'value']
        
        # Convert date column and extract year/month
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        
        # Remove rows with missing values and select relevant columns
        df = df.dropna(subset=['value'])
        df = df[['year', 'month', 'value']].reset_index(drop=True)
        
        return df
    except Exception as e:
        # Fallback: try using pandas_datareader if available
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
    print("Pulling BLS job openings data (JTS000000000000000JOR)...")
    
    # Pull BLS job openings data
    try:
        bls_data = pull_bls_data('JTS000000000000000JOR', start_year=2013)
        print(f"Successfully pulled {len(bls_data)} observations from BLS")
    except Exception as e:
        print(f"Error pulling BLS data: {str(e)}")
        return
    
    print("Pulling FRED unemployment rate data (UNRATE)...")
    
    # Pull FRED unemployment data
    try:
        fred_data = pull_fred_data('UNRATE', start_date='2013-01-01')
        print(f"Successfully pulled {len(fred_data)} observations from FRED")
    except Exception as e:
        print(f"Error pulling FRED data: {str(e)}")
        return
    
    # Merge the datasets
    print("Merging datasets...")
    
    # Rename value columns
    bls_data = bls_data.rename(columns={'value': 'V'})
    fred_data = fred_data.rename(columns={'value': 'U'})
    
    # Merge on year and month
    monthly_data = pd.merge(bls_data, fred_data, on=['year', 'month'], how='inner')
    
    # Create theta column
    monthly_data['theta'] = monthly_data['V'] / monthly_data['U']
    
    print(f"Created monthly dataset with {len(monthly_data)} observations")
    
    # Create quarterly data
    print("Creating quarterly dataset...")
    quarterly_data = create_quarterly_data(monthly_data)
    print(f"Created quarterly dataset with {len(quarterly_data)} observations")
    
    # Create yearly data
    print("Creating yearly dataset...")
    yearly_data = create_yearly_data(monthly_data)
    print(f"Created yearly dataset with {len(yearly_data)} observations")
    
    # Create output directory if it doesn't exist
    output_dir = '/Users/mitchv34/Work/searching_flexibility/data/moments'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save datasets
    print("Saving datasets...")
    
    monthly_data.to_csv(os.path.join(output_dir, 'monthly_data.csv'), index=False)
    quarterly_data.to_csv(os.path.join(output_dir, 'quarterly_data.csv'), index=False)
    yearly_data.to_csv(os.path.join(output_dir, 'yearly_data.csv'), index=False)
    
    print("Datasets saved successfully!")
    
    # Display summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    print("\nMonthly Data:")
    print(monthly_data.describe())
    
    print("\nQuarterly Data:")
    print(quarterly_data.describe())
    
    print("\nYearly Data:")
    print(yearly_data.describe())
    
    print(f"\nData range: {monthly_data['year'].min()}-{monthly_data['month'].min():02d} to {monthly_data['year'].max()}-{monthly_data['month'].max():02d}")

if __name__ == "__main__":
    main()