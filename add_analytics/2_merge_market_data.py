import pandas as pd
import yfinance as yf
import os


def merge_with_market_data(analysis_csv='consolidated_analysis.csv', ticker='^GSPC'):
    """
    Loads consolidated analysis, downloads historical market data for a ticker,
    and merges them. It then calculates future returns over various horizons.

    Args:
        analysis_csv (str): Path to the consolidated analysis CSV file.
        ticker (str): The stock ticker to download (e.g., S&P 500 index).

    Returns:
        pd.DataFrame: Enriched DataFrame with market data and future returns.
                      Returns None if the input CSV is not found.
    """
    if not os.path.exists(analysis_csv):
        print(f"Error: File not found at '{analysis_csv}'")
        print("Please run '1_data_consolidation.py' first to generate this file.")
        return None

    # Load the consolidated analysis data
    df = pd.read_csv(analysis_csv, parse_dates=['date'])

    # Determine the date range for fetching market data
    start_date = df['date'].min()
    end_date = df['date'].max() + pd.Timedelta(days=100)

    print(f"Downloading S&P 500 ({ticker}) data from {start_date.date()} to {end_date.date()}...")

    # Download S&P 500 data, keeping auto_adjust=False to get Adj Close
    sp500_data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)

    if sp500_data.empty:
        print(f"Error: Could not download market data for {ticker}.")
        return None

    # --- FIX for MultiIndex and MergeError ---
    # 1. If yfinance returns a MultiIndex, flatten it by dropping the 'Ticker' level.
    if isinstance(sp500_data.columns, pd.MultiIndex):
        sp500_data.columns = sp500_data.columns.droplevel(1)

    # 2. Reset the index to make the 'Date' index a regular column.
    sp500_data.reset_index(inplace=True)

    # 3. Select and rename the columns for the merge.
    sp500_data = sp500_data[['Date', 'Adj Close']]
    sp500_data.rename(columns={'Date': 'date', 'Adj Close': 'sp500_close'}, inplace=True)

    # 4. Ensure date columns are of the same type before merging
    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
    sp500_data['date'] = pd.to_datetime(sp500_data['date']).dt.tz_localize(None)

    # Merge analysis data with market data
    merged_df = pd.merge(df, sp500_data, on='date', how='left')

    # Forward-fill any missing market data (e.g., for holidays/weekends)
    merged_df['sp500_close'] = merged_df['sp500_close'].ffill()

    # --- Calculate Future Returns ---
    horizons = {
        'short_term': 5,
        'medium_term': 21,
        'long_term': 63
    }

    for term, days in horizons.items():
        future_price = merged_df['sp500_close'].shift(-days)
        merged_df[f'{term}_return'] = (future_price - merged_df['sp500_close']) / merged_df['sp500_close']

    # merged_df = merged_df.dropna(subset=[f'{term}_return' for term in horizons])

    return merged_df


if __name__ == "__main__":
    print("\nStep 2: Merging analysis data with S&P 500 prices...")
    enriched_df = merge_with_market_data()

    if enriched_df is not None and not enriched_df.empty:
        output_filename = 'analysis_with_market_data.csv'
        enriched_df.to_csv(output_filename, index=False)
        print(f"\nMerge complete. Data saved to '{output_filename}'")
        print("\n--- First 5 rows of enriched data: ---")
        print(enriched_df.head())
        print("\nNext step: Run '3_backtest_and_optimize.py' to evaluate performance.")
    else:
        print("\nData merging failed. Please check previous steps.")
