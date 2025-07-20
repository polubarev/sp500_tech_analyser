import os
import json
import pandas as pd
from datetime import datetime


def consolidate_analysis_data(folder_path='analysis_data'):
    """
    Reads all JSON analysis files from a folder, parses them, and
    consolidates the data into a pandas DataFrame.

    Args:
        folder_path (str): The path to the folder containing JSON files.

    Returns:
        pd.DataFrame: A DataFrame with the consolidated analysis data,
                      sorted by date. Returns None if the folder is not found.
    """
    if not os.path.isdir(folder_path):
        print(f"Error: Directory not found at '{folder_path}'")
        print("Please create the 'analysis_data' directory and place your JSON files inside.")
        return None

    all_records = []

    # Loop through each file in the specified directory
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                # Extract data for each term (short, medium, long)
                record = {
                    'date': pd.to_datetime(data['datetime']).date()
                }
                for term in ['short_term', 'medium_term', 'long_term']:
                    record[f'{term}_rec'] = data.get(term, {}).get('recommendation')
                    record[f'{term}_score'] = data.get(term, {}).get('score')

                all_records.append(record)

            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from {filename}")
            except Exception as e:
                print(f"Warning: An error occurred while processing {filename}: {e}")

    if not all_records:
        print("No valid JSON files were processed.")
        return None

    # Create DataFrame, sort by date, and reset index
    df = pd.DataFrame(all_records)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    return df


if __name__ == "__main__":
    # --- Main Execution ---
    print("Step 1: Consolidating analysis data from JSON files...")

    # Consolidate the data
    analysis_df = consolidate_analysis_data(folder_path='../predictions')

    if analysis_df is not None and not analysis_df.empty:
        # Save the consolidated data to a CSV file
        output_filename = 'consolidated_analysis.csv'
        analysis_df.to_csv(output_filename, index=False)

        print(f"\nConsolidation complete. {len(analysis_df)} records processed.")
        print(f"Data saved to '{output_filename}'")
        print("\n--- First 5 rows of consolidated data: ---")
        print(analysis_df.head())
        print("\nNext step: Run '2_merge_market_data.py' to add S&P 500 prices.")
    else:
        print("\nConsolidation failed. Please check the 'results' directory and JSON files.")
