import pandas as pd
from pathlib import Path

def sort_value_function(input_file: str, output_file: str = None):
    """Read, sort, and save the value function CSV.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to save sorted CSV (defaults to overwriting input file)
    """
    # Read the CSV
    df = pd.read_csv(input_file)
    
    # Separate summary and data rows
    summary_rows = df[df['shape1_sides'] == 'SUMMARY']
    data_rows = df[df['shape1_sides'] != 'SUMMARY']
    
    # Sort data rows by value (highest/least negative first)
    data_rows_sorted = data_rows.sort_values('value', ascending=False)
    
    # Recombine summary rows (at top) with sorted data
    df_sorted = pd.concat([summary_rows, data_rows_sorted], ignore_index=True)
    
    # Save to file
    output_file = output_file or input_file
    df_sorted.to_csv(output_file, index=False)
    print(f"Saved sorted values to {output_file}")

if __name__ == "__main__":
    input_file = "value-functions/goal_value_function_mean.csv"
    sort_value_function(input_file) 