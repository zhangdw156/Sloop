import os
import csv
import sys
import typer
from typing import List, Optional

# 初始化 Typer 应用
app = typer.Typer()

def parse_percentage(value):
    """Helper to convert string like '37.62%' to float 37.62"""
    try:
        if isinstance(value, str) and value.strip().endswith('%'):
            return float(value.strip().strip('%'))
        return float(value)
    except ValueError:
        return 0.0

def aggregate_logic(root_dir: str, output_file: str):
    """
    Core logic to aggregate, sort, and re-rank results.
    """
    
    # List to hold the aggregated rows
    aggregated_data = []
    
    # We need to capture the header from the source files
    header = None
    
    # Standard header for the output file
    # The 'Experiment' column will be the first column
    final_header = ["Experiment"]

    print(f"🔍 Scanning directory: {root_dir}")

    # Iterate through each item in the root directory
    for item in sorted(os.listdir(root_dir)):
        experiment_path = os.path.join(root_dir, item)
        
        # Check if it is a directory
        if os.path.isdir(experiment_path):
            experiment_name = item
            
            # Construct path to the specific score file
            score_file_path = os.path.join(experiment_path, "score", "data_multi_turn.csv")
            
            if os.path.exists(score_file_path):
                try:
                    with open(score_file_path, 'r', encoding='utf-8') as f:
                        reader = csv.reader(f)
                        rows = list(reader)
                        
                        if not rows:
                            print(f"⚠️  Warning: {score_file_path} is empty. Skipping.")
                            continue

                        current_header = rows[0]
                        
                        # Initialize header if it's the first file
                        if header is None:
                            header = current_header
                            final_header.extend(header)
                        elif header != current_header:
                            print(f"⚠️  Warning: Header mismatch in {experiment_name}.")
                        
                        # Collect data rows
                        for data_row in rows[1:]:
                            new_row = [experiment_name] + data_row
                            aggregated_data.append(new_row)
                            
                        print(f"✅ Found results for: {experiment_name}")
                        
                except Exception as e:
                    print(f"❌ Error reading {score_file_path}: {e}")

    if not aggregated_data:
        print("❌ No valid data found to aggregate.")
        return

    # --- Sorting and Re-ranking Logic ---
    try:
        # Dynamically find column indices to be robust against header changes
        # final_header structure example: ["Experiment", "Rank", "Model", "Multi Turn Overall Acc", ...]
        acc_col_name = "Multi Turn Overall Acc"
        rank_col_name = "Rank"

        if acc_col_name in final_header and rank_col_name in final_header:
            acc_index = final_header.index(acc_col_name)
            rank_index = final_header.index(rank_col_name)

            print(f"\n🔄 Sorting by '{acc_col_name}' and updating '{rank_col_name}'...")

            # Sort data: Descending order based on Accuracy
            aggregated_data.sort(
                key=lambda row: parse_percentage(row[acc_index]), 
                reverse=True
            )

            # Re-rank: Update the Rank column sequentially (1, 2, 3...)
            for i, row in enumerate(aggregated_data):
                row[rank_index] = str(i + 1)
        else:
            print(f"⚠️  Could not find '{acc_col_name}' or '{rank_col_name}' columns. Skipping sort/rank.")

    except Exception as e:
        print(f"❌ Error during sorting/ranking: {e}")
        # Proceed to write unsorted data rather than crashing

    # --- Writing Output ---
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(final_header)
            writer.writerows(aggregated_data)
        print(f"\n🎉 Successfully wrote aggregated results to: {output_file}")
        
        # Print a preview
        if len(final_header) > 3:
            # Try to grab relevant columns for preview (Exp, Model, Acc)
            # Indices: Experiment=0, Model usually=2, Acc usually=3 (but we use dynamic lookup if possible)
            try:
                model_idx = final_header.index("Model")
                acc_idx = final_header.index("Multi Turn Overall Acc")
                rank_idx = final_header.index("Rank")
                
                print("\n--- Preview (Top 5) ---")
                print(f"{final_header[0]:<20} | {final_header[rank_idx]:<5} | {final_header[model_idx]:<30} | {final_header[acc_idx]}")
                print("-" * 80)
                for row in aggregated_data[:5]:
                     print(f"{row[0]:<20} | {row[rank_idx]:<5} | {row[model_idx]:<30} | {row[acc_idx]}")
            except ValueError:
                # Fallback preview if columns aren't standard
                print("\n--- Preview (Raw) ---")
                print(final_header)
                for row in aggregated_data[:5]:
                    print(row)

    except IOError as e:
        print(f"❌ Error writing to output file: {e}")

@app.command()
def main(
    root_dir: str = typer.Option(..., "--root-dir", "-r", help="Path to the directory containing experiment folders"),
    output_file: str = typer.Option(..., "--output-file", "-o", help="Path to the output CSV file")
):
    """
    Aggregate BFCL multi-turn benchmark results into a single CSV with sorting and ranking.
    """
    # Validate input directory
    if not os.path.isdir(root_dir):
        print(f"Error: Input directory '{root_dir}' does not exist.")
        raise typer.Exit(code=1)
        
    aggregate_logic(root_dir, output_file)

if __name__ == "__main__":
    app()
    