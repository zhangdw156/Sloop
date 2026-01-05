import csv
import os

import typer

# 初始化 Typer 应用
app = typer.Typer()

# ==========================================
# 🛠️ 配置映射: 定义所有 CSV 的文件名与排序字段
# ==========================================
CATEGORY_CONFIG = {
    # 1. Multi Turn (原有的)
    "multi_turn": {
        "filename": "data_multi_turn.csv",
        "sort_col": "Multi Turn Overall Acc",
    },
    # 2. Overall (原有的)
    "overall": {"filename": "data_overall.csv", "sort_col": "Overall Acc"},
    # 3. Live (新增)
    "live": {"filename": "data_live.csv", "sort_col": "Live Overall Acc"},
    # 4. Non Live (新增)
    "non_live": {"filename": "data_non_live.csv", "sort_col": "Non Live Overall Acc"},
    # 5. Agentic (新增)
    "agentic": {"filename": "data_agentic.csv", "sort_col": "Agentic Overall Acc"},
    # 6. Format Sensitivity (新增)
    "format": {
        "filename": "data_format_sensitivity.csv",
        "sort_col": "Format Sensitivity Overall Acc",
    },
}


def parse_percentage(value):
    """Helper to convert string like '37.62%' to float 37.62"""
    try:
        if isinstance(value, str) and value.strip().endswith("%"):
            return float(value.strip().strip("%"))
        return float(value)
    except ValueError:
        return 0.0


def aggregate_logic(root_dir: str, output_file: str, category: str):
    """
    Core logic to aggregate, sort, and re-rank results based on category.
    """

    # 1. 获取配置信息
    if category not in CATEGORY_CONFIG:
        print(
            f"❌ Error: Unknown category '{category}'. Supported: {list(CATEGORY_CONFIG.keys())}"
        )
        return

    config = CATEGORY_CONFIG[category]
    target_filename = config["filename"]
    acc_col_name = config["sort_col"]

    print(f"🎯 Target Category: {category}")
    print(f"📄 Looking for file: score/{target_filename}")
    print(f"📊 Sorting by column: {acc_col_name}")

    aggregated_data = []
    header = None
    final_header = ["Experiment"]  # Output file always starts with Experiment name

    print(f"🔍 Scanning directory: {root_dir}")

    # Iterate through each item in the root directory
    for item in sorted(os.listdir(root_dir)):
        experiment_path = os.path.join(root_dir, item)

        # Check if it is a directory
        if os.path.isdir(experiment_path):
            experiment_name = item

            # Construct path to the specific score file (DYNAMIC)
            score_file_path = os.path.join(experiment_path, "score", target_filename)

            if os.path.exists(score_file_path):
                try:
                    with open(score_file_path, "r", encoding="utf-8") as f:
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
        rank_col_name = "Rank"

        # Check if sort column exists in header
        if acc_col_name in final_header:
            acc_index = final_header.index(acc_col_name)

            print(f"\n🔄 Sorting by '{acc_col_name}'...")

            # Sort data: Descending order based on Accuracy
            aggregated_data.sort(
                key=lambda row: parse_percentage(row[acc_index]), reverse=True
            )

            # If 'Rank' column exists, update it
            if rank_col_name in final_header:
                rank_index = final_header.index(rank_col_name)
                for i, row in enumerate(aggregated_data):
                    row[rank_index] = str(i + 1)
            else:
                print("ℹ️  'Rank' column not found in source CSV. Skipping re-ranking.")
        else:
            print(f"⚠️  Warning: Sort column '{acc_col_name}' NOT found in CSV headers.")
            print(f"   (Available headers: {final_header})")
            print("   -> Data will be aggregated but NOT sorted.")

    except Exception as e:
        print(f"❌ Error during sorting/ranking: {e}")

    # --- Writing Output ---
    try:
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(final_header)
            writer.writerows(aggregated_data)
        print(f"\n🎉 Successfully wrote aggregated results to: {output_file}")

        # Print a preview
        if len(final_header) > 3:
            try:
                # Dynamic Preview
                model_idx = (
                    final_header.index("Model") if "Model" in final_header else 2
                )
                acc_idx = (
                    final_header.index(acc_col_name)
                    if acc_col_name in final_header
                    else 3
                )

                print("\n--- Preview (Top 5) ---")
                # Handle cases where columns might be missing in preview
                h_model = (
                    final_header[model_idx]
                    if model_idx < len(final_header)
                    else "Model"
                )
                h_acc = (
                    final_header[acc_idx] if acc_idx < len(final_header) else "Score"
                )

                print(f"{final_header[0]:<25} | {h_model:<30} | {h_acc}")
                print("-" * 80)
                for row in aggregated_data[:5]:
                    r_model = row[model_idx] if model_idx < len(row) else "N/A"
                    r_acc = row[acc_idx] if acc_idx < len(row) else "N/A"
                    print(f"{row[0]:<25} | {r_model:<30} | {r_acc}")
            except Exception:
                # Fallback preview
                print("\n--- Preview (Raw) ---")
                for row in aggregated_data[:5]:
                    print(row)

    except IOError as e:
        print(f"❌ Error writing to output file: {e}")


@app.command()
def main(
    root_dir: str = typer.Option(
        ...,
        "--root-dir",
        "-r",
        help="Path to the directory containing experiment folders",
    ),
    output_file: str = typer.Option(
        ..., "--output-file", "-o", help="Path to the output CSV file"
    ),
    category: str = typer.Option(
        "multi_turn",
        "--category",
        "-c",
        help=f"Category to aggregate. Options: {', '.join(CATEGORY_CONFIG.keys())}",
    ),
):
    """
    Aggregate BFCL benchmark results into a single CSV.
    """
    if not os.path.isdir(root_dir):
        print(f"Error: Input directory '{root_dir}' does not exist.")
        raise typer.Exit(code=1)

    category = category.lower()

    if category not in CATEGORY_CONFIG:
        print(f"Error: Invalid category '{category}'.")
        print(f"Available options: {', '.join(CATEGORY_CONFIG.keys())}")
        raise typer.Exit(code=1)

    aggregate_logic(root_dir, output_file, category)


if __name__ == "__main__":
    app()
