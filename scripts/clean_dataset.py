#!/usr/bin/env python3
"""
Script to clean rlaif_eval_results dataset:
1. Remove rows with "Error code: 429" in the reasoning column
2. Ensure each prompt appears exactly 3 times (once for each comparison type)
   - If any comparison is missing for a prompt, remove all entries for that prompt
"""

import pandas as pd
import sys

def clean_dataset(input_file, output_file):
    """
    Clean the RLAIF evaluation results dataset.

    Args:
        input_file: Path to input CSV file
        output_file: Path to output cleaned CSV file
    """
    print(f"Reading data from {input_file}...")
    df = pd.read_csv(input_file)

    initial_rows = len(df)
    print(f"Initial dataset: {initial_rows} rows")

    # Step 1: Remove rows with "Error code: 429" in reasoning
    print("\nStep 1: Removing rows with 'Error code: 429'...")
    df_clean = df[~df['reasoning'].astype(str).str.contains('Error code: 429', na=False)]
    rows_after_error_removal = len(df_clean)
    removed_errors = initial_rows - rows_after_error_removal
    print(f"  Removed {removed_errors} rows with error code 429")
    print(f"  Remaining: {rows_after_error_removal} rows")

    # Step 2: Ensure each prompt appears exactly 3 times
    print("\nStep 2: Ensuring each prompt has all 3 comparison types...")

    # Count how many times each prompt appears
    prompt_counts = df_clean.groupby('prompt').size()

    # Get prompts that appear exactly 3 times
    valid_prompts = prompt_counts[prompt_counts == 3].index
    incomplete_prompts = prompt_counts[prompt_counts != 3]

    print(f"  Prompts with complete comparisons (3 entries): {len(valid_prompts)}")
    print(f"  Prompts with incomplete comparisons: {len(incomplete_prompts)}")

    if len(incomplete_prompts) > 0:
        print(f"\n  Incomplete prompt breakdown:")
        for count in sorted(incomplete_prompts.unique()):
            num_prompts = len(incomplete_prompts[incomplete_prompts == count])
            print(f"    {num_prompts} prompts with {count} entries")

    # Keep only rows where the prompt appears exactly 3 times
    df_final = df_clean[df_clean['prompt'].isin(valid_prompts)]

    rows_after_validation = len(df_final)
    removed_incomplete = rows_after_error_removal - rows_after_validation
    print(f"\n  Removed {removed_incomplete} rows from incomplete prompts")
    print(f"  Final dataset: {rows_after_validation} rows")

    # Verify the comparison types for the final dataset
    print("\nVerifying final dataset composition...")
    comparison_counts = df_final['comparison_type'].value_counts()
    print(f"  Comparison type distribution:")
    for comp_type, count in comparison_counts.items():
        print(f"    {comp_type}: {count}")

    # Verify that each comparison type has the same count
    if len(comparison_counts.unique()) == 1:
        print(f"  ✓ All comparison types have equal counts ({comparison_counts.iloc[0]} each)")
    else:
        print(f"  ⚠ Warning: Comparison types have unequal counts!")

    # Save cleaned dataset
    print(f"\nSaving cleaned dataset to {output_file}...")
    df_final.to_csv(output_file, index=False)

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Initial rows:              {initial_rows}")
    print(f"Rows with error code 429:  -{removed_errors}")
    print(f"Incomplete prompt rows:    -{removed_incomplete}")
    print(f"Final rows:                {rows_after_validation}")
    print(f"Reduction:                 {initial_rows - rows_after_validation} rows ({(initial_rows - rows_after_validation) / initial_rows * 100:.1f}%)")
    print(f"{'='*60}")

    return df_final

if __name__ == "__main__":
    input_file = "rlaif_eval_results(2).csv"
    output_file = "rlaif_eval_results_cleaned.csv"

    try:
        df_cleaned = clean_dataset(input_file, output_file)
        print(f"\n✓ Dataset cleaned successfully!")
        print(f"  Output saved to: {output_file}")
    except FileNotFoundError:
        print(f"Error: Could not find input file '{input_file}'")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
