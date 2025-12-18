#!/usr/bin/env python3
"""
Script to analyze and visualize RLAIF evaluation results.
Creates:
1. Three pie charts showing win percentages for each head-to-head matchup
2. One bar chart showing overall win percentage for each model
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_data(filename):
    """Load the cleaned dataset."""
    print(f"Loading data from {filename}...")
    df = pd.read_csv(filename)
    print(f"Loaded {len(df)} rows")
    return df

def create_pie_charts(df):
    """Create three separate pie charts for each head-to-head comparison."""
    # Define the three comparisons
    comparisons = [
        ('DPO_vs_PPO', 'DPO vs PPO', 'pie_chart_dpo_vs_ppo.png'),
        ('PPO_vs_Base', 'PPO vs Base', 'pie_chart_ppo_vs_base.png'),
        ('DPO_vs_Base', 'DPO vs Base', 'pie_chart_dpo_vs_base.png')
    ]

    for comp_type, title, filename in comparisons:
        # Filter data for this comparison
        comp_data = df[df['comparison_type'] == comp_type]

        # Count winners
        winner_counts = comp_data['winner'].value_counts()

        # Create labels and sizes
        labels = []
        sizes = []
        colors = []

        for winner, count in winner_counts.items():
            percentage = (count / len(comp_data)) * 100
            labels.append(f'{winner}\n({count}, {percentage:.1f}%)')
            sizes.append(count)

            # Assign colors based on model
            if winner == 'DPO':
                colors.append('#2ecc71')  # Green
            elif winner == 'PPO':
                colors.append('#3498db')  # Blue
            elif winner == 'Base':
                colors.append('#e74c3c')  # Red
            elif winner == 'Tie':
                colors.append('#95a5a6')  # Gray
            else:
                colors.append('#f39c12')  # Orange for others

        # Create individual pie chart with larger figure size
        fig, ax = plt.subplots(figsize=(10, 8))

        # Create pie chart with larger text
        ax.pie(sizes, labels=labels, colors=colors, autopct='',
               startangle=90, textprops={'fontsize': 16, 'fontweight': 'bold'})
        ax.set_title(title, fontsize=22, fontweight='bold', pad=30)

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved pie chart to: {filename}")
        plt.close()

def create_overall_bar_chart(df):
    """Create a bar chart showing overall win percentage for each model."""
    # Count total wins for each model across all comparisons
    model_wins = {'DPO': 0, 'PPO': 0, 'Base': 0}
    model_total_matches = {'DPO': 0, 'PPO': 0, 'Base': 0}

    # Process each comparison type
    for comp_type in df['comparison_type'].unique():
        comp_data = df[df['comparison_type'] == comp_type]

        # Determine which models are in this comparison
        if comp_type == 'DPO_vs_PPO':
            models = ['DPO', 'PPO']
        elif comp_type == 'PPO_vs_Base':
            models = ['PPO', 'Base']
        elif comp_type == 'DPO_vs_Base':
            models = ['DPO', 'Base']

        # Count wins for each model in this comparison
        winner_counts = comp_data['winner'].value_counts()

        for model in models:
            # Each model participates in this comparison
            model_total_matches[model] += len(comp_data)

            # Count wins (excluding ties)
            if model in winner_counts.index:
                model_wins[model] += winner_counts[model]

    # Calculate win percentages
    win_percentages = {}
    for model in model_wins.keys():
        if model_total_matches[model] > 0:
            win_percentages[model] = (model_wins[model] / model_total_matches[model]) * 100
        else:
            win_percentages[model] = 0

    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 6))

    models = list(win_percentages.keys())
    percentages = list(win_percentages.values())
    colors_map = {'DPO': '#2ecc71', 'PPO': '#3498db', 'Base': '#e74c3c'}
    colors = [colors_map[model] for model in models]

    bars = ax.bar(models, percentages, color=colors, edgecolor='black', linewidth=1.5)

    # Add value labels on top of bars
    for bar, pct, model in zip(bars, percentages, models):
        height = bar.get_height()
        wins = model_wins[model]
        total = model_total_matches[model]
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{pct:.1f}%\n({wins}/{total})',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('Win Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_title('Overall Win Percentage Across All Matchups', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, max(percentages) + 10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig('bar_chart_overall_wins.png', dpi=300, bbox_inches='tight')
    print("Saved bar chart to: bar_chart_overall_wins.png")
    plt.close()

def print_statistics(df):
    """Print detailed statistics."""
    print("\n" + "="*60)
    print("DETAILED STATISTICS")
    print("="*60)

    # Overall statistics
    print("\nOverall Win Counts:")
    all_winners = df['winner'].value_counts()
    for winner, count in all_winners.items():
        percentage = (count / len(df)) * 100
        print(f"  {winner}: {count} ({percentage:.1f}%)")

    # Head-to-head statistics
    print("\nHead-to-Head Breakdown:")
    for comp_type in sorted(df['comparison_type'].unique()):
        comp_data = df[df['comparison_type'] == comp_type]
        print(f"\n  {comp_type} ({len(comp_data)} matchups):")
        winner_counts = comp_data['winner'].value_counts()
        for winner, count in winner_counts.items():
            percentage = (count / len(comp_data)) * 100
            print(f"    {winner}: {count} ({percentage:.1f}%)")

    # Model-specific win rates
    print("\nModel Win Rates (excluding ties):")
    model_stats = {}

    for comp_type in df['comparison_type'].unique():
        comp_data = df[df['comparison_type'] == comp_type]

        # Determine which models are in this comparison
        if comp_type == 'DPO_vs_PPO':
            models = ['DPO', 'PPO']
        elif comp_type == 'PPO_vs_Base':
            models = ['PPO', 'Base']
        elif comp_type == 'DPO_vs_Base':
            models = ['DPO', 'Base']

        winner_counts = comp_data['winner'].value_counts()

        for model in models:
            if model not in model_stats:
                model_stats[model] = {'wins': 0, 'total': 0}

            model_stats[model]['total'] += len(comp_data)
            if model in winner_counts.index:
                model_stats[model]['wins'] += winner_counts[model]

    for model in ['DPO', 'PPO', 'Base']:
        if model in model_stats:
            stats = model_stats[model]
            win_rate = (stats['wins'] / stats['total']) * 100
            print(f"  {model}: {stats['wins']}/{stats['total']} = {win_rate:.1f}%")

    print("="*60)

if __name__ == "__main__":
    # Load cleaned data
    df = load_data("rlaif_eval_results_cleaned.csv")

    # Print statistics
    print_statistics(df)

    # Create visualizations
    print("\nCreating visualizations...")
    create_pie_charts(df)
    create_overall_bar_chart(df)

    print("\n✓ Analysis complete!")
    print("  Generated files:")
    print("    - pie_chart_dpo_vs_ppo.png")
    print("    - pie_chart_ppo_vs_base.png")
    print("    - pie_chart_dpo_vs_base.png")
    print("    - bar_chart_overall_wins.png")
