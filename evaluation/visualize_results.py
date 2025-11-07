#!/usr/bin/env python3
"""
Visualization script for A11y benchmark results.
Generates graphs comparing performance across different model providers.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


def load_result_file(filepath: Path) -> Dict[str, Any]:
    """Load a single result JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def extract_provider_name(filepath: Path) -> str:
    """Extract provider name from filename (e.g., 'openai_corrected.json' -> 'OpenAI')."""
    name = filepath.stem.replace('_corrected', '').replace('_results', '')
    
    if name.lower() == 'openai':
        return 'OpenAI'
    elif name.lower() == 'gemini':
        return 'Gemini'
    elif name.lower() == 'anthropic' or name.lower() == 'claude':
        return 'Anthropic'
    else:
        return name.capitalize()


def calculate_success_rate(data: Dict[str, Any]) -> float:
    """Calculate success rate as a percentage from actual results."""
    # Calculate from actual results, don't trust the header summary
    total = 0
    passed = 0
    for result in data['results']:
        total += 1
        if result['status'] == 'PASS':
            passed += 1
    
    if total == 0:
        return 0.0
    return (passed / total) * 100


def get_category_stats(data: Dict[str, Any]) -> Dict[str, float]:
    """Calculate success rate per category."""
    category_stats = {}
    for category, stats in data['categories'].items():
        if stats['total'] > 0:
            category_stats[category] = (stats['passed'] / stats['total']) * 100
        else:
            category_stats[category] = 0.0
    return category_stats


def get_difficulty_stats(data: Dict[str, Any]) -> Dict[str, float]:
    """Calculate success rate per difficulty level."""
    difficulty_counts = defaultdict(lambda: {'passed': 0, 'total': 0})
    
    for result in data['results']:
        difficulty = result.get('difficulty', 'Unknown')
        difficulty_counts[difficulty]['total'] += 1
        if result['status'] == 'PASS':
            difficulty_counts[difficulty]['passed'] += 1
    
    difficulty_stats = {}
    for difficulty, counts in difficulty_counts.items():
        if counts['total'] > 0:
            difficulty_stats[difficulty] = (counts['passed'] / counts['total']) * 100
        else:
            difficulty_stats[difficulty] = 0.0
    
    return difficulty_stats


def get_median_success_duration(data: Dict[str, Any]) -> float:
    """Calculate median duration for successful tasks."""
    success_durations = []
    for result in data['results']:
        if result['status'] == 'PASS' and 'duration_seconds' in result:
            success_durations.append(result['duration_seconds'])
    
    if not success_durations:
        return 0.0
    
    return np.median(success_durations)


def create_overall_performance_chart(provider_stats: Dict[str, Dict], output_dir: Path):
    """Create bar chart comparing overall performance across providers."""
    providers = list(provider_stats.keys())
    success_rates = [provider_stats[p]['overall_success_rate'] for p in providers]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(providers, success_rates, color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(providers)])
    
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('Overall Performance Comparison Across Model Providers', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'overall_performance.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'overall_performance.png'}")
    plt.close()


def create_category_performance_chart(provider_stats: Dict[str, Dict], output_dir: Path):
    """Create grouped bar chart comparing performance by category."""
    providers = list(provider_stats.keys())
    
    all_categories = set()
    for stats in provider_stats.values():
        all_categories.update(stats['category_stats'].keys())
    categories = sorted(list(all_categories))
    
    if not categories:
        print("No category data available, skipping category chart.")
        return
    
    x = np.arange(len(categories))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, provider in enumerate(providers):
        values = [provider_stats[provider]['category_stats'].get(cat, 0) for cat in categories]
        offset = width * (i - len(providers)/2 + 0.5)
        bars = ax.bar(x + offset, values, width, label=provider, color=colors[i])
        
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.0f}%',
                        ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('Performance Comparison by Category', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([cat.capitalize() for cat in categories])
    ax.set_ylim(0, 110)
    ax.legend()
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'category_performance.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'category_performance.png'}")
    plt.close()


def create_difficulty_performance_chart(provider_stats: Dict[str, Dict], output_dir: Path):
    """Create grouped bar chart comparing performance by difficulty."""
    providers = list(provider_stats.keys())
    
    all_difficulties = set()
    for stats in provider_stats.values():
        all_difficulties.update(stats['difficulty_stats'].keys())
    
    difficulty_order = ['Easy', 'Moderate', 'Complex', 'Hard']
    difficulties = [d for d in difficulty_order if d in all_difficulties]
    difficulties.extend([d for d in sorted(all_difficulties) if d not in difficulty_order])
    
    if not difficulties:
        print("No difficulty data available, skipping difficulty chart.")
        return
    
    x = np.arange(len(difficulties))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, provider in enumerate(providers):
        values = [provider_stats[provider]['difficulty_stats'].get(diff, 0) for diff in difficulties]
        offset = width * (i - len(providers)/2 + 0.5)
        bars = ax.bar(x + offset, values, width, label=provider, color=colors[i])
        
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.0f}%',
                        ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('Performance Comparison by Task Difficulty', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(difficulties)
    ax.set_ylim(0, 110)
    ax.legend()
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'difficulty_performance.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'difficulty_performance.png'}")
    plt.close()


def create_median_duration_chart(provider_stats: Dict[str, Dict], output_dir: Path):
    """Create bar chart comparing median duration for successful tasks."""
    providers = list(provider_stats.keys())
    median_durations = [provider_stats[p]['median_success_duration'] for p in providers]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(providers, median_durations, color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(providers)])
    
    ax.set_ylabel('Median Duration (seconds)', fontsize=12)
    ax.set_title('Median Time Per Successful Task', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}s',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'median_duration.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'median_duration.png'}")
    plt.close()


def generate_summary_report(provider_stats: Dict[str, Dict], output_dir: Path):
    """Generate a text summary report."""
    report_path = output_dir / 'summary_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("A11Y BENCHMARK RESULTS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        for provider, stats in provider_stats.items():
            f.write(f"\n{provider.upper()}\n")
            f.write("-" * 40 + "\n")
            f.write(f"Overall Success Rate: {stats['overall_success_rate']:.2f}%\n")
            f.write(f"Total Tasks: {stats['total_tasks']}\n")
            f.write(f"Passed: {stats['passed']}\n")
            f.write(f"Failed: {stats['failed']}\n")
            f.write(f"Errors: {stats['errors']}\n")
            f.write(f"Median Success Duration: {stats['median_success_duration']:.2f}s\n")
            
            if stats['category_stats']:
                f.write(f"\nCategory Breakdown:\n")
                for category, rate in sorted(stats['category_stats'].items()):
                    f.write(f"  {category.capitalize()}: {rate:.2f}%\n")
            
            if stats['difficulty_stats']:
                f.write(f"\nDifficulty Breakdown:\n")
                for difficulty, rate in sorted(stats['difficulty_stats'].items()):
                    f.write(f"  {difficulty}: {rate:.2f}%\n")
            
            f.write("\n")
    
    print(f"Saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate visualization graphs from A11y benchmark results'
    )
    parser.add_argument(
        'result_files',
        nargs='+',
        type=Path,
        help='Path to result JSON files (e.g., results/openai_corrected.json results/gemini_corrected.json)'
    )
    parser.add_argument(
        '-o', '--output',
        type=Path,
        default=Path('results/visualizations'),
        help='Output directory for generated graphs (default: results/visualizations)'
    )
    
    args = parser.parse_args()
    
    args.output.mkdir(parents=True, exist_ok=True)
    
    provider_stats = {}
    for filepath in args.result_files:
        if not filepath.exists():
            print(f"Warning: File not found: {filepath}")
            continue
        
        print(f"Loading: {filepath}")
        data = load_result_file(filepath)
        provider_name = extract_provider_name(filepath)
        
        total = 0
        passed = 0
        failed = 0
        errors = 0
        for result in data['results']:
            total += 1
            if result['status'] == 'PASS':
                passed += 1
            elif result['status'] == 'FAIL':
                failed += 1
            elif result['status'] == 'ERROR':
                errors += 1
        
        provider_stats[provider_name] = {
            'overall_success_rate': calculate_success_rate(data),
            'category_stats': get_category_stats(data),
            'difficulty_stats': get_difficulty_stats(data),
            'median_success_duration': get_median_success_duration(data),
            'total_tasks': total,
            'passed': passed,
            'failed': failed,
            'errors': errors
        }
    
    if not provider_stats:
        print("Error: No valid result files loaded.")
        return
    
    print(f"\nGenerating visualizations for {len(provider_stats)} providers...")
    print(f"Providers: {', '.join(provider_stats.keys())}\n")
    
    create_overall_performance_chart(provider_stats, args.output)
    create_category_performance_chart(provider_stats, args.output)
    create_difficulty_performance_chart(provider_stats, args.output)
    create_median_duration_chart(provider_stats, args.output)
    generate_summary_report(provider_stats, args.output)
    
    print(f"\n✓ All visualizations saved to: {args.output}")


if __name__ == '__main__':
    main()

