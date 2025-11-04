"""
Script to compare multiple benchmark runs.

This allows you to track improvement over time or compare different models.
"""

import argparse
import json
import os
from typing import Dict, Any, List

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


def load_summary(filepath: str) -> Dict[str, Any]:
    """
    Load a benchmark summary file.
    
    Args:
        filepath: Path to summary JSON file
        
    Returns:
        Summary dictionary
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def compare_summaries(summaries: List[tuple]) -> Dict[str, Any]:
    """
    Compare multiple benchmark summaries.
    
    Args:
        summaries: List of (name, summary_dict) tuples
        
    Returns:
        Comparison dictionary
    """
    comparison = {
        'runs': [],
        'category_comparison': {},
        'improvement': []
    }
    
    for name, summary in summaries:
        run_data = {
            'name': name,
            'timestamp': summary.get('timestamp'),
            'total_tasks': summary.get('total_tasks'),
            'passed': summary.get('passed'),
            'failed': summary.get('failed'),
            'errors': summary.get('errors'),
            'pass_rate': summary.get('pass_rate')
        }
        comparison['runs'].append(run_data)
    
    # Compare categories across runs
    all_categories = set()
    for _, summary in summaries:
        if 'categories' in summary:
            all_categories.update(summary['categories'].keys())
    
    for category in all_categories:
        comparison['category_comparison'][category] = []
        for name, summary in summaries:
            if 'categories' in summary and category in summary['categories']:
                cat_data = summary['categories'][category]
                comparison['category_comparison'][category].append({
                    'run': name,
                    'total': cat_data.get('total', 0),
                    'passed': cat_data.get('passed', 0),
                    'pass_rate': cat_data.get('pass_rate', 0)
                })
    
    # Calculate improvement if comparing exactly 2 runs
    if len(summaries) == 2:
        old_pass_rate = float(summaries[0][1].get('pass_rate', '0').rstrip('%'))
        new_pass_rate = float(summaries[1][1].get('pass_rate', '0').rstrip('%'))
        improvement = new_pass_rate - old_pass_rate
        
        comparison['improvement'] = {
            'overall': improvement,
            'old_pass_rate': old_pass_rate,
            'new_pass_rate': new_pass_rate
        }
    
    return comparison


def print_comparison(comparison: Dict[str, Any]) -> None:
    """
    Print comparison to console.
    
    Args:
        comparison: Comparison dictionary
    """
    console.print("\n")
    console.print(Panel(
        "[bold cyan]Benchmark Run Comparison[/bold cyan]",
        border_style="cyan"
    ))
    
    console.print("\n[bold]OVERALL COMPARISON[/bold]")
    overall_table = Table(show_header=True, header_style="bold magenta")
    overall_table.add_column("Run", style="cyan")
    overall_table.add_column("Timestamp", style="dim")
    overall_table.add_column("Total", justify="right")
    overall_table.add_column("Passed", justify="right", style="green")
    overall_table.add_column("Failed", justify="right", style="red")
    overall_table.add_column("Errors", justify="right", style="yellow")
    overall_table.add_column("Pass Rate", justify="right", style="bold")
    
    for run in comparison['runs']:
        overall_table.add_row(
            run['name'],
            run.get('timestamp', 'N/A')[:19],  # Just date and time
            str(run['total_tasks']),
            str(run['passed']),
            str(run['failed']),
            str(run['errors']),
            run['pass_rate']
        )
    
    console.print(overall_table)
    
    # Improvement summary (if comparing 2 runs)
    if comparison.get('improvement'):
        imp = comparison['improvement']
        console.print("\n[bold]IMPROVEMENT SUMMARY[/bold]")
        
        improvement_value = imp['overall']
        if improvement_value > 0:
            color = "green"
            arrow = "↑"
        elif improvement_value < 0:
            color = "red"
            arrow = "↓"
        else:
            color = "yellow"
            arrow = "→"
        
        console.print(f"Old Pass Rate: {imp['old_pass_rate']:.2f}%")
        console.print(f"New Pass Rate: {imp['new_pass_rate']:.2f}%")
        console.print(f"Change: [{color}]{arrow} {abs(improvement_value):.2f}%[/{color}]")
    
    console.print("\n[bold]CATEGORY COMPARISON[/bold]")
    
    for category, runs in comparison['category_comparison'].items():
        if len(runs) > 0:
            console.print(f"\n[cyan]{category}[/cyan]")
            cat_table = Table(show_header=True, header_style="bold")
            cat_table.add_column("Run", style="dim")
            cat_table.add_column("Total", justify="right")
            cat_table.add_column("Passed", justify="right", style="green")
            cat_table.add_column("Pass Rate", justify="right")
            
            for run_data in runs:
                cat_table.add_row(
                    run_data['run'],
                    str(run_data['total']),
                    str(run_data['passed']),
                    f"{run_data['pass_rate']:.1f}%"
                )
            
            console.print(cat_table)


def save_comparison(comparison: Dict[str, Any], output_file: str) -> None:
    """
    Save comparison to JSON file.
    
    Args:
        comparison: Comparison dictionary
        output_file: Path to output file
    """
    with open(output_file, 'w') as f:
        json.dump(comparison, f, indent=2)
    console.print(f"\n[green]Comparison saved to: {output_file}[/green]")


def main():
    """Main entry point for comparison script."""
    parser = argparse.ArgumentParser(
        description="Compare multiple benchmark runs"
    )
    parser.add_argument(
        "summaries",
        nargs="+",
        help="Paths to benchmark summary files to compare"
    )
    parser.add_argument(
        "--names",
        nargs="+",
        help="Names for each run (optional, defaults to filenames)"
    )
    parser.add_argument(
        "--output-file",
        help="Path to save comparison report"
    )
    
    args = parser.parse_args()
    
    summaries = []
    for i, filepath in enumerate(args.summaries):
        if not os.path.exists(filepath):
            console.print(f"[red]File not found: {filepath}[/red]")
            continue
        
        summary = load_summary(filepath)
        
        if args.names and i < len(args.names):
            name = args.names[i]
        else:
            name = os.path.basename(filepath)
        
        summaries.append((name, summary))
    
    if len(summaries) < 2:
        console.print("[red]Need at least 2 summaries to compare[/red]")
        return
    
    console.print(f"[cyan]Comparing {len(summaries)} benchmark runs...[/cyan]")
    
    comparison = compare_summaries(summaries)
    
    print_comparison(comparison)
    
    if args.output_file:
        save_comparison(comparison, args.output_file)


if __name__ == "__main__":
    main()

