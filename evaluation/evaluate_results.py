"""
Evaluation script for accessibility benchmark results.

This script analyzes benchmark results and provides detailed evaluation metrics.
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


def load_results(results_dir: str) -> List[Dict[str, Any]]:
    """
    Load all result files from a directory.
    
    Args:
        results_dir: Directory containing result JSON files
        
    Returns:
        List of result dictionaries
    """
    results = []
    for filename in os.listdir(results_dir):
        if filename.endswith('.json') and not filename.startswith('benchmark_summary'):
            filepath = os.path.join(results_dir, filename)
            with open(filepath, 'r') as f:
                results.append(json.load(f))
    return results


def calculate_category_metrics(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Calculate metrics by category.
    
    Args:
        results: List of result dictionaries
        
    Returns:
        Dictionary of category metrics
    """
    categories = {}
    
    for result in results:
        category = result.get('category', 'unknown')
        if category not in categories:
            categories[category] = {
                'total': 0,
                'passed': 0,
                'failed': 0,
                'errors': 0,
                'total_duration': 0,
                'tasks': []
            }
        
        categories[category]['total'] += 1
        categories[category]['tasks'].append(result)
        
        status = result.get('status')
        if status == 'PASS':
            categories[category]['passed'] += 1
        elif status == 'FAIL':
            categories[category]['failed'] += 1
        elif status == 'ERROR':
            categories[category]['errors'] += 1
        
        if 'duration_seconds' in result:
            categories[category]['total_duration'] += result['duration_seconds']
    
    # Calculate derived metrics
    for category, metrics in categories.items():
        total = metrics['total']
        if total > 0:
            metrics['pass_rate'] = (metrics['passed'] / total) * 100
            metrics['fail_rate'] = (metrics['failed'] / total) * 100
            metrics['error_rate'] = (metrics['errors'] / total) * 100
            metrics['avg_duration'] = metrics['total_duration'] / total
        else:
            metrics['pass_rate'] = 0
            metrics['fail_rate'] = 0
            metrics['error_rate'] = 0
            metrics['avg_duration'] = 0
    
    return categories


def calculate_difficulty_metrics(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Calculate metrics by difficulty level.
    
    Args:
        results: List of result dictionaries
        
    Returns:
        Dictionary of difficulty metrics
    """
    difficulties = {}
    
    for result in results:
        difficulty = result.get('difficulty', 'unknown')
        if difficulty not in difficulties:
            difficulties[difficulty] = {
                'total': 0,
                'passed': 0,
                'failed': 0,
                'errors': 0,
                'total_duration': 0
            }
        
        difficulties[difficulty]['total'] += 1
        
        status = result.get('status')
        if status == 'PASS':
            difficulties[difficulty]['passed'] += 1
        elif status == 'FAIL':
            difficulties[difficulty]['failed'] += 1
        elif status == 'ERROR':
            difficulties[difficulty]['errors'] += 1
        
        if 'duration_seconds' in result:
            difficulties[difficulty]['total_duration'] += result['duration_seconds']
    
    # Calculate derived metrics
    for difficulty, metrics in difficulties.items():
        total = metrics['total']
        if total > 0:
            metrics['pass_rate'] = (metrics['passed'] / total) * 100
            metrics['avg_duration'] = metrics['total_duration'] / total
        else:
            metrics['pass_rate'] = 0
            metrics['avg_duration'] = 0
    
    return difficulties


def identify_problematic_tasks(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Identify tasks that failed or had errors.
    
    Args:
        results: List of result dictionaries
        
    Returns:
        List of problematic task results
    """
    return [r for r in results if r.get('status') in ['FAIL', 'ERROR']]


def generate_evaluation_report(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate a comprehensive evaluation report.
    
    Args:
        results: List of result dictionaries
        
    Returns:
        Evaluation report dictionary
    """
    total_tasks = len(results)
    passed = sum(1 for r in results if r.get('status') == 'PASS')
    failed = sum(1 for r in results if r.get('status') == 'FAIL')
    errors = sum(1 for r in results if r.get('status') == 'ERROR')
    
    total_duration = sum(r.get('duration_seconds', 0) for r in results)
    avg_duration = total_duration / total_tasks if total_tasks > 0 else 0
    
    category_metrics = calculate_category_metrics(results)
    difficulty_metrics = calculate_difficulty_metrics(results)
    problematic_tasks = identify_problematic_tasks(results)
    
    return {
        'timestamp': datetime.now().isoformat(),
        'total_tasks': total_tasks,
        'passed': passed,
        'failed': failed,
        'errors': errors,
        'pass_rate': (passed / total_tasks * 100) if total_tasks > 0 else 0,
        'total_duration': total_duration,
        'avg_duration': avg_duration,
        'category_metrics': category_metrics,
        'difficulty_metrics': difficulty_metrics,
        'problematic_tasks': len(problematic_tasks),
        'problematic_task_details': problematic_tasks
    }


def print_evaluation_report(report: Dict[str, Any]) -> None:
    """
    Print the evaluation report to console.
    
    Args:
        report: Evaluation report dictionary
    """
    console.print("\n")
    console.print(Panel(
        "[bold cyan]Accessibility Benchmark Evaluation Report[/bold cyan]",
        border_style="cyan"
    ))
    
    # Overall metrics
    console.print("\n[bold]OVERALL METRICS[/bold]")
    console.print(f"Total Tasks: {report['total_tasks']}")
    console.print(f"Passed: [green]{report['passed']}[/green] ✓")
    console.print(f"Failed: [red]{report['failed']}[/red] ✗")
    console.print(f"Errors: [yellow]{report['errors']}[/yellow] ⚠")
    console.print(f"Pass Rate: [bold]{report['pass_rate']:.2f}%[/bold]")
    console.print(f"Total Duration: {report['total_duration']:.2f}s")
    console.print(f"Average Duration: {report['avg_duration']:.2f}s per task")
    
    # Category metrics table
    console.print("\n[bold]METRICS BY CATEGORY[/bold]")
    category_table = Table(show_header=True, header_style="bold magenta")
    category_table.add_column("Category", style="cyan")
    category_table.add_column("Total", justify="right")
    category_table.add_column("Passed", justify="right", style="green")
    category_table.add_column("Failed", justify="right", style="red")
    category_table.add_column("Errors", justify="right", style="yellow")
    category_table.add_column("Pass Rate", justify="right")
    category_table.add_column("Avg Duration", justify="right")
    
    for category, metrics in report['category_metrics'].items():
        category_table.add_row(
            category,
            str(metrics['total']),
            str(metrics['passed']),
            str(metrics['failed']),
            str(metrics['errors']),
            f"{metrics['pass_rate']:.1f}%",
            f"{metrics['avg_duration']:.2f}s"
        )
    
    console.print(category_table)
    
    console.print("\n[bold]METRICS BY DIFFICULTY[/bold]")
    difficulty_table = Table(show_header=True, header_style="bold magenta")
    difficulty_table.add_column("Difficulty", style="cyan")
    difficulty_table.add_column("Total", justify="right")
    difficulty_table.add_column("Passed", justify="right", style="green")
    difficulty_table.add_column("Pass Rate", justify="right")
    difficulty_table.add_column("Avg Duration", justify="right")
    
    for difficulty in ['easy', 'medium', 'hard']:
        if difficulty in report['difficulty_metrics']:
            metrics = report['difficulty_metrics'][difficulty]
            difficulty_table.add_row(
                difficulty.capitalize(),
                str(metrics['total']),
                str(metrics['passed']),
                f"{metrics['pass_rate']:.1f}%",
                f"{metrics['avg_duration']:.2f}s"
            )
    
    console.print(difficulty_table)
    
    if report['problematic_tasks'] > 0:
        console.print(f"\n[bold yellow]PROBLEMATIC TASKS ({report['problematic_tasks']})[/bold yellow]")
        problem_table = Table(show_header=True, header_style="bold red")
        problem_table.add_column("Task ID", style="cyan")
        problem_table.add_column("Category", style="magenta")
        problem_table.add_column("Status", style="red")
        problem_table.add_column("Error/Issue", style="dim")
        
        for task in report['problematic_task_details'][:10]:  # Show first 10
            error_msg = task.get('error', task.get('message', 'N/A'))
            if len(error_msg) > 50:
                error_msg = error_msg[:47] + "..."
            
            problem_table.add_row(
                task.get('task_id', 'unknown'),
                task.get('category', 'unknown'),
                task.get('status', 'UNKNOWN'),
                error_msg
            )
        
        console.print(problem_table)
        
        if report['problematic_tasks'] > 10:
            console.print(f"[dim]... and {report['problematic_tasks'] - 10} more[/dim]")


def save_evaluation_report(report: Dict[str, Any], output_file: str) -> None:
    """
    Save evaluation report to JSON file.
    
    Args:
        report: Evaluation report dictionary
        output_file: Path to output file
    """
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    console.print(f"\n[green]Evaluation report saved to: {output_file}[/green]")


def main():
    """Main entry point for evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate accessibility benchmark results"
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory containing result files"
    )
    parser.add_argument(
        "--output-file",
        help="Path to save evaluation report (optional)"
    )
    
    args = parser.parse_args()
    
    console.print(f"\n[cyan]Loading results from {args.results_dir}...[/cyan]")
    results = load_results(args.results_dir)
    
    if not results:
        console.print("[red]No results found![/red]")
        return
    
    console.print(f"[green]Loaded {len(results)} results[/green]")
    
    console.print("\n[cyan]Generating evaluation report...[/cyan]")
    report = generate_evaluation_report(results)
    
    print_evaluation_report(report)
    
    if args.output_file:
        save_evaluation_report(report, args.output_file)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(args.results_dir, f"evaluation_report_{timestamp}.json")
        save_evaluation_report(report, output_file)


if __name__ == "__main__":
    main()

