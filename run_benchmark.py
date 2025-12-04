"""
Main script to run the Accessibility Benchmark (A11y-Bench)
"""

import asyncio
import argparse
import traceback
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table

from stagehand import Stagehand, StagehandConfig
from utils import (
    load_tasks,
    filter_tasks_by_category,
    filter_tasks_by_difficulty,
    save_result,
    save_benchmark_summary,
    execute_task,
    DEFAULT_EVALUATOR_MODEL,
)

console = Console()
load_dotenv()


async def run_benchmark(
    category=None,
    difficulty=None,
    task_ids=None,
    env="LOCAL",
    model="gemini-2.5-computer-use-preview-10-2025",
    evaluator_model=DEFAULT_EVALUATOR_MODEL,
):
    """Run the accessibility benchmark."""
    
    console.print("\n[cyan]Loading tasks...[/cyan]")
    tasks = load_tasks()
    
    if task_ids:
        tasks = [t for t in tasks if t.get('id') in task_ids]
    elif category:
        tasks = filter_tasks_by_category(tasks, category)
    if difficulty:
        tasks = filter_tasks_by_difficulty(tasks, difficulty)
    
    if not tasks:
        console.print("[red]No tasks to run![/red]")
        return
    
    console.print(f"[green]Running {len(tasks)} tasks[/green]")
    console.print(f"[cyan]Agent model:[/cyan] {model}")
    console.print(f"[cyan]Evaluator model:[/cyan] {evaluator_model}\n")
    
    config = StagehandConfig(
        env=env,
        verbose=0
    )
    
    stagehand = Stagehand(config)
    await stagehand.init()
    
    results = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console
    ) as progress:
        task_progress = progress.add_task("[cyan]Running tasks...", total=len(tasks))
        
        for task in tasks:
            result = await execute_task(stagehand, task, model, evaluator_model)
            results.append(result)
            save_result(result)
            progress.update(task_progress, advance=1)
    
    await stagehand.close()
    
    summary_path = save_benchmark_summary(results)
    
    passed = sum(1 for r in results if r.get('status') == 'PASS')
    failed = sum(1 for r in results if r.get('status') == 'FAIL')
    errors = sum(1 for r in results if r.get('status') == 'ERROR')
    
    console.print("\n" + "="*60)
    console.print("[bold]BENCHMARK RESULTS[/bold]")
    console.print("="*60)
    console.print(f"Total: {len(results)}")
    console.print(f"[green]Passed: {passed}[/green]")
    console.print(f"[red]Failed: {failed}[/red]")
    console.print(f"[yellow]Errors: {errors}[/yellow]")
    console.print(f"Pass Rate: {(passed/len(results)*100):.1f}%")
    
    table = Table(title="\nTask Results")
    table.add_column("Task ID", style="cyan")
    table.add_column("Category", style="magenta")
    table.add_column("Status", style="bold")
    
    for result in results:
        status = result.get('status', 'UNKNOWN')
        status_color = "green" if status == "PASS" else "red" if status == "FAIL" else "yellow"
        table.add_row(
            result.get('task_id', 'unknown'),
            result.get('category', 'unknown'),
            f"[{status_color}]{status}[/{status_color}]"
        )
    
    console.print(table)
    console.print(f"\n[green]Summary saved to: {summary_path}[/green]")


def main():
    parser = argparse.ArgumentParser(description="Run the Accessibility Benchmark")
    parser.add_argument("--category", help="Filter by category")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], help="Filter by difficulty")
    parser.add_argument("--task-ids", nargs="+", help="Specific task IDs to run")
    parser.add_argument("--env", choices=["LOCAL", "BROWSERBASE"], default="LOCAL")
    parser.add_argument("--model", default="gemini-2.5-computer-use-preview-10-2025",
                        help="Agent model to use for task execution")
    parser.add_argument("--evaluator-model", default=DEFAULT_EVALUATOR_MODEL,
                        help=f"Model to use for verification (default: {DEFAULT_EVALUATOR_MODEL})")
    
    args = parser.parse_args()
    
    try:
        asyncio.run(run_benchmark(
            category=args.category,
            difficulty=args.difficulty,
            task_ids=args.task_ids,
            env=args.env,
            model=args.model,
            evaluator_model=args.evaluator_model,
        ))
    except KeyboardInterrupt:
        console.print("\n[yellow]Benchmark interrupted[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {str(e)}[/red]")
        console.print(traceback.format_exc())


if __name__ == "__main__":
    main()
