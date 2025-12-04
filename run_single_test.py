"""
Script to run a single accessibility task.
"""

import asyncio
import argparse
from dotenv import load_dotenv
from rich.console import Console

from stagehand import Stagehand, StagehandConfig
from utils import load_task_by_id, save_result, execute_task, DEFAULT_EVALUATOR_MODEL

console = Console()
load_dotenv()


async def run_single_task(
    task_id: str,
    env: str = "LOCAL",
    model: str = "gemini-2.5-computer-use-preview-10-2025",
    evaluator_model: str = DEFAULT_EVALUATOR_MODEL,
):
    """Run a single accessibility task."""
    
    task = load_task_by_id(task_id)
    if not task:
        console.print(f"[red]Task {task_id} not found[/red]")
        return
    
    console.print(f"\n[cyan]Task:[/cyan] {task.get('task')}")
    console.print(f"[cyan]URL:[/cyan] {task.get('url')}")
    console.print(f"[cyan]Category:[/cyan] {task.get('category')}")
    console.print(f"[cyan]Success Criteria:[/cyan] {task.get('success_criteria')}")
    console.print(f"[cyan]Agent Model:[/cyan] {model}")
    console.print(f"[cyan]Evaluator Model:[/cyan] {evaluator_model}\n")
    
    config = StagehandConfig(
        env=env,
        verbose=2
    )
    
    stagehand = Stagehand(config)
    await stagehand.init()
    
    try:
        console.print("[cyan]Executing task...[/cyan]")
        result = await execute_task(stagehand, task, model, evaluator_model)
        
        status = result.get('status')
        status_color = "green" if status == "PASS" else "red" if status == "FAIL" else "yellow"
        
        console.print(f"\n[bold {status_color}]Status: {status}[/bold {status_color}]")
        console.print(f"[cyan]Agent completed:[/cyan] {result.get('agent_completed')}")
        console.print(f"[cyan]Execution time:[/cyan] {result.get('execution_duration_seconds', 0):.2f}s")
        console.print(f"[cyan]Verification time:[/cyan] {result.get('verification_duration_seconds', 0):.2f}s")
        console.print(f"[cyan]Total time:[/cyan] {result.get('total_duration_seconds', 0):.2f}s")
        
        if result.get('agent_message'):
            console.print(f"[cyan]Agent message:[/cyan] {result.get('agent_message')}")
        
        if result.get('verification_reasoning'):
            console.print(f"[cyan]Verification reasoning:[/cyan] {result.get('verification_reasoning')}")
        
        if result.get('error'):
            console.print(f"[red]Error:[/red] {result.get('error')}")
        
        result_path = save_result(result)
        console.print(f"\n[green]Result saved to: {result_path}[/green]")
        
    finally:
        await stagehand.close()


def main():
    parser = argparse.ArgumentParser(description="Run a single accessibility task")
    parser.add_argument("task_id", help="ID of the task to run (e.g., a11y_001)")
    parser.add_argument("--env", choices=["LOCAL", "BROWSERBASE"], default="LOCAL")
    parser.add_argument("--model", default="gemini-2.5-computer-use-preview-10-2025",
                        help="Agent model to use for task execution")
    parser.add_argument("--evaluator-model", default=DEFAULT_EVALUATOR_MODEL,
                        help=f"Model to use for verification (default: {DEFAULT_EVALUATOR_MODEL})")
    
    args = parser.parse_args()
    asyncio.run(run_single_task(args.task_id, args.env, args.model, args.evaluator_model))


if __name__ == "__main__":
    main()
