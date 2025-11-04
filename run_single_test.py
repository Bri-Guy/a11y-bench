"""
Script to run a single accessibility task.
"""

import asyncio
import argparse
import os
from datetime import datetime
from dotenv import load_dotenv
from rich.console import Console

from stagehand import Stagehand, StagehandConfig
from utils import load_task_by_id, save_result, execute_task

console = Console()
load_dotenv()


async def run_single_task(task_id: str, env: str = "LOCAL", model: str = "gemini-2.5-computer-use-preview-10-2025"):
    """Run a single accessibility task."""
    
    task = load_task_by_id(task_id)
    if not task:
        console.print(f"[red]Task {task_id} not found[/red]")
        return
    
    console.print(f"\n[cyan]Task:[/cyan] {task.get('task')}")
    console.print(f"[cyan]URL:[/cyan] {task.get('url')}")
    console.print(f"[cyan]Category:[/cyan] {task.get('category')}")
    console.print(f"[cyan]Success Criteria:[/cyan] {task.get('success_criteria')}\n")
    
    config = StagehandConfig(
        env=env,
        model_client_options={"apiKey": os.getenv("MODEL_API_KEY")},
        verbose=2
    )
    
    stagehand = Stagehand(config)
    await stagehand.init()
    
    try:
        console.print("[cyan]Executing task...[/cyan]")
        result = await execute_task(stagehand, task, model)
        
        status = result.get('status')
        status_color = "green" if status == "PASS" else "red" if status == "FAIL" else "yellow"
        
        console.print(f"\n[bold {status_color}]Status: {status}[/bold {status_color}]")
        console.print(f"[cyan]Duration:[/cyan] {result.get('duration_seconds', 0):.2f}s")
        
        if result.get('message'):
            console.print(f"[cyan]Message:[/cyan] {result.get('message')}")
        
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
    parser.add_argument("--model", default="gemini-2.5-computer-use-preview-10-2025")
    
    args = parser.parse_args()
    asyncio.run(run_single_task(args.task_id, args.env, args.model))


if __name__ == "__main__":
    main()
