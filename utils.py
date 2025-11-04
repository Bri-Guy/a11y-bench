"""
Utility functions for the Accessibility Benchmark (A11y-Bench)
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional
from pathlib import Path


def load_tasks(filepath: str = "data/accessibility_tasks.jsonl") -> List[Dict[str, Any]]:
    """Load accessibility tasks from a JSONL file."""
    tasks = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))
    return tasks


def load_task_by_id(task_id: str, filepath: str = "data/accessibility_tasks.jsonl") -> Optional[Dict[str, Any]]:
    """Load a specific task by its ID."""
    tasks = load_tasks(filepath)
    for task in tasks:
        if task.get('id') == task_id:
            return task
    return None


def filter_tasks_by_category(tasks: List[Dict[str, Any]], category: str) -> List[Dict[str, Any]]:
    """Filter tasks by category."""
    return [task for task in tasks if task.get('category') == category]


def filter_tasks_by_difficulty(tasks: List[Dict[str, Any]], difficulty: str) -> List[Dict[str, Any]]:
    """Filter tasks by difficulty level."""
    return [task for task in tasks if task.get('difficulty') == difficulty]


def save_result(result: Dict[str, Any], output_dir: str = "results") -> str:
    """Save a benchmark result to a JSON file."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_id = result.get('task_id', 'unknown')
    filename = f"{task_id}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(result, f, indent=2)
    
    return filepath


def save_benchmark_summary(results: List[Dict[str, Any]], output_dir: str = "results") -> str:
    """Save a summary of benchmark results."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    total_tasks = len(results)
    passed = sum(1 for r in results if r.get('status') == 'PASS')
    failed = sum(1 for r in results if r.get('status') == 'FAIL')
    errors = sum(1 for r in results if r.get('status') == 'ERROR')
    
    categories = {}
    for result in results:
        category = result.get('category', 'unknown')
        if category not in categories:
            categories[category] = {'total': 0, 'passed': 0, 'failed': 0, 'errors': 0}
        categories[category]['total'] += 1
        if result.get('status') == 'PASS':
            categories[category]['passed'] += 1
        elif result.get('status') == 'FAIL':
            categories[category]['failed'] += 1
        elif result.get('status') == 'ERROR':
            categories[category]['errors'] += 1
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_tasks': total_tasks,
        'passed': passed,
        'failed': failed,
        'errors': errors,
        'pass_rate': f"{(passed / total_tasks * 100):.2f}%" if total_tasks > 0 else "0%",
        'categories': categories,
        'results': results
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark_summary_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return filepath


async def execute_task(stagehand, task: Dict[str, Any], model: str) -> Dict[str, Any]:
    """
    Execute a single accessibility task using Stagehand agent.
    
    Args:
        stagehand: Initialized Stagehand instance
        task: Task dictionary
        model: Model name to use
        
    Returns:
        Result dictionary with status and metrics
    """
    try:
        await stagehand.page.goto(task.get('url'))
        
        task_instruction = f"""
Task: {task.get('task')}
Success Criteria: {task.get('success_criteria')}

Please execute this task and provide a clear assessment.
"""
        
        agent = stagehand.agent(
            model=model,
            instructions=task_instruction,
            options={"apiKey": os.getenv("GEMINI_API_KEY")}
        )
        
        start_time = datetime.now()
        agent_result = await agent.execute(
            instruction=task.get('task'),
            max_steps=20,
            auto_screenshot=True
        )
        duration = (datetime.now() - start_time).total_seconds()
        
        status = "PASS" if agent_result.completed else "FAIL"
        
        return {
            'task_id': task.get('id'),
            'task': task.get('task'),
            'url': task.get('url'),
            'category': task.get('category'),
            'difficulty': task.get('difficulty'),
            'status': status,
            'completed': agent_result.completed,
            'message': agent_result.message,
            'duration_seconds': duration,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            'task_id': task.get('id'),
            'task': task.get('task'),
            'url': task.get('url'),
            'category': task.get('category'),
            'difficulty': task.get('difficulty'),
            'status': 'ERROR',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }
