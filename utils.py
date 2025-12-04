"""
Utility functions for the Accessibility Benchmark (A11y-Bench)
"""

import base64
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


def _get_provider_from_model(model: str) -> str:
    """Determine the provider from the model name."""
    model_lower = model.lower()
    if "claude" in model_lower:
        return "anthropic"
    elif "gemini" in model_lower:
        return "google"
    else:
        return "openai"


async def _verify_with_openai(
    initial_screenshot_b64: str,
    final_screenshot_b64: str,
    initial_url: str,
    final_url: str,
    task_description: str,
    success_criteria: str,
    agent_message: str,
) -> Dict[str, Any]:
    """Verify success criteria using OpenAI's vision API."""
    from openai import OpenAI
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    prompt = _build_verification_prompt(
        initial_url=initial_url,
        final_url=final_url,
        task_description=task_description,
        success_criteria=success_criteria,
        agent_message=agent_message,
    )
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "text",
                        "text": "INITIAL STATE (before agent started):"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{initial_screenshot_b64}",
                            "detail": "high"
                        }
                    },
                    {
                        "type": "text",
                        "text": "FINAL STATE (after agent finished):"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{final_screenshot_b64}",
                            "detail": "high"
                        }
                    },
                ],
            }
        ],
        max_tokens=1024,
    )
    
    return _parse_verification_response(response.choices[0].message.content)


async def _verify_with_anthropic(
    initial_screenshot_b64: str,
    final_screenshot_b64: str,
    initial_url: str,
    final_url: str,
    task_description: str,
    success_criteria: str,
    agent_message: str,
) -> Dict[str, Any]:
    """Verify success criteria using Anthropic's vision API."""
    from anthropic import Anthropic
    
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    prompt = _build_verification_prompt(
        initial_url=initial_url,
        final_url=final_url,
        task_description=task_description,
        success_criteria=success_criteria,
        agent_message=agent_message,
    )
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "text", "text": "INITIAL STATE (before agent started):"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": initial_screenshot_b64,
                        },
                    },
                    {"type": "text", "text": "FINAL STATE (after agent finished):"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": final_screenshot_b64,
                        },
                    },
                ],
            }
        ],
    )
    
    return _parse_verification_response(response.content[0].text)


async def _verify_with_google(
    initial_screenshot_b64: str,
    final_screenshot_b64: str,
    initial_url: str,
    final_url: str,
    task_description: str,
    success_criteria: str,
    agent_message: str,
) -> Dict[str, Any]:
    """Verify success criteria using Google's Gemini vision API."""
    from google import genai
    from google.genai import types
    
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
    prompt = _build_verification_prompt(
        initial_url=initial_url,
        final_url=final_url,
        task_description=task_description,
        success_criteria=success_criteria,
        agent_message=agent_message,
    )
    
    initial_image_bytes = base64.b64decode(initial_screenshot_b64)
    final_image_bytes = base64.b64decode(final_screenshot_b64)
    
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[
            prompt,
            "INITIAL STATE (before agent started):",
            types.Part.from_bytes(data=initial_image_bytes, mime_type="image/png"),
            "FINAL STATE (after agent finished):",
            types.Part.from_bytes(data=final_image_bytes, mime_type="image/png"),
        ],
    )
    
    return _parse_verification_response(response.text)


def _build_verification_prompt(
    initial_url: str,
    final_url: str,
    task_description: str,
    success_criteria: str,
    agent_message: str,
) -> str:
    """Build the verification prompt for the LLM judge."""
    return f"""You are a strict evaluator for a web automation benchmark. Your job is to determine whether a task was completed successfully based on the success criteria.

## Task Description
{task_description}

## Success Criteria
{success_criteria}

## URL Information
- Initial URL: {initial_url}
- Final URL: {final_url}

## Agent's Self-Report
{agent_message or "No message provided"}

## Your Evaluation Instructions

You are provided with TWO screenshots:
1. INITIAL STATE - The page before the agent started working
2. FINAL STATE - The page after the agent finished

Compare these screenshots and evaluate whether the success criteria has been met.

IMPORTANT EVALUATION RULES:
- Be STRICT: Only mark as PASS if ALL parts of the success criteria are CLEARLY met
- For tasks involving "open then close" or similar state transitions: If initial and final states look similar, that may be CORRECT - verify from context and agent's report that the intermediate action occurred
- Do NOT give benefit of the doubt - if something is unclear or cannot be verified, mark as FAIL
- Consider URL changes as evidence of navigation
- Look for visible confirmation messages, form field values, UI state changes, etc.

## Response Format

You MUST respond in EXACTLY this format (no other text before or after):

STATUS: PASS
REASONING: <your one-paragraph explanation>

OR

STATUS: FAIL
REASONING: <your one-paragraph explanation>
"""


def _parse_verification_response(response_text: str) -> Dict[str, Any]:
    """Parse the verification response from the LLM."""
    lines = response_text.strip().split('\n')
    
    status = "FAIL"
    reasoning = response_text
    
    for i, line in enumerate(lines):
        line_upper = line.upper().strip()
        if line_upper.startswith("STATUS:"):
            status_value = line.split(":", 1)[1].strip().upper()
            if "PASS" in status_value:
                status = "PASS"
            else:
                status = "FAIL"
        elif line_upper.startswith("REASONING:"):
            reasoning = line.split(":", 1)[1].strip()
            if i + 1 < len(lines):
                remaining = '\n'.join(lines[i + 1:]).strip()
                if remaining:
                    reasoning = f"{reasoning} {remaining}"
            break
    
    return {
        "status": status,
        "reasoning": reasoning,
    }


# Default evaluator model for cross-model verification (avoids self-grading bias)
DEFAULT_EVALUATOR_MODEL = "gpt-4o"


async def verify_success_criteria(
    initial_screenshot_b64: str,
    final_screenshot_b64: str,
    initial_url: str,
    final_url: str,
    task_description: str,
    success_criteria: str,
    agent_message: str,
    evaluator_model: str = DEFAULT_EVALUATOR_MODEL,
) -> Dict[str, Any]:
    """
    Verify whether the success criteria was met using a vision LLM.
    
    Uses cross-model verification by default (GPT-4o) to avoid self-grading bias.
    
    Args:
        initial_screenshot_b64: Base64 encoded screenshot before task execution
        final_screenshot_b64: Base64 encoded screenshot after task execution
        initial_url: URL before task execution
        final_url: URL after task execution
        task_description: Description of the task
        success_criteria: Criteria for success
        agent_message: Message/report from the agent
        evaluator_model: Model to use for evaluation (default: gpt-4o for cross-model verification)
        
    Returns:
        Dict with 'status' (PASS/FAIL), 'reasoning', and 'evaluator_model'
    """
    provider = _get_provider_from_model(evaluator_model)
    
    try:
        if provider == "anthropic":
            return await _verify_with_anthropic(
                initial_screenshot_b64=initial_screenshot_b64,
                final_screenshot_b64=final_screenshot_b64,
                initial_url=initial_url,
                final_url=final_url,
                task_description=task_description,
                success_criteria=success_criteria,
                agent_message=agent_message,
            )
        elif provider == "google":
            return await _verify_with_google(
                initial_screenshot_b64=initial_screenshot_b64,
                final_screenshot_b64=final_screenshot_b64,
                initial_url=initial_url,
                final_url=final_url,
                task_description=task_description,
                success_criteria=success_criteria,
                agent_message=agent_message,
            )
        else:  # Default to OpenAI
            return await _verify_with_openai(
                initial_screenshot_b64=initial_screenshot_b64,
                final_screenshot_b64=final_screenshot_b64,
                initial_url=initial_url,
                final_url=final_url,
                task_description=task_description,
                success_criteria=success_criteria,
                agent_message=agent_message,
            )
    except Exception as e:
        return {
            "status": "ERROR",
            "reasoning": f"Verification failed with error: {str(e)}",
            "evaluator_model": evaluator_model,
        }


async def execute_task(
    stagehand,
    task: Dict[str, Any],
    model: str,
    evaluator_model: str = DEFAULT_EVALUATOR_MODEL,
) -> Dict[str, Any]:
    """
    Execute a single accessibility task using Stagehand agent.
    
    Args:
        stagehand: Initialized Stagehand instance
        task: Task dictionary
        model: Model name to use for the agent
        evaluator_model: Model to use for verification (default: gpt-4o for cross-model verification)
        
    Returns:
        Result dictionary with status and metrics
    """
    try:
        await stagehand.page.goto(task.get('url'))
        
        initial_url = stagehand.page._page.url
        initial_screenshot_bytes = await stagehand.page._page.screenshot(type="png")
        initial_screenshot_b64 = base64.b64encode(initial_screenshot_bytes).decode()
        
        task_instruction = f"Task: {task.get('task')}"
        
        # Let each client read its own API key from environment variables
        # Anthropic: ANTHROPIC_API_KEY, Google: GEMINI_API_KEY, OpenAI: OPENAI_API_KEY
        agent = stagehand.agent(
            model=model,
            instructions=task_instruction,
            options={}  # API keys are read from env vars by each client automatically
        )
        
        start_time = datetime.now()
        agent_result = await agent.execute(
            instruction=task.get('task'),
            max_steps=20,
            auto_screenshot=True
        )
        execution_duration = (datetime.now() - start_time).total_seconds()
        
        final_url = stagehand.page._page.url
        final_screenshot_bytes = await stagehand.page._page.screenshot(type="png")
        final_screenshot_b64 = base64.b64encode(final_screenshot_bytes).decode()
        
        if not agent_result.completed:
            return {
                'task_id': task.get('id'),
                'task': task.get('task'),
                'url': task.get('url'),
                'category': task.get('category'),
                'difficulty': task.get('difficulty'),
                'status': 'FAIL',
                'agent_model': model,
                'evaluator_model': evaluator_model,
                'agent_completed': False,
                'agent_message': agent_result.message,
                'verification_reasoning': 'Verification skipped: agent did not complete the task',
                'initial_url': initial_url,
                'final_url': final_url,
                'execution_duration_seconds': execution_duration,
                'verification_duration_seconds': 0,
                'total_duration_seconds': execution_duration,
                'timestamp': datetime.now().isoformat()
            }
        
        verification_start = datetime.now()
        verification_result = await verify_success_criteria(
            initial_screenshot_b64=initial_screenshot_b64,
            final_screenshot_b64=final_screenshot_b64,
            initial_url=initial_url,
            final_url=final_url,
            task_description=task.get('task'),
            success_criteria=task.get('success_criteria'),
            agent_message=agent_result.message,
            evaluator_model=evaluator_model,
        )
        verification_duration = (datetime.now() - verification_start).total_seconds()
        
        total_duration = execution_duration + verification_duration
        
        status = verification_result.get('status', 'FAIL')
        if status == "ERROR":
            status = "FAIL"  # Treat verification errors as failures
        
        return {
            'task_id': task.get('id'),
            'task': task.get('task'),
            'url': task.get('url'),
            'category': task.get('category'),
            'difficulty': task.get('difficulty'),
            'status': status,
            'agent_model': model,
            'evaluator_model': evaluator_model,
            'agent_completed': agent_result.completed,
            'agent_message': agent_result.message,
            'verification_reasoning': verification_result.get('reasoning'),
            'initial_url': initial_url,
            'final_url': final_url,
            'execution_duration_seconds': execution_duration,
            'verification_duration_seconds': verification_duration,
            'total_duration_seconds': total_duration,
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
