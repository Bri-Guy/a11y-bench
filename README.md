# A11y-Bench: Accessibility Benchmark for Computer Use Agents

> By Madi, Brian, Jason

A benchmark for evaluating AI computer use agents on accessibility-related tasks.

**Credits:**
- [Browserbase](https://browserbase.com) for the Stagehand agent framework
- [WebVoyager](https://github.com/MinorJerry/WebVoyager) team for the benchmark structure inspiration

## Quick Start

1. **Install dependencies:**
```bash
uv venv .venv
uv sync
source .venv/bin/activate
uv pip install stagehand

# Optional: Install visualization tools
uv sync --extra viz
```

2. **Set up environment variables** (copy `.env.example` to `.env`):
```bash
MODEL_API_KEY=your_gemini_api_key
GEMINI_API_KEY=your_gemini_api_key
```

3. **Run a single test:**
```bash
python run_single_test.py a11y_001
```

4. **Run the full benchmark:**
```bash
python run_benchmark.py
```

## Benchmark Overview

The benchmark includes accessibility audit tasks across different categories:

- TODO

### Example Task

```json
{
  "id": "a11y_001",
  "task": "Navigate to Wikipedia homepage and verify all images have alt text",
  "url": "https://www.wikipedia.org",
  "category": "image_accessibility",
  "difficulty": "easy",
  "success_criteria": "All images on the homepage have non-empty alt attributes"
}
```

## Usage

### Run Single Task

```bash
# Basic usage
python run_single_test.py a11y_001

# With options
python run_single_test.py a11y_001 --env BROWSERBASE --model gpt-4o
```

### Run Full Benchmark

```bash
# Run all tasks
python run_benchmark.py

# Filter by category
python run_benchmark.py --category keyboard_navigation

# Filter by difficulty
python run_benchmark.py --difficulty easy

# Run specific tasks
python run_benchmark.py --task-ids a11y_001 a11y_002
```

## Project Structure

```
a11y-bench/
├── data/
│   └── accessibility_tasks.jsonl    # Benchmark tasks
├── evaluation/
│   ├── evaluate_results.py          # Evaluation script
│   ├── compare_runs.py              # Compare multiple runs
│   └── visualize_results.py         # Generate performance graphs
├── results/                          # Results directory
│   ├── examples/                     # Example results
│   └── visualizations/               # Generated graphs
├── utils.py                          # Utility functions
├── run_single_test.py               # Single task runner
└── run_benchmark.py                 # Full benchmark runner
```

## Adding New Tasks

Add entries to `data/accessibility_tasks.jsonl`:

```json
{"id": "a11y_006", "task": "Your task description", "url": "https://example.com", "category": "category_name", "difficulty": "easy", "success_criteria": "Success criteria"}
```

## Evaluation

### Evaluate Results

Evaluate benchmark results:

```bash
python evaluation/evaluate_results.py --results-dir results
```

Compare multiple runs:

```bash
python evaluation/compare_runs.py results/summary1.json results/summary2.json
```

### Generate Visualizations

Create publication-ready graphs comparing model performance:

**1. Install visualization dependencies:**

```bash
uv sync --extra viz
```

**2. Generate graphs:**

```bash
# Compare 2 or more model providers
python evaluation/visualize_results.py \
  results/openai_corrected.json \
  results/gemini_corrected.json \
  results/anthropic_corrected.json \
  -o results/visualizations
```

**Generated outputs:**
- `overall_performance.png` - Bar chart comparing success rates across providers
- `category_performance.png` - Performance breakdown by task category (motor, visual, etc.)
- `difficulty_performance.png` - Performance breakdown by difficulty level (Easy, Moderate, Complex)
- `median_duration.png` - Median time per successful task across providers
- `summary_report.txt` - Detailed text summary of all statistics

All graphs are high-resolution (300 DPI) and ready for research papers or presentations.

## Stagehand API Reference

The benchmark uses Stagehand's agent API. Key methods:

- **`page.goto(url)`** — Navigate to a URL
- **`page.act(instruction)`** — Perform actions (click, type, etc.)
- **`page.extract(instruction, schema)`** — Extract data from page
- **`page.observe(instruction)`** — Get element information
- **`agent(model, instructions)`** — Create an autonomous agent

For full documentation, see [docs.stagehand.dev](https://docs.stagehand.dev/).

## Examples

Check out `examples/agent_example_local.py` for a complete Stagehand example.

## Contributing

To add new accessibility tasks:

1. Add task to `data/accessibility_tasks.jsonl`
2. Test with `python run_single_test.py <task_id>`
3. Ensure all required fields are present (id, task, url, category, difficulty, success_criteria)

## License

MIT License (c) 2025 Browserbase, Inc.
