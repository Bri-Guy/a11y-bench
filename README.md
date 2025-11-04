# Agent-A11y-Bench: Accessibility Benchmark for Computer Use Agents
> Madi, Brian, Jason
> Credits to Browserbase for the Stagehand agent framewor. Credits to Webvoyager team for scaffolding around the underlying benchmark

### TL;DR on Stagehand API:

- **act** — Instruct the AI to perform actions (e.g. click a button or scroll).
```python
await stagehand.page.act("click on the 'Quickstart' button")
```
- **extract** — Extract and validate data from a page using a Pydantic schema.
```python
await stagehand.page.extract("the summary of the first paragraph")
```
- **observe** — Get natural language interpretations to, for example, identify selectors or elements from the page.
```python
await stagehand.page.observe("find the search bar")
```
- **agent** — Execute autonomous multi-step tasks with provider-specific agents (OpenAI, Anthropic, etc.).
```python
await stagehand.agent.execute("book a reservation for 2 people for a trip to the Maldives")
```


## Installation:

To get started, simply:

```bash
pip install stagehand
```

> We recommend using [uv](https://docs.astral.sh/uv/) for your package/project manager. If you're using uv can follow these steps:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install stagehand
```

## Quickstart

Check out ```examples/agent_example_local.py```

## Documentation

See our full documentation [here](https://docs.stagehand.dev/).

## Cache Actions

You can cache actions in Stagehand to avoid redundant LLM calls. This is particularly useful for actions that are expensive to run or when the underlying DOM structure is not expected to change.

### Using `observe` to preview an action

`observe` lets you preview an action before taking it. If you are satisfied with the action preview, you can run it in `page.act` with no further LLM calls.

```python
# Get the action preview
action_preview = await page.observe("Click the quickstart link")

# action_preview is a JSON-ified version of a Playwright action:
# {
#     "description": "The quickstart link",
#     "method": "click",
#     "selector": "/html/body/div[1]/div[1]/a",
#     "arguments": []
# }

# NO LLM INFERENCE when calling act on the preview
await page.act(action_preview[0])
```

If the website happens to change, `self_heal` will run the loop again to save you from constantly updating your scripts.


## License

MIT License (c) 2025 Browserbase, Inc.
