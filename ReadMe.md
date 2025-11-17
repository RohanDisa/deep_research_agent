# Deep Research Agent

Deep Research Agent is an end-to-end, production-style research assistant built entirely with LangGraph. It converts open-ended questions into scoped research briefs, orchestrates parallel sub-agents to gather evidence, and delivers publication-quality reports complete with citations.

---

## Highlights

- **Scoping Before Search** – Structured clarification step that decides when to question the user and produces an actionable brief.
- **Supervisor + Specialists** – A LangGraph supervisor coordinates dedicated researcher agents that can run in parallel with independent context windows.
- **Tool-Driven Evidence Gathering** – Research agents rely on Tavily web search, automatic summarisation, and reflective thinking loops (`think_tool`) to avoid hallucination.
- **Deterministic Outputs** – Pydantic schemas enforce consistent state transitions; reports are generated from cleaned research notes and reference every source.
- **Multiple Entry Points** – Run from the CLI, Streamlit UI, or Jupyter notebooks. Rich terminal output keeps the CLI readable.

---

## System Architecture

```
User Query
   │
   ▼
Scoping Graph (`clarify_with_user` → `write_research_brief`)
   │ produces research brief + supervisor kickoff message
   ▼
Supervisor Graph (`supervisor` ⇄ `supervisor_tools`)
   │ delegates ConductResearch tasks, records notes
   ▼
Research Agents (`llm_call` ⇄ `tool_node` → `compress_research`)
   │ each agent performs Tavily searches + summarises raw notes
   ▼
Final Report Generation (`final_report_generation`)
   │ synthesises brief + aggregated notes into markdown report
   ▼
Formatted Output + Sources
```
---

## Repository Layout

```text
deep_research_from_scratch/
├─ research_agent_full.py        # Top-level graph builder
├─ research_agent_scope.py       # Clarification + brief generation graph
├─ multi_agent_supervisor.py     # Supervisor that launches researcher agents
├─ research_agent.py             # Worker agent (search + summarise loop)
├─ utils.py                      # Tavily integration, summarisation helpers, tools
├─ prompts.py                    # Prompt templates for every stage
├─ state_*.py                    # Pydantic schemas + LangGraph state definitions
run_research.py                  # CLI entry point
streamlit_app.py                 # Streamlit UI with session persistence
full_agent.ipynb / research_agent.ipynb / research_supervisor.ipynb / scoping.ipynb
example_query.txt                # Sample deep-research prompt
requirements.txt                 # Python dependencies
```

---

## Prerequisites

- Python 3.11 or newer is recommended (LangGraph currently targets modern CPython releases).
- API access for:
  - OpenAI (GPT-4.1 / GPT-4.1-mini are the default chat models)
  - Tavily Search
  - LangSmith (optional, for tracing)

### Installing Dependencies

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows
# source .venv/bin/activate      # macOS / Linux
pip install -U pip
pip install -r requirements.txt
```

---

## Configuring Secrets

All runtime components read API keys from the environment. You can either export them in your shell or create a `keys.py` module that sets `os.environ[...]` before the agents initialise.

```python
# keys.py (example stub – do not commit real keys)
import os

os.environ["OPENAI_API_KEY"] = "sk-...."
os.environ["TAVILY_API_KEY"] = "tvly-...."
os.environ["LANGSMITH_API_KEY"] = "lsv2_...."  # optional
```

---

## Running the Agent

### Command Line

```bash
python run_research.py "Compare Gemini to OpenAI Deep Research agents."
python run_research.py "What are the latest developments in AI?" research_1
python run_research.py "Analyse the impact of LLMs on software development" research_2 75
```

Arguments:
1. `query` *(required)* – research question.
2. `thread_id` *(optional, default `"1"`)* – persisted LangGraph thread identifier.
3. `recursion_limit` *(optional, default `50`)* – hard cap on graph iterations.

The CLI prints workflow progress, conversation history, and the final markdown report (rendered with Rich when available).

### Streamlit UI

```bash
streamlit run streamlit_app.py
```

Features:
- Chat-style input with live clarification prompts.
- Session persistence via LangGraph checkpoints.
- Downloadable markdown report once research completes.

---

## Customising the System

- **Model selection** – all models are initialised via `langchain.chat_models.init_chat_model`. Swap providers or models by changing constructor parameters or by reading from environment variables.
- **Prompts** – tweak high-level behaviour in `deep_research_from_scratch/prompts.py`. The prompts already enforce explicit citation handling and structured outputs.
- **Tooling** – add new LangChain tools in `deep_research_from_scratch/utils.py` and register them with `tools` in `research_agent.py`. Remember to update `tools_by_name`.
- **Supervisor policy** – adjust maximum iterations or parallelism in `multi_agent_supervisor.py` (`max_researcher_iterations`, `max_concurrent_researchers`).
- **Output formatting** – `final_report_generation_prompt` governs the final markdown structure. Modify it to emit JSON, tables, or other formats.

---

Happy researching! 
