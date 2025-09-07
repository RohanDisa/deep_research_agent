# Deep Research Agent From Scratch

A comprehensive multi-agent research system built with LangGraph that can conduct intelligent research on any topic and generate detailed reports.

## What It Does

This system transforms vague user queries into comprehensive research reports through a three-phase process:
- **Scope** → Clarify research requirements and generate structured briefs
- **Research** → Multi-agent parallel research with external tools
- **Write** → Synthesize findings into professional reports

## Technical Skills Demonstrated

### Agent Architecture & Patterns
- **Multi-Agent Coordination**: Supervisor pattern with parallel worker agents
- **ReAct Agent Loops**: Iterative research with tool calling and reflection
- **State Management**: Complex state flows across subgraphs and nodes

### Advanced Integrations
- **External APIs**: Tavily search integration with content summarization
- **LangGraph Workflows**: End-to-end orchestration with conditional routing

### Production-Ready Features
- **Structured Output**: Pydantic schemas for reliable AI decision making
- **Async Orchestration**: Strategic parallel processing for concurrent research
- **Tool Ecosystem**: Custom tools, MCP servers, and search optimization
- **Error Handling**: Robust workflow design with proper state transitions

## Implementation Highlights

1. **Smart Scoping** - Clarifies ambiguous queries and generates detailed research briefs
2. **Parallel Research** - Multiple agents research different aspects simultaneously
3. **Multi-Agent Supervision** - Coordinator agent managing specialized research workers
4. **Report Synthesis** - Intelligent aggregation of research findings into cohesive reports

## Key Technologies

- **LangGraph** for agent workflows and state management
- **LangChain** for LLM integration and tool orchestration  
- **Pydantic** for structured output and data validation
- **Async/Await** for concurrent processing

Built a complete production-ready research system that showcases advanced agent patterns, multi-modal integrations, and sophisticated workflow orchestration.

## Required API Keys
- OPENAI_API_KEY=your_openai_api_key_here
- TAVILY_API_KEY=your_tavily_api_key_here
- LANGSMITH_API_KEY=your_langsmith_api_key_here

## How to Run

First install all the requirements from requirments.txt

Then, simply open and execute the `full_agent.ipynb` notebook:

```bash
jupyter notebook full_agent.ipynb