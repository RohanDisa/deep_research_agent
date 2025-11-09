"""
Main entry point for running the Deep Research Agent.

This script provides a command-line interface to run the research agent
with a user-specified query.
"""

import asyncio
import sys
import io
from pathlib import Path

# Fix Windows console encoding issues with emojis
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add the project root and deep_research_from_scratch to the path so we can import modules
project_root = Path(__file__).parent
deep_research_dir = project_root / "deep_research_from_scratch"
# Add project_root first, then deep_research_dir so package imports work
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(deep_research_dir))

# Import keys first to set up API keys
try:
    import keys
except ImportError:
    print("Warning: keys.py not found. Make sure API keys are set via environment variables.")

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from deep_research_from_scratch.research_agent_full import agent, deep_researcher_builder

# Import format_messages from the root utils.py (not deep_research_from_scratch/utils.py)
# We import it directly from the file to avoid conflicts with deep_research_from_scratch/utils.py
import importlib.util
utils_spec = importlib.util.spec_from_file_location("root_utils", project_root / "utils.py")
root_utils = importlib.util.module_from_spec(utils_spec)
utils_spec.loader.exec_module(root_utils)
format_messages = root_utils.format_messages

try:
    from rich.markdown import Markdown
    from rich.console import Console
    console = Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    print("Warning: rich library not found. Install it for better output formatting: pip install rich")


async def run_research(query: str, thread_id: str = "1", recursion_limit: int = 50):
    """
    Run the research agent with a given query, handling interactive clarification.
    
    Args:
        query: The research query/question to investigate
        thread_id: Unique identifier for the conversation thread
        recursion_limit: Maximum number of iterations allowed
    """
    # Set up checkpointer for conversation state
    checkpointer = InMemorySaver()
    full_agent = deep_researcher_builder.compile(checkpointer=checkpointer)
    
    # Configure thread
    thread = {
        "configurable": {
            "thread_id": thread_id,
            "recursion_limit": recursion_limit
        }
    }
    
    print(f"\n[STARTING] Research query: {query}\n")
    print("-" * 80)
    
    # Initial query
    current_state = {"messages": [HumanMessage(content=query)]}
    max_clarification_rounds = 5  # Prevent infinite clarification loops
    clarification_round = 0
    
    # Interactive loop: handle clarification if needed
    while clarification_round < max_clarification_rounds:
        print(f"\n[WORKFLOW] Starting iteration {clarification_round + 1}")
        print(f"[WORKFLOW] Current messages count: {len(current_state.get('messages', []))}")
        
        try:
            # Run the agent with current state
            print("[WORKFLOW] Invoking agent workflow...")
            result = await full_agent.ainvoke(current_state, config=thread)
            print("[WORKFLOW] Agent workflow completed")
        except Exception as e:
            print(f"\n[ERROR] Error during agent execution: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Display conversation so far
        print("\n[CONVERSATION]\n")
        format_messages(result['messages'])
        
        # Check if we have a final report (research completed)
        if 'final_report' in result and result.get('final_report'):
            print("\n" + "=" * 80)
            print("[FINAL RESEARCH REPORT]")
            print("=" * 80 + "\n")
            
            if HAS_RICH:
                console.print(Markdown(result['final_report']))
            else:
                print(result['final_report'])
            
            return result
        
        # Check if agent is asking for clarification
        # If no final_report and last message is from AI, likely asking for clarification
        messages = result.get('messages', [])
        if messages:
            last_message = messages[-1]
            
            # Check if last message is from AI (AIMessage) and no final report
            # This indicates the agent ended with a clarification question
            is_ai_message = isinstance(last_message, AIMessage)
            has_no_report = not result.get('final_report')
            
            if is_ai_message and has_no_report:
                # Check if this looks like a clarification question
                content = last_message.content if hasattr(last_message, 'content') else str(last_message)
                is_question = (
                    isinstance(content, str) and (
                        '?' in content or
                        any(word in content.lower() for word in ['please', 'clarify', 'specify', 'which', 'what', 'how', 'could you', 'would you'])
                    )
                )
                
                if is_question:
                    # Agent is asking for clarification
                    print("\n" + "-" * 80)
                    print("[CLARIFICATION NEEDED]")
                    print("-" * 80)
                    print("\nPlease provide your response to continue research:\n")
                    
                    # Get user input
                    user_response = input("Your response: ").strip()
                    
                    if not user_response:
                        print("[WARNING] Empty response. Continuing with original query...")
                        user_response = query
                    
                    # Add user response and continue with updated state
                    # When workflow ends at END, we need to restart it from the beginning
                    # Pass all messages including the new clarification to restart the workflow
                    all_messages = result['messages'] + [HumanMessage(content=user_response)]
                    current_state = {
                        "messages": all_messages
                    }
                    clarification_round += 1
                    print(f"\n[INFO] Continuing research with your clarification (round {clarification_round + 1})...")
                    print(f"[WORKFLOW] Total messages in conversation: {len(all_messages)}")
                    print("[WORKFLOW] Restarting workflow from START with updated conversation...\n")
                    continue
        
        # If we get here, agent should have completed but didn't return final_report
        # This might be an error state - break and show what we have
        print("\n[INFO] Research process completed. Showing results...\n")
        break
    
    if clarification_round >= max_clarification_rounds:
        print("\n[WARNING] Maximum clarification rounds reached. Showing current results...\n")
    
    # Display final conversation
    print("\n[FINAL CONVERSATION]\n")
    format_messages(result['messages'])
    
    return result


def main():
    """Main function to handle command-line execution."""
    if len(sys.argv) < 2:
        print("Usage: python run_research.py <research_query> [thread_id] [recursion_limit]")
        print("\nExample:")
        print('  python run_research.py "Compare Gemini to OpenAI Deep Research agents."')
        print('  python run_research.py "What are the latest developments in AI?" research_1 100')
        sys.exit(1)
    
    query = sys.argv[1]
    thread_id = sys.argv[2] if len(sys.argv) > 2 else "1"
    recursion_limit = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    
    print(f"[DEBUG] Query: {query}")
    print(f"[DEBUG] Thread ID: {thread_id}")
    print(f"[DEBUG] Recursion Limit: {recursion_limit}")
    
    # Run the async function
    try:
        result = asyncio.run(run_research(query, thread_id, recursion_limit))
    except KeyboardInterrupt:
        print("\n\n[WARNING] Research interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Error running research agent: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

