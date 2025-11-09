"""
Streamlit frontend for Deep Research Agent

This provides an interactive web interface for running research queries
with the multi-agent research system.
"""

import streamlit as st
import asyncio
import sys
import time
from pathlib import Path

# Add the project root to the path so we can import modules
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
deep_research_dir = project_root / "deep_research_from_scratch"
sys.path.insert(0, str(deep_research_dir))

# Import keys first to set up API keys
try:
    import keys
except ImportError:
    st.warning("keys.py not found. Make sure API keys are set via environment variables.")

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from deep_research_from_scratch.research_agent_full import deep_researcher_builder

# Import format_messages from the root utils.py
import importlib.util
utils_spec = importlib.util.spec_from_file_location("root_utils", project_root / "utils.py")
root_utils = importlib.util.module_from_spec(utils_spec)
utils_spec.loader.exec_module(root_utils)

# Page configuration
st.set_page_config(
    page_title="Deep Research Agent",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    .research-report {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "research_complete" not in st.session_state:
    st.session_state.research_complete = False
if "final_report" not in st.session_state:
    st.session_state.final_report = None
if "awaiting_clarification" not in st.session_state:
    st.session_state.awaiting_clarification = False
if "current_state" not in st.session_state:
    st.session_state.current_state = None
if "checkpointer" not in st.session_state:
    st.session_state.checkpointer = InMemorySaver()
if "full_agent" not in st.session_state:
    st.session_state.full_agent = deep_researcher_builder.compile(checkpointer=st.session_state.checkpointer)
if "pending_result" not in st.session_state:
    st.session_state.pending_result = None

async def run_research_async(current_state: dict, thread_id: str = "1", recursion_limit: int = 50):
    """Run the research agent asynchronously with current state."""
    thread = {
        "configurable": {
            "thread_id": thread_id,
            "recursion_limit": recursion_limit
        }
    }
    
    try:
        result = await st.session_state.full_agent.ainvoke(current_state, config=thread)
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }
    
    # Check if we have a final report
    if 'final_report' in result and result.get('final_report'):
        return {
            "status": "complete",
            "final_report": result['final_report'],
            "messages": result.get('messages', [])
        }
    
    # Check if agent is asking for clarification
    messages = result.get('messages', [])
    if messages:
        last_message = messages[-1]
        is_ai_message = isinstance(last_message, AIMessage)
        has_no_report = not result.get('final_report')
        
        if is_ai_message and has_no_report:
            content = last_message.content if hasattr(last_message, 'content') else str(last_message)
            is_question = (
                isinstance(content, str) and (
                    '?' in content or
                    any(word in content.lower() for word in ['please', 'clarify', 'specify', 'which', 'what', 'how', 'could you', 'would you'])
                )
            )
            
            if is_question:
                # Need clarification
                return {
                    "status": "clarification_needed",
                    "question": content,
                    "messages": messages,
                    "result": result
                }
    
    # If we get here, something unexpected happened
    return {
        "status": "incomplete",
        "messages": messages
    }

def run_research_sync(current_state: dict, thread_id: str = "1", recursion_limit: int = 50):
    """Wrapper to run async function in sync context."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(run_research_async(current_state, thread_id, recursion_limit))

# Main UI
st.markdown('<h1 class="main-header">🔍 Deep Research Agent</h1>', unsafe_allow_html=True)
st.markdown("""
<p style="text-align: center; color: #666; font-size: 1.1rem;">
A multi-agent research system that conducts comprehensive research and generates detailed reports.
</p>
""", unsafe_allow_html=True)

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    thread_id = st.text_input("Thread ID", value="1", help="Unique identifier for the conversation thread")
    recursion_limit = st.number_input("Recursion Limit", min_value=10, max_value=100, value=50, 
                                     help="Maximum number of graph iterations allowed")
    
    if st.button("🔄 Reset Session"):
        st.session_state.messages = []
        st.session_state.research_complete = False
        st.session_state.final_report = None
        st.session_state.awaiting_clarification = False
        st.session_state.current_state = None
        st.session_state.checkpointer = InMemorySaver()
        st.session_state.full_agent = deep_researcher_builder.compile(checkpointer=st.session_state.checkpointer)
        st.rerun()

# Main chat interface
if st.session_state.awaiting_clarification:
    st.info("⚠️ **Clarification Needed**")
    st.markdown("---")
    
    # Show the clarification question
    if st.session_state.messages:
        last_msg = st.session_state.messages[-1]
        if isinstance(last_msg, dict) and last_msg.get("role") == "assistant":
            st.markdown(f"**Question:** {last_msg['content']}")
    
    st.markdown("---")
    
    # Clarification input
    clarification = st.text_input(
        "Your Response:",
        key="clarification_input",
        placeholder="Provide your clarification here..."
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("✅ Submit Clarification", type="primary"):
            if clarification:
                # Add user clarification to messages
                st.session_state.messages.append({
                    "role": "user",
                    "content": clarification
                })
                
                # Update state and continue research
                if "pending_result" in st.session_state:
                    result = st.session_state.pending_result
                    all_messages = result['messages'] + [HumanMessage(content=clarification)]
                    current_state = {"messages": all_messages}
                    
                    st.session_state.awaiting_clarification = False
                    st.session_state.pending_result = None
                    
                    # Use a new thread_id to ensure fresh start after clarification
                    # Append timestamp to make it unique
                    new_thread_id = f"{thread_id}_clarified_{int(time.time())}"
                    
                    # Continue research
                    with st.spinner("🔍 Continuing research with your clarification..."):
                        new_result = run_research_sync(current_state, new_thread_id, recursion_limit)
                        
                        if new_result and new_result["status"] == "complete":
                            st.session_state.final_report = new_result["final_report"]
                            st.session_state.research_complete = True
                            # Add AI messages to chat
                            for msg in new_result.get("messages", []):
                                if isinstance(msg, AIMessage) and msg not in result['messages']:
                                    st.session_state.messages.append({
                                        "role": "assistant",
                                        "content": msg.content
                                    })
                        elif new_result and new_result["status"] == "clarification_needed":
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": new_result["question"]
                            })
                            st.session_state.pending_result = new_result
                            st.session_state.awaiting_clarification = True
                        elif new_result and new_result["status"] == "error":
                            error_msg = new_result.get('error', 'Unknown error occurred')
                            st.error(f"❌ **Error occurred:** {error_msg}")
                            st.warning("This error is preventing the research from continuing. Please check your API key permissions and quota.")
                            st.session_state.awaiting_clarification = False
                            # Add error message to chat
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": f"I encountered an error: {error_msg}. Please check your API key settings."
                            })
                        elif new_result:
                            # Handle incomplete status - show what we have
                            st.warning("Research process may not have completed fully. Check the conversation for details.")
                            # Add any new messages to the chat
                            for msg in new_result.get("messages", []):
                                if isinstance(msg, AIMessage) and msg not in result['messages']:
                                    msg_content = msg.content if hasattr(msg, 'content') else str(msg)
                                    if not any(m.get("content") == msg_content for m in st.session_state.messages):
                                        st.session_state.messages.append({
                                            "role": "assistant",
                                            "content": msg_content
                                        })
                    
                    st.rerun()
    
    with col2:
        if st.button("❌ Cancel"):
            st.session_state.awaiting_clarification = False
            st.session_state.research_complete = False
            st.session_state.pending_result = None
            st.rerun()

else:
    # Research query input
    query = st.chat_input("Enter your research query here...")
    
    if query:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Run research
        with st.spinner("🔍 Conducting research... This may take several minutes."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Initializing research agent...")
            progress_bar.progress(10)
            
            status_text.text("Analyzing query and generating research plan...")
            progress_bar.progress(30)
            
            current_state = {"messages": [HumanMessage(content=query)]}
            result = run_research_sync(current_state, thread_id, recursion_limit)
            progress_bar.progress(60)
            
            if result:
                if result["status"] == "complete":
                    progress_bar.progress(100)
                    status_text.text("Research complete!")
                    
                    # Add AI messages to chat (including verification messages)
                    for msg in result.get("messages", []):
                        if isinstance(msg, AIMessage):
                            # Check if this message is already in the chat
                            msg_content = msg.content if hasattr(msg, 'content') else str(msg)
                            if not any(m.get("content") == msg_content for m in st.session_state.messages):
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": msg_content
                                })
                    
                    st.session_state.final_report = result["final_report"]
                    st.session_state.research_complete = True
                    st.rerun()
                
                elif result["status"] == "clarification_needed":
                    progress_bar.progress(50)
                    status_text.text("⚠️ Clarification needed")
                    
                    # Add AI clarification question to messages
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["question"]
                    })
                    
                    st.session_state.awaiting_clarification = True
                    st.session_state.pending_result = result
                    st.rerun()
                
                elif result["status"] == "error":
                    progress_bar.progress(0)
                    status_text.text("❌ Error occurred")
                    st.error(f"Error: {result.get('error', 'Unknown error')}")
                
                else:
                    status_text.text("⚠️ Research incomplete")
                    st.warning("Research process did not complete. Please try again.")
            
            progress_bar.empty()
            status_text.empty()

# Display chat messages
if st.session_state.messages:
    st.markdown("---")
    st.subheader("💬 Conversation")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Display final report
if st.session_state.research_complete and st.session_state.final_report:
    st.markdown("---")
    st.markdown('<div class="research-report">', unsafe_allow_html=True)
    st.subheader("📄 Final Research Report")
    st.markdown(st.session_state.final_report)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Download button
    st.download_button(
        label="📥 Download Report",
        data=st.session_state.final_report,
        file_name="research_report.md",
        mime="text/markdown"
    )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>Deep Research Agent - Multi-Agent Research System</p>
</div>
""", unsafe_allow_html=True)

