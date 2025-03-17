import streamlit as st
import asyncio
import os
import time
from typing import List, Optional, Tuple, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

# Try importing the agents SDK components
try:
    from agents import Agent, Runner, WebSearchTool, trace
    from openai import OpenAI, APIError, AuthenticationError
    AGENTS_SDK_AVAILABLE = True
except ImportError:
    AGENTS_SDK_AVAILABLE = False
    st.error("Agents SDK not found. Some functionality will be simulated.")

# Define ReportData as a global model - simplified for robustness
class ReportData(BaseModel):
    short_summary: str = Field(..., description="A concise summary of the research findings (1-2 paragraphs)")
    markdown_report: str = Field(..., description="Brief markdown-formatted report (500-800 words)")
    follow_up_questions: List[str] = Field(default_factory=list, description="2-3 suggested follow-up questions")
    
# Planner Agent
class WebSearchItem(BaseModel):
    reason: str
    query: str

class WebSearchPlan(BaseModel):
    searches: List[WebSearchItem]

# Handoff tracking
class HandoffEvent(BaseModel):
    from_agent: str
    to_agent: str
    reason: str
    input: str
    output: str
    timestamp: str
    
    @classmethod
    def create(cls, from_agent, to_agent, reason, input_data, output_data):
        # Truncate long inputs/outputs for display
        input_str = str(input_data)
        output_str = str(output_data)
        
        if len(input_str) > 200:
            input_str = input_str[:197] + "..."
        if len(output_str) > 200:
            output_str = output_str[:197] + "..."
            
        return cls(
            from_agent=from_agent,
            to_agent=to_agent,
            reason=reason,
            input=input_str,
            output=output_str,
            timestamp=datetime.now().strftime("%H:%M:%S")
        )

# Function to validate API key
def validate_openai_key(api_key: str) -> Tuple[bool, Optional[str]]:
    """
    Validate the OpenAI API key
    
    Returns:
    - Tuple of (is_valid, error_message)
    - is_valid: Boolean indicating if the key is valid
    - error_message: Detailed error message if validation fails
    """
    if not api_key or api_key.strip() == "":
        return False, "API Key cannot be empty."
    
    try:
        if AGENTS_SDK_AVAILABLE:
            client = OpenAI(api_key=api_key)
            # Light validation check - just checking if the key format is valid
            # Don't actually call the API to avoid unnecessary costs
            if api_key.startswith("sk-") and len(api_key) > 20:
                return True, None
            else:
                return False, "Invalid API key format. OpenAI API keys typically start with 'sk-'"
        else:
            # When SDK not available, just do format validation
            if api_key.startswith("sk-") and len(api_key) > 20:
                return True, None
            else:
                return False, "Invalid API key format. OpenAI API keys typically start with 'sk-'"
    except AuthenticationError:
        return False, (
            "Authentication failed. Possible reasons:\n"
            "- Incorrect API key\n"
            "- Key has been revoked\n"
            "- Billing issues with your OpenAI account"
        )
    except APIError as e:
        return False, (
            f"API Error: {str(e)}\n"
            "Possible reasons:\n"
            "- Network connectivity issue\n"
            "- Temporary OpenAI service disruption\n"
            "- Invalid API key format"
        )
    except Exception as e:
        return False, (
            f"Unexpected error: {str(e)}\n"
            "Please check your API key and network connection."
        )
def setup_agents(api_key: str, num_searches: int, writer_model: str):
    """Set up agents with the provided API key and configuration"""
    if not AGENTS_SDK_AVAILABLE:
        # Return mock agents when SDK isn't available (unchanged)
        class MockAgent:
            def __init__(self, name, instructions="", model=""):
                self.name = name
                self.instructions = instructions
                self.model = model
        
        # Mock agent setup (unchanged)
        # ...
        
        return runner_agent
    
    # Real implementation with Agents SDK
    # Configure the OpenAI client globally
    os.environ['OPENAI_API_KEY'] = api_key

    # Writer Agent - Simple prompt with clear instructions
    WRITER_PROMPT = (
        "You are a research writer creating a report from search results. "
        "You will receive a query and search results from the SearchAgent. "
        "Your task is to compile the findings into a comprehensive report with: "
        "1. A concise summary (1-2 paragraphs) "
        "2. A markdown-formatted report (500-800 words) "
        "3. 2-3 follow-up questions\n\n"
        "Do NOT hand off to any other agent. You are the final step in the research process."
    )

    # Create the writer agent with standard settings - no handoffs needed as it's the final agent
    writer_agent = Agent(
        name="WriterAgent",
        instructions=WRITER_PROMPT,
        model=writer_model,
        output_type=ReportData,
    )

    # Search Agent - Simplified and more focused instructions with explicit handoff
    SEARCH_INSTRUCTIONS = (
        "You are a research assistant performing web searches. "
        "You will receive search queries from the PlannerAgent. "
        "For each search term:\n"
        "1. Extract key facts and insights relevant to the query\n"
        "2. Focus on objective information from reliable sources\n"
        "3. Create a brief summary (150-200 words maximum)\n"
        "4. Format information as bullet points whenever possible\n"
        "5. Prioritize recent and factual information\n\n"
        "After completing ALL searches, you MUST hand off to the WriterAgent by using the transfer_to_writeragent function. "
        "Do NOT hand off after each individual search - only after all searches are complete."
    )

    # Create the search agent with handoff to writer
    search_agent = Agent(
        name="SearchAgent",
        instructions=SEARCH_INSTRUCTIONS,
        tools=[WebSearchTool(user_location={"type": "approximate", "city": "New York"})],
        handoffs=[writer_agent],  # Search hands off to Writer
    )

    # Updated Planner Prompt with explicit handoff instructions
    PLANNER_PROMPT = (
        "You are a helpful research assistant. Given a query from the RunnerAgent, "
        f"come up with a set of web searches to perform to best answer the query. "
        f"Output between 1-{num_searches} search terms to query for.\n\n"
        "IMPORTANT: You MUST output your search plan in the format required by WebSearchPlan, with a 'searches' array "
        "containing objects with 'reason' and 'query' fields.\n\n"
        "AFTER creating your search plan, you MUST ALWAYS hand off to the SearchAgent by using the transfer_to_searchagent "
        "function call. Do NOT try to conduct the searches yourself.\n\n"
        "Workflow steps:\n"
        "1. Create your search plan\n"
        "2. Output the search plan in the required format\n"
        "3. Call the transfer_to_searchagent function to hand off"
    )

    # Create the planner agent with handoff to search
    planner_agent = Agent(
        name="PlannerAgent",
        instructions=PLANNER_PROMPT,
        model="gpt-4o",
        output_type=WebSearchPlan,
        handoffs=[search_agent],  # Planner hands off to Search
    )
    
    # Runner Agent with explicit sequential instructions
    RUNNER_PROMPT = (
        "You are a research workflow coordinator. Your job is to start a sequential research process. "
        "For any research query, IMMEDIATELY hand off to the PlannerAgent by using the transfer_to_planneragent function. "
        "The research process will then follow this exact sequence:\n"
        "1. PlannerAgent will create search queries and hand off to SearchAgent\n"
        "2. SearchAgent will execute searches and hand off to WriterAgent\n"
        "3. WriterAgent will compile the findings into a final report\n\n"
        "Your only task is to start this process by handing off to the PlannerAgent."
    )
    
    # Create the runner agent with initial handoff only to the planner
    runner_agent = Agent(
        name="RunnerAgent",
        instructions=RUNNER_PROMPT,
        model="gpt-4o",
        handoffs=[planner_agent],  # Runner hands off to Planner
    )

    return runner_agent
async def perform_research(query: str, runner_agent, session_state):
    """Async function to perform the research workflow using a runner agent"""
    handoff_details = []
    search_details = []
    
    # Initialize the progress container
    progress_container = st.empty()
    progress_bar = progress_container.progress(0)
    status_text = st.empty()
    error_container = st.empty()
    
    if not AGENTS_SDK_AVAILABLE:
        # Simulate the research process
        status_text.text("Runner agent coordinating research workflow...")
        await asyncio.sleep(1.5)
        
        # First handoff to planner agent
        status_text.text("Handing off to planning agent...")
        progress_bar.progress(0.2)
        await asyncio.sleep(1.5)
        
        # Mock search plan
        search_plan = WebSearchPlan(searches=[
            WebSearchItem(reason="To get latest information", query=f"latest developments in {query}"),
            WebSearchItem(reason="To get historical context", query=f"history of {query}"),
            WebSearchItem(reason="To get expert opinions", query=f"expert analysis {query}"),
        ])
        
        # Record handoff details
        handoff_details.append(HandoffEvent.create(
            from_agent="RunnerAgent",
            to_agent="PlannerAgent",
            reason="Planning search strategy",
            input_data=query,
            output_data=f"Created search plan with {len(search_plan.searches)} queries"
        ))
        
        # Second handoff to search agent
        status_text.text("Handing off to search agent...")
        progress_bar.progress(0.4)
        await asyncio.sleep(1.5)
        
        # Mock search results
        for i, item in enumerate(search_plan.searches):
            status_text.text(f"Searching: {item.query}")
            await asyncio.sleep(1)
            
            # Mock search result
            search_result = f"Found information about {item.query}. This includes various sources and citations relevant to the query."
            
            # Collect detailed run information
            search_details.append({
                "query": item.query,
                "reason": item.reason,
                "result": search_result,
            })
            
            # Update progress
            progress = 0.4 + (0.3 * (i+1) / len(search_plan.searches))
            progress_bar.progress(progress)
        
        # Record combined handoff details for search
        handoff_details.append(HandoffEvent.create(
            from_agent="PlannerAgent",
            to_agent="SearchAgent",
            reason="Execute web searches",
            input_data="Search plan with queries",
            output_data=f"Completed {len(search_plan.searches)} searches"
        ))
        
        # Third handoff to writer agent
        status_text.text("Handing off to writer agent...")
        progress_bar.progress(0.8)
        await asyncio.sleep(2)
        
        # Mock report
        report = ReportData(
            short_summary=f"This is a summary of research findings about {query}.",
            markdown_report=f"""
# Comprehensive Research: {query}

## Introduction
This research explores {query} in depth, analyzing various aspects and perspectives.

## Key Findings
1. The first major finding about {query}
2. The second major finding about {query}
3. The third major finding about {query}

## Historical Context
{query} has a rich history dating back to its origins.

## Expert Analysis
Experts in the field suggest that {query} will continue to evolve.

## Conclusion
In conclusion, {query} represents an important area for further study.
            """,
            follow_up_questions=[
                f"What are the future trends for {query}?",
                f"How does {query} compare to similar topics?",
                f"What are the practical applications of {query}?"
            ]
        )
        
        # Record handoff details
        handoff_details.append(HandoffEvent.create(
            from_agent="SearchAgent",
            to_agent="WriterAgent",
            reason="Compile research report",
            input_data="Collected search results",
            output_data="Creating comprehensive report"
        ))
        
        # Update progress
        progress_bar.progress(1.0)
        status_text.text("Research complete!")
        
    else:
        # Use actual Agents SDK with the runner agent
        try:
            # Start the research process
            status_text.text("Starting research workflow with runner agent...")
            progress_bar.progress(0.1)
            
            # Add first handoff for UI display (User to Runner)
            handoff_details.append(HandoffEvent.create(
                from_agent="User",
                to_agent="RunnerAgent",
                reason="Initial query submission",
                input_data=query,
                output_data="Initiating research workflow"
            ))
            
            # Update UI for first handoff
            status_text.text("Runner agent analyzing query...")
            progress_bar.progress(0.2)
            
            # Run the entire agent chain
            formatted_query = f"Research Query: {query}"
            with trace("Research Workflow"):
                runner_result = await Runner.run(
                    runner_agent,
                    formatted_query
                )
            
            # Process all items to track handoffs and search operations
            current_agent = "RunnerAgent"
            current_search_query = None
            search_queries = []
            search_plans = []
            
            # First pass: extract search queries from planner output
            for item in runner_result.new_items:
                # Look for search plans in message output
                if item.type == "message_output_item":
                    output_text = None
                    # Try different ways to get output text based on SDK version
                    if hasattr(item, 'output_text'):
                        output_text = item.output_text
                    elif hasattr(item, 'raw_item') and hasattr(item.raw_item, 'content'):
                        output_text = item.raw_item.content
                        
                    if output_text and isinstance(output_text, str):
                        # Look for JSON content with searches
                        if '"searches"' in output_text or "'searches'" in output_text:
                            # Clean up JSON content
                            content = output_text.strip()
                            if "```json" in content:
                                content = content.split("```json")[1].split("```")[0]
                            elif "```" in content:
                                content = content.split("```")[1].split("```")[0]
                            try:
                                import json
                                data = json.loads(content)
                                if "searches" in data:
                                    search_plans.append(data)
                                    for search in data["searches"]:
                                        if "query" in search and "reason" in search:
                                            search_queries.append({
                                                "query": search["query"],
                                                "reason": search.get("reason", "Search query")
                                            })
                            except json.JSONDecodeError:
                                pass
            
            # Second pass: process handoffs and tool calls
            for idx, item in enumerate(runner_result.new_items):
                # Process handoffs
                if item.type == "handoff_call_item":
                    # Extract target agent name from handoff call
                    target_agent = None
                    try:
                        # Try different ways to get the target agent name based on SDK version
                        if hasattr(item.raw_item, 'function'):
                            target_agent = item.raw_item.function.name.replace("transfer_to_", "")
                        elif hasattr(item.raw_item, 'name'):
                            target_agent = item.raw_item.name.replace("transfer_to_", "")
                        else:
                            raw_str = str(item.raw_item)
                            if "transfer_to_" in raw_str:
                                target_agent = raw_str.split("transfer_to_")[1].split("(")[0]
                    except:
                        target_agent = "Unknown"
                    
                    # Update UI for handoff
                    if target_agent:
                        if "planner" in target_agent.lower():
                            status_text.text("Planning agent creating search strategy...")
                            progress_bar.progress(0.3)
                        elif "search" in target_agent.lower():
                            status_text.text("Search agent executing web searches...")
                            progress_bar.progress(0.5)
                        elif "writer" in target_agent.lower():
                            status_text.text("Writer agent creating final report...")
                            progress_bar.progress(0.8)
                
                elif item.type == "handoff_output_item":
                    # Extract source and target agent names
                    source_agent = current_agent
                    target_agent = None
                    
                    try:
                        # Try different ways to get the target agent name
                        if hasattr(item, 'target_agent') and item.target_agent:
                            target_agent = item.target_agent.name
                        elif hasattr(item, 'raw_item'):
                            raw_content = str(item.raw_item)
                            if "'assistant':" in raw_content:
                                target_agent = raw_content.split("'assistant': '")[1].split("'")[0]
                    except:
                        target_agent = "Unknown"
                    
                    # Add to handoff details
                    if target_agent:
                        # Determine handoff reason
                        reason = "Agent handoff"
                        if "planner" in target_agent.lower():
                            reason = "Planning search strategy"
                        elif "search" in target_agent.lower():
                            reason = "Execute web searches"
                        elif "writer" in target_agent.lower():
                            reason = "Compile research report"
                        
                        handoff_details.append(HandoffEvent.create(
                            from_agent=source_agent,
                            to_agent=target_agent,
                            reason=reason,
                            input_data="Processing query",
                            output_data="Handling specialized task"
                        ))
                        
                        current_agent = target_agent
                
                # Process tool calls to identify web searches
                elif item.type == "tool_call_item" and "web_search" in str(item.raw_item):
                    try:
                        args = None
                        # Try different ways to get the arguments based on SDK version
                        if hasattr(item.raw_item, 'function') and hasattr(item.raw_item.function, 'arguments'):
                            args = item.raw_item.function.arguments
                        elif hasattr(item.raw_item, 'arguments'):
                            args = item.raw_item.arguments
                        
                        if args:
                            import json
                            try:
                                arg_dict = json.loads(args)
                                if "query" in arg_dict:
                                    current_search_query = arg_dict["query"]
                            except:
                                current_search_query = str(args)
                        else:
                            current_search_query = str(item.raw_item)
                    except:
                        current_search_query = "Unknown search query"
                
                # Process tool outputs to capture search results
                elif item.type == "tool_call_output_item" and current_search_query:
                    try:
                        # Get result text
                        result_text = None
                        if hasattr(item, 'output'):
                            result_text = str(item.output)
                        elif hasattr(item, 'raw_item'):
                            result_text = str(item.raw_item)
                        
                        if result_text:
                            # Try to match with a query from the search plan
                            matching_query = None
                            for sq in search_queries:
                                if sq["query"] in current_search_query or current_search_query in sq["query"]:
                                    matching_query = sq
                                    break
                            
                            # If no match found, create a basic entry
                            if not matching_query:
                                matching_query = {"query": current_search_query, "reason": "Search query"}
                            
                            # Add to search details
                            search_details.append({
                                "query": matching_query["query"],
                                "reason": matching_query["reason"],
                                "result": result_text
                            })
                    except:
                        # Fallback for error
                        search_details.append({
                            "query": current_search_query,
                            "reason": "Unknown reason",
                            "result": "Error retrieving search result"
                        })
            
            # Final step: process the report
            status_text.text("Finalizing research report...")
            progress_bar.progress(0.9)
            
            # Process the final output
            try:
                raw_output = runner_result.final_output
                
                # Handle different output types
                if isinstance(raw_output, ReportData):
                    report = raw_output
                elif isinstance(raw_output, dict):
                    # Convert dictionary to ReportData
                    report = ReportData(
                        short_summary=raw_output.get('short_summary', f"Summary of research on {query}"),
                        markdown_report=raw_output.get('markdown_report', f"# Research on {query}"),
                        follow_up_questions=raw_output.get('follow_up_questions', [f"What more can we learn about {query}?"])
                    )
                else:
                    # Create a fallback report
                    report = ReportData(
                        short_summary=f"Research on '{query}' completed successfully.",
                        markdown_report=(
                            f"# Research on: {query}\n\n"
                            f"The research workflow has completed. Here's what was found:\n\n"
                            + str(raw_output)
                        ),
                        follow_up_questions=[
                            f"What are the most important aspects of {query}?",
                            f"What future developments are expected in {query}?"
                        ]
                    )
                
                # If we have no search details but have search queries, generate basic search details
                if not search_details and search_queries:
                    for query_item in search_queries:
                        search_details.append({
                            "query": query_item["query"],
                            "reason": query_item["reason"],
                            "result": "Search results were used to generate the report, but detailed results weren't captured."
                        })
                        
            except Exception as report_err:
                error_container.error(f"Error processing final output: {str(report_err)}")
                # Create a fallback report
                report = ReportData(
                    short_summary=f"Research on '{query}' was conducted, but there was an issue formatting the final report.",
                    markdown_report=(
                        f"# Research on: {query}\n\n"
                        f"## Research Process\n\n"
                        f"The research process was completed, but there was an error formatting the final report: {str(report_err)}\n\n"
                    ),
                    follow_up_questions=[
                        f"What are the key aspects of {query}?",
                        f"What are expert opinions on {query}?"
                    ]
                )
                
        except Exception as e:
            error_container.error(f"Error in research workflow: {str(e)}. Creating a simplified report instead.")
            # Create a fallback report
            report = ReportData(
                short_summary=f"Research on '{query}' encountered an issue but found some information.",
                markdown_report=(
                    f"# Research on: {query}\n\n"
                    f"## Research Process\n\n"
                    f"The research workflow encountered an issue: {str(e)}\n\n"
                ),
                follow_up_questions=[
                    f"What are the key aspects of {query}?",
                    f"What are expert opinions on {query}?"
                ]
            )
        
        # Update progress
        progress_bar.progress(1.0)
        status_text.text("Research complete!")
    
    # Clean up progress display
    await asyncio.sleep(1)
    progress_container.empty()
    status_text.empty()
    
    # Store handoff details and search details in session state
    session_state.handoff_details = handoff_details
    session_state.search_details = search_details
    
    return report, handoff_details

def render_handoff_chain(handoff_details):
    """Render the handoff chain as a visual flow"""
    if not handoff_details:
        return
    
    st.subheader("Agent Handoff Chain", divider="blue")
    
    for i, handoff in enumerate(handoff_details):
        col1, col2, col3 = st.columns([2, 1, 2])
        
        with col1:
            st.markdown(f"""
            <div style='background-color:var(--bg-card); padding:15px; border-radius:5px; border-left:5px solid var(--primary-color);'>
                <strong>{handoff.from_agent}</strong><br>
                <small>Input: {handoff.input}</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style='text-align:center; padding:10px;'>
                <span style='font-size:20px; color:var(--primary-color);'>↓</span><br>
                <small style='color:var(--text-color);'>{handoff.reason}</small><br>
                <small style='color:var(--text-color); opacity:0.7;'>{handoff.timestamp}</small>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
            <div style='background-color:var(--bg-card); padding:15px; border-radius:5px; border-left:5px solid var(--secondary-color);'>
                <strong>{handoff.to_agent}</strong><br>
                <small>Output: {handoff.output}</small>
            </div>
            """, unsafe_allow_html=True)
        
        if i < len(handoff_details) - 1:
            st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

def main():
    # Set page configuration
    st.set_page_config(
        page_title="AI Research Assistant",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # CSS styling (unchanged)
    st.markdown("""
    <style>
    /* Only style the custom components in the main content area */
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        margin-bottom: 1rem;
    }

    .sub-header {
        font-size: 1.5rem;
        color: #1E88E5;
        margin: 1rem 0;
    }

    .info-box {
        background-color: #F0F7FF;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1E88E5;
        margin-bottom: 1rem;
    }

    .success-box {
        background-color: #F0FFF0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #43A047;
        margin-bottom: 1rem;
    }

    .warning-box {
        background-color: #FFFAF0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #FFA000;
        margin-bottom: 1rem;
    }

    .error-box {
        background-color: #FFF0F0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #E53935;
        margin-bottom: 1rem;
    }

    .stButton button {
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
        padding: 0.5rem 2rem;
        border-radius: 0.3rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state for storing persistent data
    if 'api_key_valid' not in st.session_state:
        st.session_state.api_key_valid = False
    if 'handoff_details' not in st.session_state:
        st.session_state.handoff_details = []
    if 'search_details' not in st.session_state:
        st.session_state.search_details = []
    if 'report' not in st.session_state:
        st.session_state.report = None
    
    # App Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("<h1 class='main-header'>🔍 AI Research Assistant</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color:var(--text-color);'>Powered by OpenAI Agents SDK</p>", unsafe_allow_html=True)
    with col2:
        if AGENTS_SDK_AVAILABLE:
            st.markdown("<div class='success-box'>Agents SDK Loaded ✓</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='warning-box'>Agents SDK Not Found - Using Simulation</div>", unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.markdown("<h2 class='sub-header'>Research Configuration</h2>", unsafe_allow_html=True)
        
        # Display agent workflow
        st.markdown("<h3 style='color:var(--text-color);'>Agent Workflow</h3>", unsafe_allow_html=True)
        
        with st.container():
            # Show runner agent at the top
            st.markdown("""
            <div class='agent-card'>
                <h4 style='color:var(--text-color);'>🧠 Runner Agent</h4>
                <p style='color:var(--text-color);'>Orchestrates the entire research process by delegating to specialized agents in sequence.</p>
                <small style='color:var(--text-color); opacity:0.8;'>Model: GPT-4o</small>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<div style='text-align:center; padding:5px;'><span style='color:var(--primary-color);'>↓ First</span></div>", unsafe_allow_html=True)
            
            st.markdown("""
            <div class='agent-card'>
                <h4 style='color:var(--text-color);'>📋 Planner Agent</h4>
                <p style='color:var(--text-color);'>Analyzes your query and designs a research strategy with targeted search queries.</p>
                <small style='color:var(--text-color); opacity:0.8;'>Model: GPT-4o</small>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<div style='text-align:center; padding:5px;'><span style='color:var(--primary-color);'>↓ Second</span></div>", unsafe_allow_html=True)
            
            st.markdown("""
            <div class='agent-card'>
                <h4 style='color:var(--text-color);'>🔎 Search Agent</h4>
                <p style='color:var(--text-color);'>Performs web searches based on the planned queries and summarizes the findings.</p>
                <small style='color:var(--text-color); opacity:0.8;'>Tools: Web Search</small>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<div style='text-align:center; padding:5px;'><span style='color:var(--primary-color);'>↓ Third</span></div>", unsafe_allow_html=True)
            
            st.markdown("""
            <div class='agent-card'>
                <h4 style='color:var(--text-color);'>✍️ Writer Agent</h4>
                <p style='color:var(--text-color);'>Synthesizes search results into a cohesive, detailed research report.</p>
                <small style='color:var(--text-color); opacity:0.8;'>Output: Formatted Report</small>
            </div>
            """, unsafe_allow_html=True)
        
        # API Key Input
        st.markdown("<h3>OpenAI API Key</h3>", unsafe_allow_html=True)
        
        # Check if API key is already set in environment
        existing_key = os.environ.get('OPENAI_API_KEY', '')
        
        api_key = st.text_input(
            "Enter your OpenAI API Key", 
            value=existing_key, 
            type="password",
            help="Your API key will be used to authenticate with OpenAI services."
        )

        # Number of Searches Slider
        num_searches = st.slider(
            "Number of Web Searches", 
            min_value=1, 
            max_value=10, 
            value=5, 
            step=1,
            help="Maximum number of web searches to perform for your query."
        )

        # Writer Model Selection
        writer_model = st.radio(
            "Report Writer Model",
           ["gpt-4o", "gpt-3.5-turbo"],
           index=0,
           help="Select the model to use for writing the final research report."
       )

       # Validate and save API key
    if api_key:
        if st.button("Validate API Key"):
            is_valid, error_message = validate_openai_key(api_key)
            
            if is_valid:
                st.session_state.api_key_valid = True
                st.markdown("<div class='success-box'>API Key validated successfully! ✓</div>", unsafe_allow_html=True)
            else:
                st.session_state.api_key_valid = False
                st.markdown(f"<div class='error-box'>Invalid API Key: {error_message}</div>", unsafe_allow_html=True)
    
    # Troubleshooting Information (unchanged)
    with st.expander("Troubleshooting"):
        st.markdown("""
        ### 🔑 API Key Issues
        
        1. **Invalid format** - OpenAI API keys start with `sk-`
        2. **Authentication failed** - Check your OpenAI account status
        3. **Billing issues** - Ensure your OpenAI account has valid billing
        4. **Usage limits** - You may have hit your usage cap
        
        ### 📡 Connection Issues
        
        1. **Network problems** - Check your internet connection
        2. **Timeouts** - The request might be taking too long
        3. **Service disruption** - OpenAI services might be experiencing issues
        
        For more help, visit [OpenAI Support](https://help.openai.com/)
        """)

   # Main content area
    if api_key and st.session_state.api_key_valid:
        # Setup only the runner agent with the validated key
        try:
            runner_agent = setup_agents(
                api_key, 
                num_searches, 
                writer_model
            )
        except Exception as e:
            st.error(f"Error setting up agents: {e}")
        
        # Query input
        st.markdown("<h2 class='sub-header'>Research Query</h2>", unsafe_allow_html=True)
        query = st.text_input(
            "What would you like to research?",
            placeholder="E.g., 'The impact of artificial intelligence on healthcare'"
        )
        
        # Research button
        if st.button("Start Research"):
            if query:
                try:
                    # Reset previous results
                    st.session_state.report = None
                    st.session_state.handoff_details = []
                    st.session_state.search_details = []
                    
                    # Use asyncio to run the async function with just the runner agent
                    report, handoff_details = asyncio.run(perform_research(
                        query, 
                        runner_agent,
                        st.session_state
                    ))
                    
                    # Store the report in session state
                    st.session_state.report = report
                    
                    # Force a rerun to show the new content
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"An error occurred during research: {e}")
            else:
                st.warning("Please enter a research query.")
    
        # Display results if available (same as original)
        if st.session_state.report:
            report = st.session_state.report
            
            # Research Results
            st.markdown("<h2 class='sub-header'>Research Results</h2>", unsafe_allow_html=True)
            
            # Display summary in a nice box
            st.markdown(f"""
            <div class='info-box'>
                <h3>Summary</h3>
                <p>{report.short_summary}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Render the handoff chain
            render_handoff_chain(st.session_state.handoff_details)
            
            # Tabs for detailed content
            tab1, tab2, tab3 = st.tabs(["📝 Full Report", "🔍 Search Details", "❓ Follow-up Questions"])
            
            with tab1:
                st.markdown(report.markdown_report)
            
            with tab2:
                if st.session_state.search_details:
                    for i, detail in enumerate(st.session_state.search_details):
                        with st.expander(f"Search {i+1}: {detail['query']}"):
                            st.markdown(f"**Reason:** {detail['reason']}")
                            st.markdown(f"**Result:** {detail['result']}")
                else:
                    st.info("No search details available.")
            
            with tab3:
                if report.follow_up_questions:
                    for question in report.follow_up_questions:
                        st.markdown(f"- {question}")
                else:
                    st.info("No follow-up questions generated.")
    else:
        # When no API key is present or valid (same as original)
        st.markdown("""
        <div class='info-box'>
            <h3>Getting Started</h3>
            <p>To use the AI Research Assistant, please follow these steps:</p>
            <ol>
                <li>Enter your OpenAI API Key in the sidebar</li>
                <li>Click "Validate API Key" to verify your credentials</li>
                <li>Enter your research query and start researching!</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        # Sample queries to help users get started
        st.markdown("<h3>Sample Research Topics</h3>", unsafe_allow_html=True)
        sample_queries = [
            "The environmental impact of electric vehicles",
            "Advancements in quantum computing in the last 5 years",
            "The role of gut microbiome in human health",
            "The economic effects of remote work"
        ]
        
        for query in sample_queries:
            st.markdown(f"- {query}")

if __name__ == "__main__":
    main()