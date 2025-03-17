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
        # Return mock agents when SDK isn't available
        class MockAgent:
            def __init__(self, name, instructions="", model=""):
                self.name = name
                self.instructions = instructions
                self.model = model
        
        planner_agent = MockAgent(
            name="PlannerAgent",
            instructions=f"Create a set of web searches to perform to best answer the query. Output between 1-{num_searches} search terms.",
            model="gpt-4o"
        )
        
        search_agent = MockAgent(
            name="SearchAgent",
            instructions="Search the web for the given term and produce a concise summary of the results."
        )
        
        writer_agent = MockAgent(
            name="WriterAgent",
            instructions="Write a cohesive report based on the research.",
            model=writer_model
        )
        
        return planner_agent, search_agent, writer_agent
    
    # Real implementation with Agents SDK
    # Configure the OpenAI client globally
    os.environ['OPENAI_API_KEY'] = api_key

    # Dynamic Planner Prompt
    PLANNER_PROMPT = (
        "You are a helpful research assistant. Given a query, come up with a "
        f"set of web searches to perform to best answer the query. "
        f"Output between 1-{num_searches} search terms to query for."
    )

    # Planner Agent
    planner_agent = Agent(
        name="PlannerAgent",
        instructions=PLANNER_PROMPT,
        model="gpt-4o",
        output_type=WebSearchPlan,
    )

    # Search Agent - Simplified and more focused instructions
    SEARCH_INSTRUCTIONS = (
        "You are a research assistant performing web searches and providing concise information. "
        "For each search term:\n"
        "1. Extract key facts, data points, and insights relevant to the query\n"
        "2. Focus on objective information from reliable sources\n"
        "3. Create a brief summary (150-200 words maximum)\n"
        "4. Format information as bullet points whenever possible\n"
        "5. Prioritize recent and factual information\n\n"
        "DO NOT include personal opinions, lengthy explanations, or irrelevant details. "
        "Your summary will be used to create a research report, so focus on extracting "
        "the most valuable information."
    )

    search_agent = Agent(
        name="SearchAgent",
        instructions=SEARCH_INSTRUCTIONS,
        tools=[WebSearchTool(user_location={"type": "approximate", "city": "New York"})],
    )

    # Writer Agent - Simple prompt with clear instructions
    WRITER_PROMPT = (
        "You are a research writer creating a report from search results. "
        "You will receive a query and search results. Create a report with: "
        "1. A concise summary (1-2 paragraphs) "
        "2. A markdown-formatted report (500-800 words) "
        "3. 2-3 follow-up questions"
    )

    # Create the writer agent with standard settings
    writer_agent = Agent(
        name="WriterAgent",
        instructions=WRITER_PROMPT,
        model=writer_model,
        output_type=ReportData,
    )

    return planner_agent, search_agent, writer_agent

async def perform_research(query: str, planner_agent, search_agent, writer_agent, session_state):
    """Async function to perform the research workflow"""
    handoff_details = []
    search_results = []
    search_details = []
    
    # Initialize the progress container
    progress_container = st.empty()
    progress_bar = progress_container.progress(0)
    status_text = st.empty()
    error_container = st.empty()
    
    if not AGENTS_SDK_AVAILABLE:
        # Simulate the research process
        # Plan searches
        status_text.text("Planning searches...")
        await asyncio.sleep(1.5)
        
        # Mock search plan
        search_plan = WebSearchPlan(searches=[
            WebSearchItem(reason="To get latest information", query=f"latest developments in {query}"),
            WebSearchItem(reason="To get historical context", query=f"history of {query}"),
            WebSearchItem(reason="To get expert opinions", query=f"expert analysis {query}"),
        ])
        
        # Record handoff details
        handoff_details.append(HandoffEvent.create(
            from_agent="User",
            to_agent="PlannerAgent",
            reason="Initial query processing",
            input_data=query,
            output_data=f"Created search plan with {len(search_plan.searches)} queries"
        ))
        
        # Update progress
        progress_bar.progress(0.2)
        status_text.text("Search planning complete")
        
        # Perform searches
        for i, item in enumerate(search_plan.searches):
            status_text.text(f"Searching: {item.query}")
            await asyncio.sleep(2)
            
            # Mock search result
            search_result = f"Found information about {item.query}. This includes various sources and citations relevant to the query."
            search_results.append(search_result)
            
            # Collect detailed run information
            search_details.append({
                "query": item.query,
                "reason": item.reason,
                "result": search_result,
            })
            
            # Record handoff details
            handoff_details.append(HandoffEvent.create(
                from_agent="PlannerAgent",
                to_agent="SearchAgent",
                reason=item.reason,
                input_data=item.query,
                output_data=search_result
            ))
            
            # Update progress
            progress = 0.2 + (0.5 * (i+1) / len(search_plan.searches))
            progress_bar.progress(progress)
            status_text.text(f"Completed search {i+1} of {len(search_plan.searches)}")
        
        # Write report
        status_text.text("Writing research report...")
        await asyncio.sleep(3)
        
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
            input_data="Search results and analysis",
            output_data="Created comprehensive report with summary and follow-up questions"
        ))
        
        # Update progress
        progress_bar.progress(1.0)
        status_text.text("Research complete!")
        
    else:
        # Use actual Agents SDK
        # Plan searches
        status_text.text("Planning searches...")
        with trace("Research Planning"):
            search_plan_result = await Runner.run(planner_agent, f"Query: {query}")
            search_plan = search_plan_result.final_output_as(WebSearchPlan)
        
        # Record handoff
        handoff_details.append(HandoffEvent.create(
            from_agent="User",
            to_agent="PlannerAgent",
            reason="Initial query processing",
            input_data=query,
            output_data=f"Created search plan with {len(search_plan.searches)} queries"
        ))
        
        # Update progress
        progress_bar.progress(0.2)
        status_text.text("Search planning complete")
        
        # Perform searches
        for i, item in enumerate(search_plan.searches):
            status_text.text(f"Searching: {item.query}")
            with trace(f"Web Search: {item.query}"):
                # Construct search input with query and reason
                search_input = f"Search the web for '{item.query}' and provide a concise summary. Reason: {item.reason}"
                
                try:
                    # Perform web search using the search agent with timeout
                    search_result = await asyncio.wait_for(
                        Runner.run(search_agent, search_input),
                        timeout=60  # 60 second timeout per search
                    )
                    
                    # Collect detailed run information
                    result_text = str(search_result.final_output)
                    
                    # Trim very long results to prevent issues in the writer stage
                    if len(result_text) > 1000:
                        result_text = result_text[:997] + "..."
                    
                    search_details.append({
                        "query": item.query,
                        "reason": item.reason,
                        "result": result_text,
                        "new_items": search_result.new_items,
                        "last_agent": search_result.last_agent.name,
                    })
                    
                    # Record handoff
                    handoff_details.append(HandoffEvent.create(
                        from_agent="PlannerAgent",
                        to_agent="SearchAgent",
                        reason=item.reason,
                        input_data=item.query,
                        output_data=str(search_result.final_output)
                    ))
                    
                    search_results.append(str(search_result.final_output))
                except Exception as e:
                    st.error(f"Error searching {item.query}: {e}")
                    search_results.append(f"Could not retrieve information for {item.query}")
            
            # Update progress
            progress = 0.2 + (0.5 * (i+1) / len(search_plan.searches))
            progress_bar.progress(progress)
            status_text.text(f"Completed search {i+1} of {len(search_plan.searches)}")
        
        # Write report with better error handling
        status_text.text("Handing off to writer agent...")
        error_container.empty()
        
        try:
            with trace("Report Writing"):
                # Format the input in a more structured way
                formatted_search_results = "\n\n".join([
                    f"SEARCH RESULT {i+1}:\n{result}" 
                    for i, result in enumerate(search_results)
                ])
                
                # Create explicit handoff from search agent to writer agent
                handoff_details.append(HandoffEvent.create(
                    from_agent="SearchAgent",
                    to_agent="WriterAgent",
                    reason="Compile research report",
                    input_data="Search results compilation",
                    output_data="Creating comprehensive research report"
                ))
                
                # Create simple input for the writer agent
                writer_input = f"RESEARCH QUERY: {query}\n\n{formatted_search_results}\n\nCreate a research report on this topic."
                
                # Set timeout to avoid waiting too long
                status_text.text("Generating research report...")
                
                # Run the writer agent directly
                report_result = await asyncio.wait_for(
                    Runner.run(writer_agent, writer_input),
                    timeout=120  # 2 minute timeout
                )
                
                # Get the report from the result
                raw_output = report_result.final_output
                
                # Handle different types of output
                if isinstance(raw_output, ReportData):
                    # Already a ReportData object
                    report = raw_output
                elif isinstance(raw_output, dict):
                    # We have a dictionary output, create ReportData manually
                    report = ReportData(
                        short_summary=raw_output.get('short_summary', f"Summary of research on {query}"),
                        markdown_report=raw_output.get('markdown_report', f"# Research on {query}"),
                        follow_up_questions=raw_output.get('follow_up_questions', [f"What more can we learn about {query}?"])
                    )
                else:
                    # If we got a string or other unexpected type, create a structured report
                    error_container.warning("Writer agent output format unexpected, creating structured report.")
                    
                    if isinstance(raw_output, str):
                        # Use the string as the report content
                        report = ReportData(
                            short_summary=f"Research on '{query}' found relevant information from {len(search_results)} sources.",
                            markdown_report=raw_output if len(raw_output) > 100 else f"# Research on {query}\n\n" + "\n\n".join(search_results[:3]),
                            follow_up_questions=[f"What are the key aspects of {query}?", f"How might {query} evolve in the future?"]
                        )
                    else:
                        # Create a completely new report
                        report = ReportData(
                            short_summary=f"Research on '{query}' found information from multiple sources.",
                            markdown_report=f"# Research Report: {query}\n\n" + "\n\n".join([f"## Source {i+1}\n{result[:300]}..." for i, result in enumerate(search_results)]),
                            follow_up_questions=[f"What is the historical significance of {query}?", f"What are modern perspectives on {query}?"]
                        )
                
        except Exception as e:
            error_container.error(f"Error generating report: {str(e)}. Creating a simplified report instead.")
            # Create a fallback simple report
            report = ReportData(
                short_summary=f"Research on '{query}' found relevant information from {len(search_results)} different sources.",
                markdown_report=(
                    f"# Research on: {query}\n\n"
                    f"## Search Results Summary\n\n"
                    + "\n\n".join([f"### Result {i+1}\n\n{result[:300]}..." for i, result in enumerate(search_results)])
                ),
                follow_up_questions=[
                    f"What are the most important aspects of {query}?",
                    f"What future developments are expected in {query}?"
                ]
            )
        
        # Update progress
        progress_bar.progress(1.0)
        status_text.text("Research complete!")
    
    # Clean up progress display
    await asyncio.sleep(1)
    progress_container.empty()
    status_text.empty()
    
    # Store handoff details in session state for persistent display
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
    
    # Detect if dark mode is enabled
    # We'll use a JavaScript hack to detect the theme
    # and set a session state variable
    # CSS with dark sidebar and clean main content
    # CSS with precise sidebar styling
# Simplified CSS that doesn't affect the sidebar styling
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

    /* Remove any styling that affects the sidebar */
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
        
        # Display agent cards
        st.markdown("<h3 style='color:var(--text-color);'>Agent Workflow</h3>", unsafe_allow_html=True)
        
        with st.container():
            st.markdown("""
            <div class='agent-card'>
                <h4 style='color:var(--text-color);'>📋 Planner Agent</h4>
                <p style='color:var(--text-color);'>Analyzes your query and designs a research strategy with targeted search queries.</p>
                <small style='color:var(--text-color); opacity:0.8;'>Model: GPT-4o</small>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<div style='text-align:center; padding:5px;'><span style='color:var(--primary-color);'>↓</span></div>", unsafe_allow_html=True)
            
            st.markdown("""
            <div class='agent-card'>
                <h4 style='color:var(--text-color);'>🔎 Search Agent</h4>
                <p style='color:var(--text-color);'>Performs web searches based on the planned queries and summarizes the findings.</p>
                <small style='color:var(--text-color); opacity:0.8;'>Tools: Web Search</small>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<div style='text-align:center; padding:5px;'><span style='color:var(--primary-color);'>↓</span></div>", unsafe_allow_html=True)
            
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
    
    # Troubleshooting Information
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
        # Setup agents with the validated key
        try:
            planner_agent, search_agent, writer_agent = setup_agents(
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
                    
                    # Use asyncio to run the async function
                    report, handoff_details = asyncio.run(perform_research(
                        query, 
                        planner_agent, 
                        search_agent, 
                        writer_agent,
                        st.session_state
                    ))
                    
                    # Store the report in session state
                    st.session_state.report = report
                    
                    # Force a rerun to show the new content (using the correct function)
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"An error occurred during research: {e}")
            else:
                st.warning("Please enter a research query.")
    
        # Display results if available
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
                            
                            # Show additional details if available
                            if 'new_items' in detail:
                                st.markdown("**Source Agent:** " + detail.get('last_agent', 'Unknown'))
                else:
                    st.info("No search details available.")
            
            with tab3:
                if report.follow_up_questions:
                    for question in report.follow_up_questions:
                        st.markdown(f"- {question}")
                else:
                    st.info("No follow-up questions generated.")
    else:
        # When no API key is present or valid
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