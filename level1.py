import streamlit as st
import os
import asyncio
from typing import Dict, Any
import random

# Import OpenAI Agents SDK components
from agents import Agent, Runner, function_tool, RunContextWrapper, WebSearchTool
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions

# Configure the page
st.set_page_config(
    page_title="Agents SDK - Level 1 Demo",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
    }
    .chat-message.user {
        background-color: #e6f7ff;
        border-left: 5px solid #1890ff;
    }
    .chat-message.assistant {
        background-color: #f6ffed;
        border-left: 5px solid #52c41a;
    }
    .chat-message .avatar {
        width: 20%;
    }
    .chat-message .content {
        width: 80%;
    }
    .sidebar-content {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .handoff-indicator {
        padding: 0.5rem;
        background-color: #fff1f0;
        border-left: 5px solid #ff4d4f;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'api_key_set' not in st.session_state:
    st.session_state.api_key_set = False
if 'last_agent' not in st.session_state:
    st.session_state.last_agent = "Triage Agent"
if 'handoffs' not in st.session_state:
    st.session_state.handoffs = []

# Main title
st.title("🤖 OpenAI Agents SDK - Level 1 Demo")
st.subheader("Agents, Runners, and Handoffs")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # API Key input
    api_key = st.text_input("OpenAI API Key", type="password", 
                            help="Enter your OpenAI API key to use the Agents SDK")
    
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        st.session_state.api_key_set = True
    
    st.divider()
    
    # Toggle for showing handoffs
    show_handoffs = st.toggle("Show Agent Handoffs", value=True,
                             help="Show when the system routes to a different agent")
    
    # Reset conversation
    if st.button("Reset Conversation"):
        st.session_state.messages = []
        st.session_state.handoffs = []
        st.session_state.last_agent = "Triage Agent"
        st.rerun()
    
    st.divider()
    
    # Description of the demo
    st.markdown("""
    <div class="sidebar-content">
    <h4>About This Demo</h4>
    <p>This Streamlit app demonstrates Level 1 concepts from the OpenAI Agents SDK:</p>
    <ul>
        <li>Creating different types of specialized agents</li>
        <li>Using the Runner to execute agent tasks</li>
        <li>Setting up agent handoffs for task routing</li>
        <li>Implementing function tools and web search</li>
    </ul>
    <p>Ask any question and our triage agent will route your request to the most appropriate specialist!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show current state
    st.markdown(f"""
    <div class="sidebar-content">
    <h4>Current Agent: {st.session_state.last_agent}</h4>
    <p>Total handoffs: {len(st.session_state.handoffs)}</p>
    </div>
    """, unsafe_allow_html=True)

# Define a simple context class
class DemoContext:
    def __init__(self, user_name: str = "User"):
        self.user_name = user_name
        self.conversation_history = []

# Define tools
@function_tool
def get_current_time(ctx: RunContextWrapper[DemoContext]) -> str:
    """Get the current date and time."""
    from datetime import datetime
    now = datetime.now()
    return f"The current date and time is {now.strftime('%Y-%m-%d %H:%M:%S')}"

@function_tool
def calculate(ctx: RunContextWrapper[DemoContext], expression: str) -> str:
    """Calculate the result of a mathematical expression."""
    try:
        # Using eval with caution for demo purposes
        result = eval(expression)
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"

@function_tool
def get_weather(ctx: RunContextWrapper[DemoContext], location: str) -> str:
    """Get the current weather for a location (simulated)."""
    conditions = ["sunny", "cloudy", "rainy", "snowy", "windy", "foggy"]
    temps = random.randint(0, 40)  # Celsius
    condition = random.choice(conditions)
    return f"The current weather in {location} is {condition} with a temperature of {temps}°C"

@function_tool
def generate_itinerary(ctx: RunContextWrapper[DemoContext], location: str, days: int) -> str:
    """Generate a travel itinerary for the specified location and duration."""
    activities = [
        "Visit the local museum",
        "Try authentic local cuisine at a restaurant",
        "Take a guided tour of historical sites",
        "Explore natural parks and hiking trails",
        "Visit the shopping district",
        "Relax at a local beach or park",
        "Attend a cultural event or performance",
        "Take a day trip to a nearby attraction"
    ]
    
    itinerary = f"🌟 {days}-Day Itinerary for {location} 🌟\n\n"
    
    for day in range(1, min(days + 1, 5)):  # Cap at 4 days for demo
        itinerary += f"Day {day}:\n"
        day_activities = random.sample(activities, min(3, len(activities)))
        for i, activity in enumerate(day_activities):
            itinerary += f"  • {activity}\n"
        itinerary += "\n"
    
    return itinerary

# Create specialized agents
def create_specialized_agents():
    # Math tutor agent
    math_agent = Agent[DemoContext](
        name="Math Tutor",
        handoff_description="Specialist for mathematical problems and calculations",
        instructions=prompt_with_handoff_instructions("""
        You are a math tutor that helps with mathematical problems and concepts.
        Explain your reasoning clearly, step by step. Use the calculate tool when calculations are needed.
        Always verify your calculations and provide intuitive explanations along with formal solutions.
        """),
        tools=[calculate]
    )
    
    # General information agent
    info_agent = Agent[DemoContext](
        name="General Information Assistant",
        handoff_description="Specialist for general knowledge and information queries, excluding weather and travel topics",
        instructions=prompt_with_handoff_instructions("""
        You are a helpful assistant that provides general information on a wide range of topics.
        Be concise but thorough in your responses. Provide factual information and acknowledge when 
        you're uncertain. Use the current time tool when time-related questions are asked.
        
        NOTE: You do NOT handle weather queries or travel questions - those are handled by the Travel Advisor.
        """),
        tools=[get_current_time]
    )
    
    # Creative content agent
    creative_agent = Agent[DemoContext](
        name="Creative Content Assistant",
        handoff_description="Specialist for creative writing, stories, and content generation",
        instructions=prompt_with_handoff_instructions("""
        You are a creative writing assistant that helps with generating stories, poetry, and creative content.
        Be imaginative and engaging in your responses. Adjust your tone to match the user's request.
        Provide original content that is appropriate and tailored to the user's specifications.
        """)
    )
    
    # Travel agent with web search
    travel_agent = Agent[DemoContext](
        name="Travel Advisor",
        handoff_description="Specialist for travel topics including weather information for any location, travel recommendations, and itineraries",
        instructions=prompt_with_handoff_instructions("""
        You are a travel advisor that provides recommendations for destinations, activities, and travel planning.
        You are ALSO the weather specialist and handle ALL weather-related questions for ANY location.
        
        Use the weather tool to check conditions at destinations. Provide helpful travel tips and consider 
        the user's preferences. Use the generate_itinerary tool for creating custom travel plans.
        Use web search when you need up-to-date information about destinations, travel requirements, 
        or specific attractions.
        """),
        tools=[get_weather, generate_itinerary, WebSearchTool()]
    )
    
    return [math_agent, info_agent, creative_agent, travel_agent]

# Create triage agent
def create_triage_agent():
    specialized_agents = create_specialized_agents()
    
    triage_agent = Agent[DemoContext](
        name="Triage Agent",
        instructions=prompt_with_handoff_instructions("""
        You are the initial point of contact for all user queries. Your job is to analyze the user's question
        and route it to the most appropriate specialized agent:
        
        - For math problems, equations, or numerical calculations, use the Math Tutor.
        - For general knowledge, facts, or information requests, use the General Information Assistant.
        - For creative writing, stories, poems, or content creation, use the Creative Content Assistant.
        - For travel plans, destination recommendations, or itineraries, use the Travel Advisor.
        
        If the query doesn't clearly fit any specialist, handle it yourself using your general knowledge.
        Always prioritize giving the user the best experience by routing to the appropriate specialist.
        """),
        handoffs=specialized_agents
    )
    
    return triage_agent

# Process user message with agent system
async def process_with_agent_system(user_prompt, context):
    try:
        # Create the triage agent and get the result
        triage_agent = create_triage_agent()
        
        # Track handoffs
        original_handoffs_count = len(st.session_state.handoffs)
        
        # Include conversation history if it exists
        if st.session_state.messages:
            # Convert previous messages to input list format
            input_list = []
            for msg in st.session_state.messages:
                role = "user" if msg["role"] == "user" else "assistant"
                input_list.append({"role": role, "content": msg["content"]})
            
            # Add the current user prompt
            input_list.append({"role": "user", "content": user_prompt})
            
            # Run the agent system with history
            result = await Runner.run(triage_agent, input_list, context=context)
        else:
            # First message - no history
            result = await Runner.run(triage_agent, user_prompt, context=context)
        
        # Store information about which agent responded
        st.session_state.last_agent = result.last_agent.name
        
        # Check for handoffs by comparing the result's agent with the triage agent
        if result.last_agent.name != triage_agent.name:
            st.session_state.handoffs.append({
                "from": "Triage Agent",
                "to": result.last_agent.name,
                "query": user_prompt
            })
        
        return result.final_output, result.last_agent.name, (len(st.session_state.handoffs) > original_handoffs_count)
    except Exception as e:
        return f"Error: {str(e)}", "Error", False

# Get avatar URL based on agent name
def get_avatar_url(agent_name):
    avatar_seeds = {
        "Triage Agent": "triage",
        "Math Tutor": "calculator",
        "General Information Assistant": "info",
        "Creative Content Assistant": "artist",
        "Travel Advisor": "globe",
        "Error": "error"
    }
    
    seed = avatar_seeds.get(agent_name, "default")
    return f"https://api.dicebear.com/7.x/bottts/svg?seed={seed}"

# Display chat messages
def display_chat(show_handoffs):
    for i, message in enumerate(st.session_state.messages):
        with st.container():
            # Show handoff indicator if this is the first message after a handoff
            if show_handoffs and i > 0 and message.get("handoff", False):
                st.markdown(f"""
                <div class="handoff-indicator">
                    🔄 <strong>Handoff:</strong> Question routed to <strong>{message["agent"]}</strong> for specialized assistance
                </div>
                """, unsafe_allow_html=True)
                
            # Display the message
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user">
                    <div class="avatar">
                        <img src="https://api.dicebear.com/7.x/personas/svg?seed=user" width="50" style="border-radius: 50%">
                    </div>
                    <div class="content">
                        <p>{message["content"]}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                avatar_url = get_avatar_url(message["agent"])
                st.markdown(f"""
                <div class="chat-message assistant">
                    <div class="avatar">
                        <img src="{avatar_url}" width="50" style="border-radius: 50%">
                    </div>
                    <div class="content">
                        <p><small style="color: #888;">From: {message["agent"]}</small></p>
                        <p>{message["content"]}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)

# Display available agents and example questions more compactly
st.subheader("Available Specialist Agents")
st.write("Our system has several specialized agents ready to help you. Ask any question and our triage agent will route your request to the most appropriate specialist!")

# Create compact layout for agent cards
col1, col2 = st.columns(2)

# Helper function to get avatar URL
def get_compact_avatar(agent_name):
    avatar_seeds = {
        "Math Tutor": "calculator",
        "General Information": "info",
        "Creative Content": "artist",
        "Travel Advisor": "globe"
    }
    seed = avatar_seeds.get(agent_name, "default")
    return f"https://api.dicebear.com/7.x/bottts/svg?seed={seed}"

# More compact agent cards with avatars
with col1:
    # Math Tutor
    st.markdown(f"""
    <div style="display: flex; align-items: center; margin-bottom: 5px;">
        <img src="{get_compact_avatar('Math Tutor')}" width="40" style="border-radius: 50%; margin-right: 10px;">
        <h3 style="margin: 0;">🧮 Math Tutor</h3>
    </div>
    <p><strong>🔧 Tools:</strong> Calculate expressions</p>
    <p><em>Example:</em> What is the formula for the area of a circle?</p>
    """, unsafe_allow_html=True)
    
    # Creative Content below Math
    st.markdown(f"""
    <div style="display: flex; align-items: center; margin-bottom: 5px; margin-top: 15px;">
        <img src="{get_compact_avatar('Creative Content')}" width="40" style="border-radius: 50%; margin-right: 10px;">
        <h3 style="margin: 0;">✍️ Creative Content</h3>
    </div>
    <p><strong>🔧 Tools:</strong> Natural language generation</p>
    <p><em>Example:</em> Write a short poem about autumn</p>
    """, unsafe_allow_html=True)

with col2:
    # General Information
    st.markdown(f"""
    <div style="display: flex; align-items: center; margin-bottom: 5px;">
        <img src="{get_compact_avatar('General Information')}" width="40" style="border-radius: 50%; margin-right: 10px;">
        <h3 style="margin: 0;">ℹ️ General Information</h3>
    </div>
    <p><strong>🔧 Tools:</strong> Get current time</p>
    <p><em>Example:</em> What time is it right now?</p>
    """, unsafe_allow_html=True)
    
    # Travel Advisor below General Information
    st.markdown(f"""
    <div style="display: flex; align-items: center; margin-bottom: 5px; margin-top: 15px;">
        <img src="{get_compact_avatar('Travel Advisor')}" width="40" style="border-radius: 50%; margin-right: 10px;">
        <h3 style="margin: 0;">🌎 Travel Advisor</h3>
    </div>
    <p><strong>🔧 Tools:</strong> Weather, itineraries, web search</p>
    <p><em>Example:</em> What's the weather like in Tokyo?</p>
    """, unsafe_allow_html=True)

st.divider()

# Main chat interface
if not st.session_state.api_key_set:
    st.warning("Please enter your OpenAI API key in the sidebar to use this demo.")
else:
    # Chat interface
    st.subheader("Chat with AI Assistant System")
    display_chat(show_handoffs)
    
    # User input
    prompt = st.chat_input("Type your message here...")
    
    if prompt:
        # Add user message to chat
        st.session_state.messages.append({
            "role": "user", 
            "content": prompt,
            "agent": "User"
        })
        
        # Display updated chat
        st.rerun()

# Process messages if the most recent one is from the user
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user" and st.session_state.api_key_set:
    with st.spinner("AI system is processing your request..."):
        # Create context
        context = DemoContext()
        
        # Get user prompt
        prompt = st.session_state.messages[-1]["content"]
        
        # Process with agent system
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        response, agent_name, handoff_occurred = loop.run_until_complete(process_with_agent_system(prompt, context))
        loop.close()
        
        # Add response to chat
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response,
            "agent": agent_name,
            "handoff": handoff_occurred
        })
        
        # Update display
        st.rerun()

# Show technical details in an expander
with st.expander("See Agent SDK Details"):
    # Tabs for different aspects of the code
    tab1, tab2, tab3, tab4 = st.tabs(["Agent Creation", "Handoffs", "Conversation History", "Complete Agent Code"])
    
    with tab1:
        st.markdown("""
        ## Creating Agents
        
        ```python
        # Create a specialized agent
        math_agent = Agent[DemoContext](
            name="Math Tutor",
            handoff_description="Specialist for mathematical problems and calculations",
            instructions="You are a math tutor that helps with mathematical problems...",
            tools=[calculate]  # Tools this agent can use
        )
        
        # Create a tool
        @function_tool
        def calculate(ctx: RunContextWrapper[DemoContext], expression: str) -> str:
            \"\"\"Calculate the result of a mathematical expression.\"\"\"
            try:
                result = eval(expression)
                return f"The result of {expression} is {result}"
            except Exception as e:
                return f"Error calculating {expression}: {str(e)}"
        ```
        """)
    
    with tab2:
        st.markdown("""
        ## Setting Up Handoffs
        
        ```python
        # Create specialized agents
        specialized_agents = [math_agent, info_agent, creative_agent, travel_agent]
        
        # Create a triage agent that can hand off to specialists
        triage_agent = Agent[DemoContext](
            name="Triage Agent",
            instructions="You are the initial point of contact for all user queries...",
            handoffs=specialized_agents  # This defines the agents that can be handed off to
        )
        ```
        
        The triage agent's instructions should specify when to use each specialist agent. The SDK handles the mechanics of the handoff automatically.
        """)
    
    with tab3:
        st.markdown("""
        ## Handling Conversation History
        
        ```python
        # Convert previous messages to input list format
        input_list = []
        for msg in st.session_state.messages:
            role = "user" if msg["role"] == "user" else "assistant"
            input_list.append({"role": role, "content": msg["content"]})
        
        # Add the current user prompt
        input_list.append({"role": "user", "content": user_prompt})
        
        # Run the agent system with history
        result = await Runner.run(triage_agent, input_list, context=context)
        ```
        
        By providing the full conversation history to the agent each time, we enable it to understand the context of the current question.
        """)
    
    with tab4:
        st.code("""
# Create specialized agents
def create_specialized_agents():
    # Math tutor agent
    math_agent = Agent[DemoContext](
        name="Math Tutor",
        handoff_description="Specialist for mathematical problems and calculations",
        instructions=prompt_with_handoff_instructions('''
        You are a math tutor that helps with mathematical problems and concepts.
        Explain your reasoning clearly, step by step. Use the calculate tool when calculations are needed.
        Always verify your calculations and provide intuitive explanations along with formal solutions.
        '''),
        tools=[calculate]
    )
    
    # General information agent
    info_agent = Agent[DemoContext](
        name="General Information Assistant",
        handoff_description="Specialist for general knowledge and information queries",
        instructions=prompt_with_handoff_instructions('''
        You are a helpful assistant that provides general information on a wide range of topics.
        Be concise but thorough in your responses. Provide factual information and acknowledge when 
        you're uncertain. Use the current time tool when time-related questions are asked.
        '''),
        tools=[get_current_time]
    )
    
    # Creative content agent
    creative_agent = Agent[DemoContext](
        name="Creative Content Assistant",
        handoff_description="Specialist for creative writing, stories, and content generation",
        instructions=prompt_with_handoff_instructions('''
        You are a creative writing assistant that helps with generating stories, poetry, and creative content.
        Be imaginative and engaging in your responses. Adjust your tone to match the user's request.
        Provide original content that is appropriate and tailored to the user's specifications.
        ''')
    )
    
    # Travel agent with web search
    travel_agent = Agent[DemoContext](
        name="Travel Advisor",
        handoff_description="Specialist for travel recommendations and planning",
        instructions=prompt_with_handoff_instructions('''
        You are a travel advisor that provides recommendations for destinations, activities, and travel planning.
        Use the weather tool to check conditions at destinations. Provide helpful travel tips and consider 
        the user's preferences. Use the generate_itinerary tool for creating custom travel plans.
        Use web search when you need up-to-date information about destinations, travel requirements, 
        or specific attractions.
        '''),
        tools=[get_weather, generate_itinerary, WebSearchTool()]
    )
    
    return [math_agent, info_agent, creative_agent, travel_agent]

# Create triage agent
def create_triage_agent():
    specialized_agents = create_specialized_agents()
    
    triage_agent = Agent[DemoContext](
        name="Triage Agent",
        instructions=prompt_with_handoff_instructions('''
        You are the initial point of contact for all user queries. Your job is to analyze the user's question
        and route it to the most appropriate specialized agent:
        
        - For math problems, equations, or numerical calculations, use the Math Tutor.
        - For general knowledge, facts, or information requests, use the General Information Assistant.
        - For creative writing, stories, poems, or content creation, use the Creative Content Assistant.
        - For travel-related queries, use the Travel Advisor, including:
          * Weather questions about specific destinations
          * Itinerary requests or travel planning
          * Questions about visiting specific locations
          * Any travel recommendations or advice
        
        IMPORTANT: ANY questions about weather in specific cities or locations should ALWAYS go to the Travel Advisor,
        as they have specialized weather tools. Weather questions should NOT go to the General Information Assistant.
        
        If the query doesn't clearly fit any specialist, handle it yourself using your general knowledge.
        Always prioritize giving the user the best experience by routing to the appropriate specialist.
        '''),
        handoffs=specialized_agents
    )
    
    return triage_agent
""", language="python")

    # Display current handoffs
    if st.session_state.handoffs:
        st.subheader("Handoffs in Current Conversation")
        for i, handoff in enumerate(st.session_state.handoffs):
            st.markdown(f"""
            **Handoff {i+1}**:  
            From: `{handoff['from']}`  
            To: `{handoff['to']}`  
            Query: "{handoff['query']}"
            """)
    else:
        st.info("No handoffs have occurred in this conversation yet.")