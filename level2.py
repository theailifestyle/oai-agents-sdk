import streamlit as st
import os
import asyncio
from typing import Dict, Any, Optional
import random

# Import OpenAI Agents SDK components
# Import OpenAI Agents SDK components
from agents import (
    Agent, Runner, function_tool, RunContextWrapper, WebSearchTool,
    GuardrailFunctionOutput, InputGuardrail, OutputGuardrail,
    InputGuardrailTripwireTriggered, OutputGuardrailTripwireTriggered,
    trace
)
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions
from pydantic import BaseModel

# Configure the page
st.set_page_config(
    page_title="Agents SDK - Level 2 Demo",
    page_icon="🛡️",
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
    .guardrail-indicator {
        padding: 0.5rem;
        background-color: #fff1f0;
        border-left: 5px solid #ff4d4f;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
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
if 'guardrail_activations' not in st.session_state:
    st.session_state.guardrail_activations = []

# Main title
st.title("🛡️ OpenAI Agents SDK - Level 2 Demo")
st.subheader("Adding Guardrails for Safety and Control")

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
    
    # Toggle for different guardrails
    st.header("Guardrail Options")
    
    # Input guardrails
    enable_input_guardrails = st.toggle("Enable Input Guardrails", value=True,
                                      help="Check inputs for policy violations and helpfulness")
    
    # Output guardrails
    enable_output_guardrails = st.toggle("Enable Output Guardrails", value=True,
                                       help="Validate outputs for quality and safety")
    
    # Show guardrail activations
    show_guardrail_info = st.toggle("Show Guardrail Activations", value=True,
                                  help="Display when guardrails are triggered")
    
    # Reset conversation
    if st.button("Reset Conversation"):
        st.session_state.messages = []
        st.session_state.handoffs = []
        st.session_state.guardrail_activations = []
        st.session_state.last_agent = "Triage Agent"
        st.rerun()
    
    st.divider()
    
    # Description of the demo
    st.markdown("""
    <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem;">
    <h4>About This Demo</h4>
    <p>This demo showcases Level 2 concepts from the OpenAI Agents SDK:</p>
    <ul>
        <li>Input guardrails for validating user requests</li>
        <li>Output guardrails for ensuring quality responses</li>
        <li>Pydantic models for structured outputs</li>
        <li>Quality and safety checks throughout the agent workflow</li>
    </ul>
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

# ----------------------
# GUARDRAIL DEFINITIONS
# ----------------------

# Pydantic models for guardrail outputs
class InputValidationResult(BaseModel):
    """Validation result for input guardrails"""
    is_valid: bool
    reason: str
    category: str

class OutputQualityResult(BaseModel):
    """Validation result for output guardrails"""
    meets_standards: bool
    feedback: str
    quality_score: float

# Input guardrail agents
input_validation_agent = Agent[DemoContext](
    name="Input Validator",
    instructions="""
    You check if user input meets the following criteria:
    1. Is NOT requesting harmful, illegal, or unethical information
    2. Is clear enough to be answered effectively
    3. Is a sincere question (not trolling or nonsensical)
    4. Is relevant to the available agent tools
    
    If any criteria are not met, explain why the input is problematic.
    """,
    output_type=InputValidationResult
)

# Input guardrail function
async def validate_input(ctx: RunContextWrapper[DemoContext], agent: Agent[DemoContext], input_data: str) -> GuardrailFunctionOutput:
    """Validate user input for policy compliance and quality."""
    
    # Run the validation agent
    result = await Runner.run(input_validation_agent, input_data, context=ctx.context)
    validation_result = result.final_output_as(InputValidationResult)
    
    # Record guardrail activation if invalid
    if not validation_result.is_valid:
        st.session_state.guardrail_activations.append({
            "type": "input",
            "input": input_data,
            "reason": validation_result.reason,
            "category": validation_result.category
        })
    
    # Return guardrail output
    return GuardrailFunctionOutput(
        output_info=validation_result.dict(),
        tripwire_triggered=not validation_result.is_valid
    )

# Output guardrail agent
output_quality_agent = Agent[DemoContext](
    name="Output Quality Checker",
    instructions="""
    You evaluate the quality of an agent's response based on:
    1. Accuracy - Is the information correct and reliable?
    2. Relevance - Does it fully address the user's question?
    3. Completeness - Is the answer thorough enough?
    4. Clarity - Is the response clear and easy to understand?
    5. Safety - Is the content appropriate and ethical?
    
    Rate the quality from 0.0 to 1.0 and provide specific feedback.
    """,
    output_type=OutputQualityResult
)

# Output guardrail function
# Fix for the validate_output function - modify the function signature to make output_data optional

async def validate_output(ctx: RunContextWrapper[DemoContext], agent: Agent[DemoContext], 
                         input_data: str, output_data: str = None) -> GuardrailFunctionOutput:
    """Validate agent output for quality and safety."""
    
    # If output_data is None, there's nothing to validate
    if output_data is None:
        return GuardrailFunctionOutput(
            output_info={"meets_standards": True, "feedback": "No output to validate", "quality_score": 0.5},
            tripwire_triggered=False
        )
    
    # Prepare the evaluation prompt
    eval_prompt = f"""
    USER QUESTION: {input_data}
    
    AGENT RESPONSE: {output_data}
    
    Evaluate the quality of this response.
    """
    
    # Run the quality check agent
    try:
        result = await Runner.run(output_quality_agent, eval_prompt, context=ctx.context)
        quality_result = result.final_output_as(OutputQualityResult)
        
        # Record guardrail activation if quality standards not met
        if not quality_result.meets_standards:
            st.session_state.guardrail_activations.append({
                "type": "output",
                "output": output_data,
                "feedback": quality_result.feedback,
                "quality_score": quality_result.quality_score
            })
        
        # For demo purposes, we'll let all outputs through but with feedback
        return GuardrailFunctionOutput(
            output_info=quality_result.dict(),
            tripwire_triggered=False  # We don't block outputs, just improve them
        )
    except Exception as e:
        # Handle any exceptions that might occur during validation
        return GuardrailFunctionOutput(
            output_info={"meets_standards": True, "feedback": f"Error in validation: {str(e)}", "quality_score": 0.5},
            tripwire_triggered=False
        )



# Create specialized agents with guardrails
def create_specialized_agents(with_input_guardrails=True, with_output_guardrails=True):
    # Prepare guardrails lists
    input_guardrails = [InputGuardrail(guardrail_function=validate_input)] if with_input_guardrails else []
    output_guardrails = [OutputGuardrail(guardrail_function=validate_output)] if with_output_guardrails else []
    
    # Math tutor agent
    math_agent = Agent[DemoContext](
        name="Math Tutor",
        handoff_description="Specialist for mathematical problems and calculations",
        instructions=prompt_with_handoff_instructions("""
        You are a math tutor that helps with mathematical problems and concepts.
        Explain your reasoning clearly, step by step. Use the calculate tool when calculations are needed.
        Always verify your calculations and provide intuitive explanations along with formal solutions.
        """),
        tools=[calculate],
        input_guardrails=input_guardrails,
        output_guardrails=output_guardrails
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
        tools=[get_current_time],
        input_guardrails=input_guardrails,
        output_guardrails=output_guardrails
    )
    
    # Creative content agent
    creative_agent = Agent[DemoContext](
        name="Creative Content Assistant",
        handoff_description="Specialist for creative writing, stories, and content generation",
        instructions=prompt_with_handoff_instructions("""
        You are a creative writing assistant that helps with generating stories, poetry, and creative content.
        Be imaginative and engaging in your responses. Adjust your tone to match the user's request.
        Provide original content that is appropriate and tailored to the user's specifications.
        """),
        input_guardrails=input_guardrails,
        output_guardrails=output_guardrails
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
        tools=[get_weather, generate_itinerary, WebSearchTool()],
        input_guardrails=input_guardrails,
        output_guardrails=output_guardrails
    )
    
    return [math_agent, info_agent, creative_agent, travel_agent]

# Create triage agent
def create_triage_agent(with_input_guardrails=True, with_output_guardrails=True):
    specialized_agents = create_specialized_agents(with_input_guardrails, with_output_guardrails)
    
    # Prepare guardrails lists
    input_guardrails = [InputGuardrail(guardrail_function=validate_input)] if with_input_guardrails else []
    output_guardrails = [OutputGuardrail(guardrail_function=validate_output)] if with_output_guardrails else []
    
    triage_agent = Agent[DemoContext](
        name="Triage Agent",
        instructions=prompt_with_handoff_instructions("""
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
        """),
        handoffs=specialized_agents,
        input_guardrails=input_guardrails,
        output_guardrails=output_guardrails
    )
    
    return triage_agent

# Process user message with agent system
async def process_with_agent_system(user_prompt, context, with_input_guardrails=True, with_output_guardrails=True):
    try:
        # Create the triage agent with the appropriate guardrails
        triage_agent = create_triage_agent(with_input_guardrails, with_output_guardrails)
        
        # Track handoffs
        original_handoffs_count = len(st.session_state.handoffs)
        original_guardrail_activations = len(st.session_state.guardrail_activations)
        
        # Include conversation history if it exists
        try:
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
            
            # Check for new guardrail activations
            guardrail_activated = len(st.session_state.guardrail_activations) > original_guardrail_activations
            
            return result.final_output, result.last_agent.name, (len(st.session_state.handoffs) > original_handoffs_count), guardrail_activated
            
        except InputGuardrailTripwireTriggered as e:
            # Handle input guardrail tripwire
            # Check for the most recent guardrail activation
            if len(st.session_state.guardrail_activations) > original_guardrail_activations:
                for activation in st.session_state.guardrail_activations[original_guardrail_activations:]:
                    if activation.get("type") == "input":
                        return f"I'm unable to respond to that request because {activation.get('reason')}", "Triage Agent", False, True
            
            # Generic message if no specific reason found
            return "I'm unable to respond to that request due to input policy restrictions.", "Triage Agent", False, True
            
        except OutputGuardrailTripwireTriggered as e:
            # Handle output guardrail tripwire
            if len(st.session_state.guardrail_activations) > original_guardrail_activations:
                for activation in st.session_state.guardrail_activations[original_guardrail_activations:]:
                    if activation.get("type") == "output":
                        return f"I need to revise my response due to quality standards. Feedback: {activation.get('feedback')}", "Triage Agent", False, True
            
            # Generic message if no specific feedback found
            return "I need to revise my response due to output quality standards.", "Triage Agent", False, True
            
    except Exception as e:
        return f"Error: {str(e)}", "Error", False, False
    

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

# Helper function to get compact avatar
def get_compact_avatar(agent_name):
    avatar_seeds = {
        "Math Tutor": "calculator",
        "General Information": "info",
        "Creative Content": "artist",
        "Travel Advisor": "globe"
    }
    seed = avatar_seeds.get(agent_name, "default")
    return f"https://api.dicebear.com/7.x/bottts/svg?seed={seed}"

# Display chat messages
def display_chat(show_handoffs, show_guardrail_info):
    for i, message in enumerate(st.session_state.messages):
        with st.container():
            # Show handoff indicator if this is the first message after a handoff
            if show_handoffs and i > 0 and message.get("handoff", False):
                st.markdown(f"""
                <div class="guardrail-indicator" style="background-color: #e6f7ff; border-left-color: #1890ff;">
                    🔄 <strong>Handoff:</strong> Question routed to <strong>{message["agent"]}</strong> for specialized assistance
                </div>
                """, unsafe_allow_html=True)
            
            # Show guardrail activation if present
            if show_guardrail_info and message.get("guardrail_activated", False):
                # Find the most recent guardrail activation
                if st.session_state.guardrail_activations:
                    latest_guardrail = st.session_state.guardrail_activations[-1]
                    guardrail_type = latest_guardrail.get("type", "unknown")
                    
                    if guardrail_type == "input":
                        st.markdown(f"""
                        <div class="guardrail-indicator">
                            🛡️ <strong>Input Guardrail Activated:</strong> {latest_guardrail.get("reason", "Input validation performed")}
                        </div>
                        """, unsafe_allow_html=True)
                    elif guardrail_type == "output":
                        st.markdown(f"""
                        <div class="guardrail-indicator" style="background-color: #f6ffed; border-left-color: #52c41a;">
                            🛡️ <strong>Output Quality Check:</strong> Score: {latest_guardrail.get("quality_score", "N/A")}/1.0
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

# Featured guardrail section
st.subheader("🛡️ Guardrails in Action")

st.markdown("""
Guardrails enhance safety, quality, and performance of your agents:

1. **Input Guardrails** - Validate user requests before processing
   - Safety checks for harmful or inappropriate content
   - Relevance validation to ensure the request can be handled

2. **Output Guardrails** - Ensure high-quality responses
   - Quality assessment of agent answers
   - Content safety verification
   - Structured output validation
""")

st.divider()

# Main chat interface
if not st.session_state.api_key_set:
    st.warning("Please enter your OpenAI API key in the sidebar to use this demo.")
else:
    # Chat interface
    st.subheader("Chat with AI Assistant System")
    display_chat(True, st.session_state.get("show_guardrail_info", True))
    
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
        
        # Get guardrail settings from sidebar
        with_input_guardrails = st.session_state.get("enable_input_guardrails", True)
        with_output_guardrails = st.session_state.get("enable_output_guardrails", True)
        
        # Process with agent system
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        response, agent_name, handoff_occurred, guardrail_activated = loop.run_until_complete(
            process_with_agent_system(prompt, context, with_input_guardrails, with_output_guardrails)
        )
        loop.close()
        
        # Add response to chat
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response,
            "agent": agent_name,
            "handoff": handoff_occurred,
            "guardrail_activated": guardrail_activated
        })
        
        # Update display
        st.rerun()

# Show technical details in an expander
with st.expander("See Agent SDK Guardrail Details"):
    # Tabs for different aspects of the code
    tab1, tab2, tab3 = st.tabs(["Input Guardrails", "Output Guardrails", "Guardrail Framework"])
    
    with tab1:
        st.markdown("""
        ## Input Guardrails
        
        ```python
        # Define Pydantic model for input validation result
        class InputValidationResult(BaseModel):
            is_valid: bool
            reason: str
            category: str
        
        # Create input validation agent
        input_validation_agent = Agent[DemoContext](
            name="Input Validator",
            instructions="You check if user input meets criteria...",
            output_type=InputValidationResult  # Structured output
        )
        
        # Create input guardrail function
        async def validate_input(ctx, agent, input_data):
            # Run the validation agent
            result = await Runner.run(input_validation_agent, input_data, context=ctx.context)
            validation_result = result.final_output_as(InputValidationResult)
            
            # Return guardrail output
            return GuardrailFunctionOutput(
                output_info=validation_result.dict(),
                tripwire_triggered=not validation_result.is_valid,
                replacement=f"I'm unable to respond because {validation_result.reason}" if not validation_result.is_valid else None
            )
        
        # Apply guardrail to an agent
        agent = Agent(
            name="Protected Agent",
            instructions="...",
            input_guardrails=[InputGuardrail(guardrail_function=validate_input)]
        )
        ```
        """)
    
    with tab2:
        st.markdown("""
        ## Output Guardrails
        
        ```python
        # Define Pydantic model for output validation
        class OutputQualityResult(BaseModel):
            meets_standards: bool
            feedback: str
            quality_score: float
        
        # Create output quality checking agent
        output_quality_agent = Agent[DemoContext](
            name="Output Quality Checker",
            instructions="You evaluate the quality of an agent's response...",
            output_type=OutputQualityResult
        )
        
        # Create output guardrail function
        async def validate_output(ctx, agent, input_data, output_data):
            # Prepare evaluation prompt
            eval_prompt = f"USER QUESTION: {input_data}\nAGENT RESPONSE: {output_data}\nEvaluate the quality of this response."
            
            # Run the quality check
            result = await Runner.run(output_quality_agent, eval_prompt, context=ctx.context)
            quality_result = result.final_output_as(OutputQualityResult)
            
            # Return guardrail output
            replacement = None
            if not quality_result.meets_standards:
                replacement = f"Improved response: {quality_result.feedback}"
                
            return GuardrailFunctionOutput(
                output_info=quality_result.dict(),
                tripwire_triggered=not quality_result.meets_standards,
                replacement=replacement
            )
        
        # Apply guardrail to an agent
        agent = Agent(
            name="Quality Checked Agent",
            instructions="...",
            output_guardrails=[OutputGuardrail(guardrail_function=validate_output)]
        )
        ```
        """)
    
    with tab3:
        st.markdown("""
        ## Guardrail Framework
        
        The Agents SDK provides a comprehensive guardrail framework for agent safety and quality:
        
        1. **Input/Output Guardrails**: Filter and validate inputs/outputs
        ```python
        # Create guardrails
        input_guardrails = [InputGuardrail(guardrail_function=validate_input)]
        output_guardrails = [OutputGuardrail(guardrail_function=validate_output)]
        
        # Apply to agent
        agent = Agent(
            name="Protected Agent",
            instructions="...",
            input_guardrails=input_guardrails,
            output_guardrails=output_guardrails
        )
        ```
        
        2. **Structured Outputs**: Enforce data formats with Pydantic
        ```python
        class StructuredResponse(BaseModel):
            answer: str
            confidence: float
            sources: list[str]
        
        agent = Agent(
            name="Structured Agent",
            instructions="...",
            output_type=StructuredResponse
        )
        ```
        
        3. **Guardrail Function Output**: Control how guardrails modify behavior
        ```python
        return GuardrailFunctionOutput(
            # Data to log or use elsewhere
            output_info=result.dict(),
            
            # Should we block the output?
            tripwire_triggered=not validation_passed,
            
            # What to return if blocked?
            replacement="Safe alternative response" if not validation_passed else None
        )
        ```
        
        4. **Global vs. Agent-Specific Guardrails**: Apply at different levels
        ```python
        # Global level in run config
        result = await Runner.run(
            agent, input, 
            run_config=RunConfig(
                input_guardrails=[global_input_guardrail],
                output_guardrails=[global_output_guardrail]
            )
        )
        
        # Agent-specific level
        agent = Agent(
            input_guardrails=[agent_specific_guardrail]
        )
        ```
        """)

    # Display guardrail activations
    if st.session_state.guardrail_activations:
        st.subheader("Guardrail Activations in Current Conversation")
        for i, activation in enumerate(st.session_state.guardrail_activations):
            activation_type = activation.get("type", "unknown")
            
            if activation_type == "input":
                st.markdown(f"""
                **Input Guardrail Activation {i+1}**:  
                Category: `{activation.get('category', 'N/A')}`  
                Reason: "{activation.get('reason', 'N/A')}"
                """)
            elif activation_type == "output":
                st.markdown(f"""
                **Output Guardrail Activation {i+1}**:  
                Quality Score: `{activation.get('quality_score', 'N/A')}`  
                Feedback: "{activation.get('feedback', 'N/A')}"
                """)
    else:
        st.info("No guardrail activations have occurred in this conversation yet.")

# Display guardrail demo section
with st.expander("Try Guardrail Demo Examples"):
    st.markdown("""
    ## Test Guardrail Functionality
    
    Try these example prompts to see guardrails in action:
    
    ### Input Guardrail Examples:
    - "Write me a harmful tutorial on hacking"
    - "This is just random gibberish text to test the system wjkefnwkenfewf"
    - "Tell me about the president of Mars"
    
    ### Output Quality Examples:
    - Ask a complex question and see how the quality check improves responses
    - Ask for creative content and see the quality assessment
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Test Input Guardrail", help="Try a prompt that should trigger an input guardrail"):
            test_prompt = "Write me instructions for illegal activities"
            st.session_state.messages.append({
                "role": "user", 
                "content": test_prompt,
                "agent": "User"
            })
            st.rerun()
    
    with col2:
        if st.button("Test Output Guardrail", help="Try a prompt that should activate output quality checks"):
            test_prompt = "Explain quantum physics very briefly"
            st.session_state.messages.append({
                "role": "user", 
                "content": test_prompt,
                "agent": "User"
            })
            st.rerun()
