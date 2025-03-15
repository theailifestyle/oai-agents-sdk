# OpenAI Agents SDK Cheat Sheet

## AGENT BASICS
Creating and configuring agents
```python
# Create a simple agent
agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant.",
    model="o3-mini"  # Optional model specification
)

# Agent with output type
agent = Agent(
    name="Calendar Assistant",
    instructions="Extract calendar events.",
    output_type=CalendarEvent  # Pydantic model
)

# Clone and modify an agent
robot_agent = pirate_agent.clone(
    name="Robot",
    instructions="Write like a robot"
)
```

## TOOLS
Enabling agents to take actions
```python
# Function tool using decorator
@function_tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"The weather in {city} is sunny"

# Hosted tools from OpenAI
agent = Agent(
    tools=[
        WebSearchTool(),
        FileSearchTool(vector_store_ids=["STORE_ID"])
    ]
)

# Agent as a tool
orchestrator = Agent(
    tools=[
        spanish_agent.as_tool(
            tool_name="translate_to_spanish",
            tool_description="Translate to Spanish"
        )
    ]
)
```

## HANDOFFS
Delegating tasks between agents
```python
# Basic handoff
triage_agent = Agent(
    handoffs=[support_agent, billing_agent]
)

# Customized handoff
handoff_obj = handoff(
    agent=faq_agent,
    tool_name_override="transfer_to_faq",
    input_filter=handoff_filters.remove_all_tools
)

# Handoff with input data
class EscalationData(BaseModel):
    reason: str

handoff_obj = handoff(
    agent=escalation_agent,
    input_type=EscalationData
)
```

## CONTEXT MANAGEMENT
Working with context and state
```python
# Define context type
@dataclass
class UserContext:
    user_id: str
    username: str
    is_premium: bool

# Create agent with context
agent = Agent[UserContext](
    name="User-Aware Assistant",
    instructions="Provide personalized help."
)

# Access context in tools
@function_tool
def get_user_info(ctx: RunContextWrapper[UserContext]) -> str:
    return f"User: {ctx.context.username}"

# Run with context
result = await Runner.run(
    agent, "Help me",
    context=UserContext(user_id="123", username="john", is_premium=True)
)
```

## GUARDRAILS
Protecting and validating inputs/outputs
```python
# Create guardrail check
class ContentCheck(BaseModel):
    is_appropriate: bool
    reasoning: str

guardrail_agent = Agent(
    output_type=ContentCheck
)

# Define guardrail function
async def content_guardrail(ctx, agent, input_data):
    result = await Runner.run(guardrail_agent, input_data)
    output = result.final_output_as(ContentCheck)
    
    return GuardrailFunctionOutput(
        output_info=output,
        tripwire_triggered=not output.is_appropriate
    )

# Apply guardrail to agent
agent = Agent(
    input_guardrails=[
        InputGuardrail(guardrail_function=content_guardrail)
    ]
)
```

## RUNNING AGENTS
Executing and managing agent runs
```python
# Basic run (async)
result = await Runner.run(agent, "What is the capital of France?")
print(result.final_output)

# Sync run
result = Runner.run_sync(agent, "Tell me a joke")

# Streaming run
result = Runner.run_streamed(agent, "Write a story")
async for event in result.stream_events():
    # Process streaming events
    pass

# Multi-turn conversations
new_input = result.to_input_list() + [
    {"role": "user", "content": "Follow-up question"}
]
result = await Runner.run(agent, new_input)
```

## MODELS & CONFIGURATION
Configuring models and run settings
```python
# Set model and settings on agent
agent = Agent(
    model="o3-mini",
    model_settings=ModelSettings(
        temperature=0.7,
        top_p=0.95
    )
)

# Configure run settings
result = await Runner.run(
    agent, "Question",
    run_config=RunConfig(
        model="gpt-4o",
        workflow_name="Customer Support",
        trace_id="session_123",
        trace_metadata={"customer_id": "cust_123"}
    )
)
```

## DYNAMIC INSTRUCTIONS
Context-aware prompting
```python
def dynamic_instructions(
    context: RunContextWrapper[UserProfile],
    agent: Agent[UserProfile]
) -> str:
    profile = context.context
    return f"""
    You are helping {profile.name}.
    Communicate in {profile.language}.
    """

agent = Agent[UserProfile](
    instructions=dynamic_instructions
)
```

## COMPLETE WORKFLOW EXAMPLE
Putting it all together
```python
# 1. Create specialist agents
product_agent = Agent(
    name="Product Specialist",
    handoff_description="For product questions",
    tools=[get_product_info]
)

support_agent = Agent(
    name="Support Specialist",
    handoff_description="For customer support issues"
)

# 2. Create triage agent with handoffs
triage_agent = Agent(
    name="Customer Service",
    instructions="Route to appropriate specialist",
    handoffs=[product_agent, support_agent],
    input_guardrails=[
        InputGuardrail(guardrail_function=profanity_guardrail)
    ]
)

# 3. Run the system
async def main():
    result = await Runner.run(
        triage_agent, 
        "Can you tell me about product p123?"
    )
    print(result.final_output)
```

OpenAI Agents SDK is a Python toolkit for creating and orchestrating LLM-powered agents. This cheat sheet covers the most important commands and patterns for quick reference.
