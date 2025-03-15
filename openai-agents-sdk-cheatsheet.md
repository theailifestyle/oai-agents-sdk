# OpenAI Agents SDK Cheatsheet

## Installation & Setup

```bash
# Create a project and virtual environment
mkdir my_project
cd my_project
python -m venv .venv

# Activate the virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the Agents SDK
pip install openai-agents

# Set OpenAI API key
export OPENAI_API_KEY=sk-...  # On Windows: set OPENAI_API_KEY=sk-...
```

## Agent Basics

### Creating a Simple Agent

```python
from agents import Agent, Runner
import asyncio

# Define a basic agent
agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant that provides clear, concise answers."
)

# Run the agent
async def main():
    result = await Runner.run(agent, "What is the capital of France?")
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())
```

### Agent with Tools

```python
from agents import Agent, function_tool

# Define a tool using the decorator
@function_tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.
    
    Args:
        city: The name of the city to get weather for.
    """
    # In a real app, call a weather API here
    return f"The weather in {city} is sunny and 75°F"

# Create agent with the tool
agent = Agent(
    name="Weather Assistant",
    instructions="Help users with weather-related queries.",
    tools=[get_weather]
)
```

### Agent with Output Type

```python
from pydantic import BaseModel
from agents import Agent

class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]

agent = Agent(
    name="Calendar Assistant",
    instructions="Extract calendar events from user messages.",
    output_type=CalendarEvent
)
```

## Agent Orchestration

### Creating Agents with Handoffs

```python
from agents import Agent

# Create specialist agents
history_agent = Agent(
    name="History Tutor",
    handoff_description="Specialist for historical questions",
    instructions="You provide assistance with historical queries. Explain important events and context clearly."
)

math_agent = Agent(
    name="Math Tutor",
    handoff_description="Specialist for math questions",
    instructions="You provide help with math problems. Explain your reasoning at each step."
)

# Create a triage agent with handoffs
triage_agent = Agent(
    name="Triage Agent",
    instructions="Determine which specialist to use based on the user's question",
    handoffs=[history_agent, math_agent]
)
```

### Customizing Handoffs

```python
from agents import Agent, handoff
from agents.extensions import handoff_filters

# Create a specialized agent
faq_agent = Agent(
    name="FAQ Agent",
    instructions="Answer frequently asked questions about our product."
)

# Create customized handoff
handoff_obj = handoff(
    agent=faq_agent,
    tool_name_override="transfer_to_faq_specialist",
    tool_description_override="Transfer to the FAQ specialist for product questions",
    input_filter=handoff_filters.remove_all_tools
)

# Use the handoff in another agent
main_agent = Agent(
    name="Main Agent",
    instructions="Help users with their queries. Transfer to specialists as needed.",
    handoffs=[handoff_obj]
)
```

## Tools

### Function Tools

```python
from agents import Agent, function_tool
from typing import Dict, List

@function_tool
def search_database(query: str) -> List[Dict]:
    """Search the database for information.
    
    Args:
        query: The search query string.
    """
    # Implement database search here
    return [{"title": "Sample result", "content": "Sample content"}]

@function_tool
def calculate_total(items: List[Dict[str, float]]) -> float:
    """Calculate the total price of multiple items.
    
    Args:
        items: List of items with their prices.
    """
    return sum(item["price"] for item in items)

agent = Agent(
    name="Database Assistant",
    instructions="Help users search the database and calculate totals.",
    tools=[search_database, calculate_total]
)
```

### OpenAI Hosted Tools

```python
from agents import Agent, WebSearchTool, FileSearchTool

agent = Agent(
    name="Research Assistant",
    instructions="Help users find information online and from our knowledge base.",
    tools=[
        WebSearchTool(),
        FileSearchTool(
            max_num_results=3,
            vector_store_ids=["YOUR_VECTOR_STORE_ID"]
        )
    ]
)
```

### Agents as Tools

```python
from agents import Agent

translator_spanish = Agent(
    name="Spanish Translator",
    instructions="Translate text to Spanish accurately and naturally."
)

translator_french = Agent(
    name="French Translator",
    instructions="Translate text to French accurately and naturally."
)

orchestrator = Agent(
    name="Translation Hub",
    instructions="Help users translate text to different languages.",
    tools=[
        translator_spanish.as_tool(
            tool_name="translate_to_spanish",
            tool_description="Translate the user's text to Spanish"
        ),
        translator_french.as_tool(
            tool_name="translate_to_french",
            tool_description="Translate the user's text to French"
        )
    ]
)
```

## Guardrails

```python
from agents import Agent, InputGuardrail, GuardrailFunctionOutput, Runner
from pydantic import BaseModel

# Define output structure for guardrail
class HomeworkCheck(BaseModel):
    is_homework: bool
    reasoning: str

# Create guardrail agent
guardrail_agent = Agent(
    name="Guardrail Check",
    instructions="Check if the user is asking about homework.",
    output_type=HomeworkCheck
)

# Define guardrail function
async def homework_guardrail(ctx, agent, input_data):
    result = await Runner.run(guardrail_agent, input_data, context=ctx.context)
    final_output = result.final_output_as(HomeworkCheck)
    
    return GuardrailFunctionOutput(
        output_info=final_output,
        tripwire_triggered=not final_output.is_homework
    )

# Apply guardrail to agent
agent_with_guardrail = Agent(
    name="Homework Helper",
    instructions="Help with legitimate educational questions, not just giving homework answers.",
    input_guardrails=[
        InputGuardrail(guardrail_function=homework_guardrail)
    ]
)
```

## Context Management

```python
from dataclasses import dataclass
from agents import Agent, RunContextWrapper, function_tool, Runner

# Define context structure
@dataclass
class UserContext:
    user_id: str
    username: str
    is_premium: bool

# Create a tool that uses context
@function_tool
def get_user_info(ctx: RunContextWrapper[UserContext]) -> str:
    """Get information about the current user."""
    user = ctx.context
    return f"User: {user.username} (ID: {user.user_id}), Premium: {user.is_premium}"

# Create agent with context type
agent = Agent[UserContext](
    name="User-Aware Assistant",
    instructions="Provide personalized assistance based on the user's information.",
    tools=[get_user_info]
)

# Use the agent with context
async def main():
    user_context = UserContext(
        user_id="12345",
        username="john_doe",
        is_premium=True
    )
    
    result = await Runner.run(
        starting_agent=agent,
        input="Tell me about my account",
        context=user_context
    )
    print(result.final_output)
```

## Running Agents

### Basic Running

```python
from agents import Agent, Runner
import asyncio

async def main():
    agent = Agent(
        name="Assistant",
        instructions="You are a helpful assistant."
    )
    
    # Async run
    result = await Runner.run(agent, "Write a haiku about programming.")
    print(result.final_output)
    
    # For synchronous code, use run_sync instead
    # result = Runner.run_sync(agent, "Write a haiku about programming.")

if __name__ == "__main__":
    asyncio.run(main())
```

### Streaming

```python
from agents import Agent, Runner
import asyncio

async def main():
    agent = Agent(
        name="Storyteller",
        instructions="You create engaging short stories."
    )
    
    # Get streaming result
    result = Runner.run_streamed(
        agent, 
        input="Tell me a short story about a robot learning to paint."
    )
    
    # Print content as it's generated
    print("Story is being written...")
    async for event in result.stream_events():
        if event.type == "raw_response_event":
            if hasattr(event.data, "delta"):
                print(event.data.delta, end="", flush=True)
    
    print("\n\nStory complete!")

if __name__ == "__main__":
    asyncio.run(main())
```

### Multi-turn Conversations

```python
from agents import Agent, Runner
import asyncio

async def main():
    agent = Agent(
        name="Conversation Agent",
        instructions="You are a helpful assistant that maintains context throughout a conversation."
    )
    
    # First turn
    result = await Runner.run(agent, "What's the capital of Japan?")
    print(f"Agent: {result.final_output}")
    
    # Second turn (using previous context)
    new_input = result.to_input_list() + [{"role": "user", "content": "What about South Korea?"}]
    result = await Runner.run(agent, new_input)
    print(f"Agent: {result.final_output}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Model Configuration

```python
from agents import Agent, ModelSettings, Runner

# Configure model settings
agent = Agent(
    name="Assistant",
    instructions="You are a helpful, concise assistant.",
    model="o3-mini",  # Specify model name
    model_settings=ModelSettings(
        temperature=0.7,
        top_p=0.95,
        max_tokens=1000
    )
)

# Or override at runtime
async def main():
    result = await Runner.run(
        agent,
        "Explain quantum computing briefly",
        run_config=RunConfig(
            model="gpt-4o",
            model_settings=ModelSettings(temperature=0)
        )
    )
    print(result.final_output)
```

## Dynamic Instructions

```python
from agents import Agent, RunContextWrapper
from dataclasses import dataclass

@dataclass
class UserProfile:
    name: str
    language: str
    expertise_level: str

def dynamic_instructions(
    context: RunContextWrapper[UserProfile],
    agent: Agent[UserProfile]
) -> str:
    profile = context.context
    return f"""
    You are a helpful assistant for {profile.name}.
    Communicate in {profile.language}.
    Adjust explanations to a {profile.expertise_level} level of expertise.
    Be friendly, clear, and concise in your responses.
    """

agent = Agent[UserProfile](
    name="Adaptive Assistant",
    instructions=dynamic_instructions
)
```

## Error Handling

```python
from agents import Agent, function_tool, Runner
import asyncio

# Custom error handler for tools
def handle_tool_error(error, function_name, args):
    if isinstance(error, ValueError):
        return f"I couldn't process your request. The value provided was invalid: {str(error)}"
    return f"An unexpected error occurred: {str(error)}"

@function_tool(failure_error_function=handle_tool_error)
def divide_numbers(a: float, b: float) -> float:
    """Divide two numbers.
    
    Args:
        a: The dividend
        b: The divisor (must not be zero)
    """
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

agent = Agent(
    name="Calculator",
    instructions="Help users with mathematical calculations.",
    tools=[divide_numbers]
)

async def main():
    try:
        result = await Runner.run(agent, "What is 10 divided by 0?")
        print(result.final_output)
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Tracing

```python
from agents import Agent, Runner, trace, set_tracing_disabled

# Disable tracing globally
# set_tracing_disabled(True)

async def main():
    agent = Agent(
        name="Assistant",
        instructions="You are a helpful assistant."
    )
    
    # Add trace information for this run
    with trace(
        workflow_name="Customer Support",
        group_id="user_session_123",
        trace_metadata={"customer_id": "cust_123", "channel": "chat"}
    ):
        result = await Runner.run(agent, "I need help with my order")
        print(result.final_output)
    
    # View traces in the OpenAI Dashboard: https://platform.openai.com/traces
```

## Complete Example: Multi-Agent System

```python
from agents import Agent, Runner, function_tool, InputGuardrail, GuardrailFunctionOutput
from pydantic import BaseModel
import asyncio

# 1. Define tools
@function_tool
def get_product_info(product_id: str) -> dict:
    """Get information about a product.
    
    Args:
        product_id: The unique identifier for the product
    """
    # Mock database lookup
    products = {
        "p123": {"name": "Wireless Headphones", "price": 99.99, "in_stock": True},
        "p456": {"name": "Smart Watch", "price": 249.99, "in_stock": False}
    }
    return products.get(product_id, {"error": "Product not found"})

# 2. Define guardrail check
class ProfanityCheck(BaseModel):
    contains_profanity: bool
    reasoning: str

profanity_agent = Agent(
    name="Profanity Check",
    instructions="Check if the user's message contains profanity or inappropriate language.",
    output_type=ProfanityCheck
)

async def profanity_guardrail(ctx, agent, input_data):
    result = await Runner.run(profanity_agent, input_data, context=ctx.context)
    final_output = result.final_output_as(ProfanityCheck)
    
    return GuardrailFunctionOutput(
        output_info=final_output,
        tripwire_triggered=final_output.contains_profanity
    )

# 3. Create specialized agents
product_agent = Agent(
    name="Product Specialist",
    handoff_description="Specialist for product-related questions",
    instructions="You help customers with product information, features, and comparisons.",
    tools=[get_product_info]
)

support_agent = Agent(
    name="Support Specialist",
    handoff_description="Specialist for customer support issues",
    instructions="You help customers with order issues, returns, and technical support."
)

# 4. Create main triage agent
triage_agent = Agent(
    name="Customer Service",
    instructions="""
    You are the initial point of contact for customer inquiries.
    Analyze the customer's question and route to the appropriate specialist:
    - Route product questions to the Product Specialist
    - Route support issues to the Support Specialist
    - For simple questions, answer directly
    """,
    handoffs=[product_agent, support_agent],
    input_guardrails=[
        InputGuardrail(guardrail_function=profanity_guardrail)
    ]
)

# 5. Run the system
async def main():
    # Run with a product question
    result = await Runner.run(
        triage_agent, 
        "Can you tell me more about the wireless headphones with product ID p123?"
    )
    print("RESULT 1:\n", result.final_output)
    print("\n" + "-"*50 + "\n")
    
    # Run with a support question
    result = await Runner.run(
        triage_agent,
        "I ordered something 3 days ago but haven't received a shipping confirmation."
    )
    print("RESULT 2:\n", result.final_output)

if __name__ == "__main__":
    asyncio.run(main())
```
