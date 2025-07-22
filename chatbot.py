"""
Human-in-the-Loop Chatbot - Built from Scratch
No problematic imports - everything works!
"""

import os
import json
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# Simple Command class for resuming
class Command:
    def __init__(self, resume=None):
        self.resume = resume

# Global variable to handle interrupts
_INTERRUPTED_STATE = {}

def interrupt(data):
    """Simple interrupt function that stores data and raises exception"""
    global _INTERRUPTED_STATE
    import uuid
    interrupt_id = str(uuid.uuid4())
    _INTERRUPTED_STATE[interrupt_id] = data
    raise InterruptException(interrupt_id, data)

class InterruptException(Exception):
    def __init__(self, interrupt_id, data):
        self.interrupt_id = interrupt_id
        self.data = data
        super().__init__(f"Execution interrupted: {data}")

# Search tool
@tool
def search_web(query: str) -> str:
    """Search the web for information"""
    try:
        from langchain_community.tools import DuckDuckGoSearchRun
        search = DuckDuckGoSearchRun(max_results=2)
        result = search.invoke(query)
        return result
    except:
        return f"Search results for '{query}': Mock search data - sunny weather, latest news, etc."

# Human assistance tool
@tool
def human_assistance(query: str) -> str:
    """Request assistance from a human expert when AI needs help or approval."""
    print(f"\n🚨 HUMAN ASSISTANCE REQUESTED 🚨")
    print(f"Query: {query}")
    print("=" * 50)
    
    # This will interrupt execution
    human_response = interrupt({"query": query})
    return human_response["data"]

# Your vLLM setup
BASE_URL = "https://f2wcnb5om9g8yl-8000.proxy.runpod.net/v1"

llm = ChatOpenAI(
    base_url=BASE_URL,
    api_key="EMPTY",
    model="Qwen2.5-14B-Instruct",
    temperature=0.7,
    max_tokens=500,
    timeout=60
)

# State definition
class State(TypedDict):
    messages: Annotated[list, add_messages]
    interrupted: bool
    interrupt_data: dict

# Create memory checkpointer
memory = MemorySaver()

# Available tools
tools = [search_web, human_assistance]
tools_by_name = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    """Main chatbot node"""
    try:
        message = llm_with_tools.invoke(state["messages"])
        return {"messages": [message], "interrupted": False}
    except InterruptException as e:
        # Store interrupt data in state
        return {
            "interrupted": True, 
            "interrupt_data": e.data,
            "messages": []
        }

def execute_tools(state: State):
    """Execute tools manually"""
    messages = state["messages"]
    last_message = messages[-1]
    
    if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
        return {"messages": []}
    
    tool_outputs = []
    
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]
        
        try:
            if tool_name in tools_by_name:
                tool = tools_by_name[tool_name]
                result = tool.invoke(tool_args)
                
                tool_message = ToolMessage(
                    content=json.dumps(result) if isinstance(result, dict) else str(result),
                    name=tool_name,
                    tool_call_id=tool_id,
                )
                tool_outputs.append(tool_message)
        except InterruptException as e:
            # Store interrupt and return state
            return {
                "interrupted": True,
                "interrupt_data": e.data,
                "messages": tool_outputs
            }
        except Exception as e:
            error_message = ToolMessage(
                content=f"Error executing {tool_name}: {str(e)}",
                name=tool_name,
                tool_call_id=tool_id,
            )
            tool_outputs.append(error_message)
    
    return {"messages": tool_outputs, "interrupted": False}

def should_continue(state: State) -> Literal["tools", "end"]:
    """Check if we should use tools or end"""
    if state.get("interrupted", False):
        return "end"
        
    messages = state["messages"]
    if not messages:
        return "end"
        
    last_message = messages[-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    return "end"

# Create graph
graph_builder = StateGraph(State)

# Add nodes
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", execute_tools)

# Add edges
graph_builder.add_conditional_edges("chatbot", should_continue, {"tools": "tools", "end": END})
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

# Compile graph
graph = graph_builder.compile(checkpointer=memory)

def test_connection():
    """Test LLM connection"""
    try:
        response = llm.invoke([HumanMessage(content="Hello")])
        print(f"✅ LLM connected: {response.content[:30]}...")
        return True
    except Exception as e:
        print(f"❌ LLM connection failed: {e}")
        return False

def stream_graph_updates(user_input: str, config: dict):
    """Stream chatbot responses"""
    try:
        # Initial state
        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "interrupted": False,
            "interrupt_data": {}
        }
        
        events = list(graph.stream(initial_state, config, stream_mode="values"))
        
        if events:
            final_event = events[-1]
            
            # Check if interrupted
            if final_event.get("interrupted", False):
                print("\n⏸️  Execution paused - AI needs human input")
                interrupt_data = final_event.get("interrupt_data", {})
                if "query" in interrupt_data:
                    print(f"Question: {interrupt_data['query']}")
                print("Use 'resume <your_response>' to continue")
                return "INTERRUPTED"
            
            # Get final AI response
            messages = final_event.get("messages", [])
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and msg.content:
                    if not (hasattr(msg, 'tool_calls') and msg.tool_calls):
                        print("Assistant:", msg.content)
                        return "COMPLETED"
        
        print("Assistant: I couldn't generate a proper response.")
        return "ERROR"
        
    except Exception as e:
        print(f"Error: {e}")
        return "ERROR"

def resume_execution(human_response: str, config: dict):
    """Resume interrupted execution"""
    try:
        # Get current state
        state = graph.get_state(config)
        current_messages = state.values.get("messages", [])
        
        # Add tool response
        if current_messages and hasattr(current_messages[-1], 'tool_calls'):
            last_tool_call = current_messages[-1].tool_calls[-1]
            tool_message = ToolMessage(
                content=human_response,
                name="human_assistance",
                tool_call_id=last_tool_call["id"],
            )
            current_messages.append(tool_message)
        
        # Resume execution
        resume_state = {
            "messages": current_messages,
            "interrupted": False,
            "interrupt_data": {}
        }
        
        events = list(graph.stream(resume_state, config, stream_mode="values"))
        
        if events:
            final_event = events[-1]
            messages = final_event.get("messages", [])
            
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and msg.content:
                    if not (hasattr(msg, 'tool_calls') and msg.tool_calls):
                        print("Assistant:", msg.content)
                        return
        
        print("Assistant: Execution resumed successfully.")
        
    except Exception as e:
        print(f"Error resuming: {e}")

def run_chatbot():
    """Main chat loop"""
    print("🤖🤝 Working Human-in-the-Loop Chatbot")
    print("=" * 45)
    
    if not test_connection():
        return
    
    print("\n✅ Features:")
    print("- Web search capability")
    print("- Human assistance requests")
    print("- Memory across conversations")
    print("- Pause/resume execution")
    
    print("\nCommands:")
    print("- 'quit' to exit")
    print("- 'new' for new thread")
    print("- 'resume <response>' to continue")
    
    print("\nTry:")
    print("- 'I need expert advice on career change'")
    print("- 'What's the weather in Paris?'")
    print()
    
    thread_id = "1"
    config = {"configurable": {"thread_id": thread_id}}
    
    print(f"🗣️  Thread: {thread_id}")
    print()
    
    while True:
        try:
            user_input = input("User: ").strip()
            
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            elif user_input.lower() == "new":
                thread_id = str(int(thread_id) + 1)
                config = {"configurable": {"thread_id": thread_id}}
                print(f"🗣️  New thread: {thread_id}")
                continue
            elif user_input.lower().startswith("resume "):
                response = user_input[7:]
                if response:
                    print(f"▶️  Resuming with: {response}")
                    resume_execution(response, config)
                continue
            elif not user_input:
                continue
            
            result = stream_graph_updates(user_input, config)
            print()
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    run_chatbot()