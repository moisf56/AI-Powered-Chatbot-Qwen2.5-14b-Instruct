"""
Flask Backend for Human-in-the-Loop Chatbot (with memory and web search capabilities)
Connects your LangGraph chatbot with the React frontend
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import uuid
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import threading
import time

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables for interrupt handling
_INTERRUPTED_STATES = {}
_GRAPH_INSTANCES = {}

class InterruptException(Exception):
    def __init__(self, interrupt_id, data):
        self.interrupt_id = interrupt_id
        self.data = data
        super().__init__(f"Execution interrupted: {data}")

def interrupt(data):
    """Simple interrupt function that stores data and raises exception"""
    interrupt_id = str(uuid.uuid4())
    _INTERRUPTED_STATES[interrupt_id] = data
    raise InterruptException(interrupt_id, data)

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
    # This will interrupt execution
    human_response = interrupt({"query": query, "type": "human_assistance"})
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

# Create and compile graph
def create_graph():
    """Create a new graph instance"""
    graph_builder = StateGraph(State)
    
    # Add nodes
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_node("tools", execute_tools)
    
    # Add edges
    graph_builder.add_conditional_edges("chatbot", should_continue, {"tools": "tools", "end": END})
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge(START, "chatbot")
    
    return graph_builder.compile(checkpointer=memory)

# Initialize global graph
graph = create_graph()

@app.route('/api/test-connection', methods=['POST'])
def test_connection():
    """Test LLM connection"""
    try:
        response = llm.invoke([HumanMessage(content="Hello")])
        return jsonify({
            "status": "success",
            "message": "Connection successful",
            "response": response.content[:50] + "..."
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Connection failed: {str(e)}"
        }), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    try:
        data = request.json
        user_input = data.get('message', '')
        thread_id = data.get('thread_id', '1')
        
        if not user_input:
            return jsonify({"error": "No message provided"}), 400
        
        config = {"configurable": {"thread_id": thread_id}}
        
        # Initial state
        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "interrupted": False,
            "interrupt_data": {}
        }
        
        try:
            events = list(graph.stream(initial_state, config, stream_mode="values"))
            
            if events:
                final_event = events[-1]
                
                # Check if interrupted
                if final_event.get("interrupted", False):
                    interrupt_data = final_event.get("interrupt_data", {})
                    return jsonify({
                        "status": "INTERRUPTED",
                        "interrupt_data": interrupt_data,
                        "message": "AI needs human assistance"
                    })
                
                # Get final AI response
                messages = final_event.get("messages", [])
                tools_used = []
                
                for msg in messages:
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            tools_used.append(tool_call["name"])
                
                # Find the last AI message
                for msg in reversed(messages):
                    if isinstance(msg, AIMessage) and msg.content:
                        if not (hasattr(msg, 'tool_calls') and msg.tool_calls):
                            return jsonify({
                                "status": "COMPLETED",
                                "response": msg.content,
                                "tools_used": list(set(tools_used))
                            })
            
            return jsonify({
                "status": "COMPLETED",
                "response": "I couldn't generate a proper response. Please try again."
            })
            
        except Exception as e:
            return jsonify({
                "status": "ERROR",
                "response": f"Error processing request: {str(e)}"
            }), 500
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/resume', methods=['POST'])
def resume_execution():
    """Resume interrupted execution"""
    try:
        data = request.json
        human_response = data.get('response', '')
        thread_id = data.get('thread_id', '1')
        
        if not human_response:
            return jsonify({"error": "No response provided"}), 400
        
        config = {"configurable": {"thread_id": thread_id}}
        
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
                            return jsonify({
                                "status": "COMPLETED",
                                "response": msg.content
                            })
            
            return jsonify({
                "status": "COMPLETED",
                "response": "Execution resumed successfully."
            })
            
        except Exception as e:
            return jsonify({
                "status": "ERROR",
                "response": f"Error resuming execution: {str(e)}"
            }), 500
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "message": "Chatbot API is running"
    })

@app.route('/', methods=['GET'])
def home():
    """Basic home page"""
    return """
    <h1>Human-in-the-Loop Chatbot API</h1>
    <p>Your chatbot backend is running!</p>
    <ul>
        <li><strong>POST /api/test-connection</strong> - Test LLM connection</li>
        <li><strong>POST /api/chat</strong> - Send chat message</li>
        <li><strong>POST /api/resume</strong> - Resume interrupted execution</li>
        <li><strong>GET /api/health</strong> - Health check</li>
    </ul>
    """

if __name__ == '__main__':
    print("🤖 Starting Human-in-the-Loop Chatbot API")
    print("=" * 45)
    
    # Test connection on startup
    try:
        test_response = llm.invoke([HumanMessage(content="Hello")])
        print("✅ LLM connection successful")
    except Exception as e:
        print(f"❌ LLM connection failed: {e}")
    
    print("🚀 Server starting...")
    app.run(host='0.0.0.0', port=5000, debug=True)