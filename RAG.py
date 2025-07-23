"""
JSON Structured RAG Chatbot - Industry Standard Output
Features: Structured JSON responses with metadata, sources, confidence scores
"""

import os
import json
import uuid
from typing import Annotated, Literal, List, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from datetime import datetime
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.documents import Document

# Industry Standard JSON Response Schema
class SourceMetadata(BaseModel):
    """Source information with metadata"""
    type: Literal["knowledge_base", "web_search", "human_expert"] = Field(..., description="Source type")
    title: Optional[str] = Field(None, description="Document/page title")
    url: Optional[str] = Field(None, description="Source URL if available")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0-1")
    chunk_id: Optional[str] = Field(None, description="Document chunk identifier")
    relevance_score: Optional[float] = Field(None, description="Similarity score")

class ChatResponse(BaseModel):
    """Industry standard structured chat response"""
    chat_id: str = Field(..., description="Unique conversation identifier")
    message_id: str = Field(..., description="Unique message identifier") 
    timestamp: str = Field(..., description="ISO 8601 timestamp")
    response: str = Field(..., description="Natural language response")
    
    # Tool and source information
    tools_used: List[str] = Field(default=[], description="Tools invoked for this response")
    sources: List[SourceMetadata] = Field(default=[], description="Information sources used")
    
    # Metadata and context
    reasoning: Optional[str] = Field(None, description="AI reasoning process")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall response confidence")
    tokens_used: Optional[int] = Field(None, description="LLM tokens consumed")
    processing_time_ms: Optional[int] = Field(None, description="Response time in milliseconds")
    
    # Status and error handling
    status: Literal["success", "partial", "error", "interrupted"] = Field(..., description="Response status")
    error_message: Optional[str] = Field(None, description="Error details if status is error")
    
    # Human-in-the-loop
    requires_human_input: bool = Field(default=False, description="Whether human input is needed")
    human_query: Optional[str] = Field(None, description="Question for human expert")

# RAG tools with enhanced metadata
from langchain_community.document_loaders import WebBaseLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
import bs4

# Initialize components
try:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print("✅ Using HuggingFace embeddings")
except Exception as e:
    print(f"⚠️  Using fallback embeddings: {e}")
    from langchain_core.embeddings import DummyEmbeddings
    embeddings = DummyEmbeddings(size=384)

vector_store = InMemoryVectorStore(embeddings)

BASE_URL = "https://m6xhg6dhn6h64h-8000.proxy.runpod.net/v1"

# Configure LLM for structured output
llm = ChatOpenAI(
    base_url=BASE_URL,
    api_key="EMPTY",
    model="Qwen2.5-14B-Instruct",
    temperature=0.7,
    max_tokens=500,
    timeout=60
)

# Enhanced tools with metadata
@tool(response_format="content_and_artifact")
def retrieve_knowledge(query: str) -> tuple[str, dict]:
    """Retrieve information from knowledge base with metadata"""
    try:
        retrieved_docs = vector_store.similarity_search_with_score(query, k=3)
        
        if not retrieved_docs:
            return "No relevant information found", {"sources": [], "confidence": 0.0}
        
        sources = []
        content_parts = []
        
        for doc, score in retrieved_docs:
            # Calculate confidence from similarity score
            confidence = max(0.0, min(1.0, 1.0 - score))  # Convert distance to confidence
            
            source_meta = {
                "type": "knowledge_base",
                "title": doc.metadata.get("title", "Document"),
                "url": doc.metadata.get("source"),
                "confidence": round(confidence, 2),
                "chunk_id": doc.metadata.get("chunk_id", str(uuid.uuid4())[:8]),
                "relevance_score": round(score, 3)
            }
            sources.append(source_meta)
            
            content_parts.append(f"Content: {doc.page_content[:300]}...")
        
        combined_content = "\n\n".join(content_parts)
        overall_confidence = sum(s["confidence"] for s in sources) / len(sources)
        
        return combined_content, {
            "sources": sources,
            "confidence": round(overall_confidence, 2),
            "documents_found": len(sources)
        }
        
    except Exception as e:
        return f"Error: {str(e)}", {"sources": [], "confidence": 0.0, "error": str(e)}

@tool
def search_web(query: str) -> dict:
    """Web search with structured metadata"""
    try:
        from langchain_community.tools import DuckDuckGoSearchRun
        search = DuckDuckGoSearchRun(max_results=2)
        result = search.invoke(query)
        
        # Parse web results (simplified)
        source_meta = {
            "type": "web_search",
            "title": f"Web search: {query}",
            "url": None,  # DuckDuckGo doesn't return URLs directly
            "confidence": 0.8,  # Web search generally reliable
            "relevance_score": 0.8
        }
        
        return {
            "content": result,
            "sources": [source_meta],
            "confidence": 0.8
        }
    except Exception as e:
        return {
            "content": f"Mock web search for '{query}': Current information, trends, recent updates.",
            "sources": [{
                "type": "web_search",
                "title": f"Mock search: {query}",
                "confidence": 0.6
            }],
            "confidence": 0.6,
            "error": str(e)
        }

# Exception for human assistance
class InterruptException(Exception):
    def __init__(self, data):
        self.data = data
        super().__init__(f"Human assistance requested: {data}")

@tool
def human_assistance(query: str) -> dict:
    """Request human expert assistance"""
    print(f"\n🚨 HUMAN ASSISTANCE REQUESTED 🚨")
    print(f"Query: {query}")
    print("=" * 50)
    
    raise InterruptException({
        "type": "human_assistance", 
        "query": query,
        "requires_input": True
    })

# Enhanced state with JSON response tracking
class State(TypedDict):
    messages: Annotated[list, add_messages]
    interrupted: bool
    interrupt_data: dict
    chat_id: str
    processing_start: float
    sources_collected: List[dict]
    tools_used: List[str]
    confidence_scores: List[float]

memory = MemorySaver()
tools = [retrieve_knowledge, search_web, human_assistance]
tools_by_name = {tool.name: tool for tool in tools}

def create_structured_llm():
    """Create LLM with structured output capability"""
    return llm.with_structured_output(ChatResponse)

def chatbot(state: State):
    """Enhanced chatbot with structured output"""
    import time
    start_time = time.time()
    
    try:
        # System prompt for structured thinking
        system_prompt = """You are an intelligent assistant that provides structured, accurate responses.

Available tools:
1. retrieve_knowledge: Search knowledge base for relevant information
2. search_web: Search internet for current information  
3. human_assistance: Request expert help for complex decisions

Always:
- Choose appropriate tools based on query type
- Provide clear, helpful responses
- Include reasoning for your decisions
- Be honest about confidence levels

Response format: Natural language answer with supporting sources."""

        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        
        # Use regular LLM for tool selection, structured LLM for final response
        llm_with_tools = llm.bind_tools(tools)
        message = llm_with_tools.invoke(messages)
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return {
            "messages": [message], 
            "interrupted": False,
            "processing_start": start_time,
            "tools_used": state.get("tools_used", []),
            "sources_collected": state.get("sources_collected", []),
            "confidence_scores": state.get("confidence_scores", [])
        }
        
    except InterruptException as e:
        return {
            "interrupted": True, 
            "interrupt_data": e.data,
            "messages": [],
            "processing_start": start_time
        }

def execute_tools(state: State):
    """Execute tools and collect metadata"""
    messages = state["messages"]
    last_message = messages[-1]
    
    if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
        return {"messages": []}
    
    tool_outputs = []
    sources_collected = state.get("sources_collected", [])
    tools_used = state.get("tools_used", [])
    confidence_scores = state.get("confidence_scores", [])
    
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]
        
        tools_used.append(tool_name)
        
        try:
            if tool_name in tools_by_name:
                tool = tools_by_name[tool_name]
                
                if tool_name == "retrieve_knowledge":
                    content, metadata = tool.invoke(tool_args)
                    sources_collected.extend(metadata.get("sources", []))
                    confidence_scores.append(metadata.get("confidence", 0.0))
                    
                elif tool_name == "search_web":
                    result = tool.invoke(tool_args)
                    content = result["content"]
                    sources_collected.extend(result.get("sources", []))
                    confidence_scores.append(result.get("confidence", 0.0))
                    
                else:  # human_assistance
                    result = tool.invoke(tool_args)
                    content = result
                
                tool_message = ToolMessage(
                    content=content,
                    name=tool_name,
                    tool_call_id=tool_id,
                )
                tool_outputs.append(tool_message)
                
        except InterruptException as e:
            return {
                "interrupted": True,
                "interrupt_data": e.data,
                "messages": tool_outputs,
                "sources_collected": sources_collected,
                "tools_used": tools_used,
                "confidence_scores": confidence_scores
            }
        except Exception as e:
            error_message = ToolMessage(
                content=f"Error: {str(e)}",
                name=tool_name,
                tool_call_id=tool_id,
            )
            tool_outputs.append(error_message)
    
    return {
        "messages": tool_outputs, 
        "interrupted": False,
        "sources_collected": sources_collected,
        "tools_used": tools_used,
        "confidence_scores": confidence_scores
    }

def generate_structured_response(state: State) -> dict:
    """Generate final structured JSON response"""
    import time
    
    messages = state["messages"]
    chat_id = state.get("chat_id", str(uuid.uuid4()))
    start_time = state.get("processing_start", time.time())
    processing_time = int((time.time() - start_time) * 1000)
    
    # Get AI response content
    ai_response = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content and not getattr(msg, 'tool_calls', None):
            ai_response = msg.content
            break
    
    # Handle interruption case
    if state.get("interrupted", False):
        interrupt_data = state.get("interrupt_data", {})
        return ChatResponse(
            chat_id=chat_id,
            message_id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            response="I need human assistance to properly answer your question.",
            tools_used=state.get("tools_used", []),
            sources=[],
            confidence=0.0,
            processing_time_ms=processing_time,
            status="interrupted",
            requires_human_input=True,
            human_query=interrupt_data.get("query", "")
        ).dict()
    
    # Calculate overall confidence
    confidence_scores = state.get("confidence_scores", [])
    overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.7
    
    # Prepare sources
    sources = []
    for source_data in state.get("sources_collected", []):
        sources.append(SourceMetadata(**source_data))
    
    # Generate reasoning
    tools_used = state.get("tools_used", [])
    reasoning = f"Used {', '.join(tools_used)} to gather information" if tools_used else "Responded using general knowledge"
    
    return ChatResponse(
        chat_id=chat_id,
        message_id=str(uuid.uuid4()),
        timestamp=datetime.now().isoformat(),
        response=ai_response or "I apologize, but I couldn't generate a proper response.",
        tools_used=tools_used,
        sources=sources,
        reasoning=reasoning,
        confidence=round(overall_confidence, 2),
        processing_time_ms=processing_time,
        status="success" if ai_response else "error",
        error_message=None if ai_response else "No valid response generated",
        requires_human_input=False
    ).dict()

def should_continue(state: State) -> Literal["tools", "generate", "end"]:
    """Enhanced routing with response generation"""
    if state.get("interrupted", False):
        return "generate"  # Generate interrupted response
        
    messages = state["messages"]
    if not messages:
        return "end"
        
    last_message = messages[-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    return "generate"  # Generate final structured response

# Create enhanced graph
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", execute_tools)
graph_builder.add_node("generate", generate_structured_response)

graph_builder.add_conditional_edges("chatbot", should_continue, {
    "tools": "tools", 
    "generate": "generate",
    "end": END
})
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("generate", END)
graph_builder.add_edge(START, "chatbot")

graph = graph_builder.compile(checkpointer=memory)

def load_knowledge_base():
    """Load sample knowledge base"""
    try:
        sources = ["https://lilianweng.github.io/posts/2023-06-23-agent/"]
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        
        for source in sources:
            loader = WebBaseLoader(
                web_paths=[source],
                bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("post-content", "post-title")))
            )
            docs = loader.load()
            splits = text_splitter.split_documents(docs)
            
            # Add chunk IDs for tracking
            for i, doc in enumerate(splits):
                doc.metadata["chunk_id"] = f"chunk_{i:03d}"
                doc.metadata["title"] = "LLM Powered Autonomous Agents"
            
            vector_store.add_documents(splits)
            print(f"✅ Loaded {len(splits)} chunks from {source}")
            
    except Exception as e:
        print(f"⚠️  Knowledge base loading failed: {e}")

def test_json_response(query: str, chat_id: str = None) -> dict:
    """Test structured JSON response"""
    import time
    
    config = {"configurable": {"thread_id": chat_id or str(uuid.uuid4())}}
    
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "interrupted": False,
        "interrupt_data": {},
        "chat_id": chat_id or str(uuid.uuid4()),
        "processing_start": time.time(),
        "sources_collected": [],
        "tools_used": [],
        "confidence_scores": []
    }
    
    result = graph.invoke(initial_state, config)
    return result

def run_json_chatbot():
    """Interactive JSON chatbot"""
    print("🚀 JSON Structured RAG Chatbot")
    print("=" * 40)
    
    # Load knowledge base
    print("📚 Loading knowledge base...")
    load_knowledge_base()
    
    print("\n✨ Features:")
    print("✅ Industry standard JSON responses")
    print("✅ Source metadata with confidence scores")
    print("✅ Processing time tracking")
    print("✅ Error handling and status codes")
    print("✅ Human-in-the-loop support")
    
    chat_id = str(uuid.uuid4())
    print(f"\n🆔 Chat ID: {chat_id}")
    print("\nType 'quit' to exit\n")
    
    while True:
        try:
            query = input("User: ").strip()
            if query.lower() in ["quit", "exit", "q"]:
                break
                
            if not query:
                continue
                
            print("\n📤 Processing...")
            response = test_json_response(query, chat_id)
            
            print("\n📋 Structured JSON Response:")
            print("=" * 50)
            print(json.dumps(response, indent=2, ensure_ascii=False))
            print("=" * 50)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            error_response = ChatResponse(
                chat_id=chat_id,
                message_id=str(uuid.uuid4()),
                timestamp=datetime.now().isoformat(),
                response="An error occurred while processing your request.",
                status="error",
                error_message=str(e),
                confidence=0.0
            )
            print(f"\n❌ Error Response:")
            print(json.dumps(error_response.dict(), indent=2))

if __name__ == "__main__":
    run_json_chatbot()