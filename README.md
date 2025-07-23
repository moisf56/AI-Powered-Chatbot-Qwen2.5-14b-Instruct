# **🤖 Human-in-the-Loop AI Chatbot**

**Intelligent Conversational AI with Expert Human Assistance Integration**

[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat&logo=python)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-green?style=flat)](https://langchain.com)
[![Flask](https://img.shields.io/badge/Flask-2.3+-orange?style=flat&logo=flask)](https://flask.palletsprojects.com)
[![React](https://img.shields.io/badge/React-18+-61DAFB?style=flat&logo=react)](https://reactjs.org)

## **🎯 Overview**

An advanced conversational AI system that seamlessly combines autonomous AI capabilities with human expert intervention when needed. Built using LangGraph for sophisticated conversation flow management, the system can pause execution to request human assistance for complex decisions, sensitive topics, or specialized knowledge requirements.

### **Key Innovation**
Unlike traditional chatbots that operate in isolation, this system recognizes its limitations and can **interrupt its own execution** to request human expert input, then seamlessly resume the conversation with enhanced context and accuracy.

## **⚡ Features**

### **🧠 Advanced AI Capabilities**
- **LangGraph Integration**: State-of-the-art conversation flow management
- **Memory Persistence**: Maintains context across conversation threads
- **Web Search Integration**: Real-time information retrieval using DuckDuckGo
- **Multi-turn Conversations**: Complex dialogue handling with state management

### **🤝 Human-in-the-Loop Architecture**
- **Intelligent Interruption**: AI autonomously requests human assistance when needed
- **Seamless Handoff**: Smooth transition between AI and human responses
- **Resume Capability**: Continues conversation flow after human input
- **Expert Integration**: Escalation to domain specialists for complex queries

### **🔍 Real-time Information Access**
- **Web Search Tools**: Live data retrieval and fact-checking
- **Source Attribution**: Transparent sourcing of external information
- **Content Filtering**: Safe and relevant information processing

### **💻 Professional Interface**
- **React Frontend**: Modern, responsive user interface
- **Real-time Indicators**: Visual cues for AI thinking, human assistance requests
- **Thread Management**: Multiple conversation sessions with persistence
- **Mobile Responsive**: Optimized for all device sizes

## **🛠 Technology Stack**

### **Backend Architecture**
- **Python 3.8+**: Core application logic and API development
- **LangGraph**: Advanced conversation state management and flow control
- **LangChain**: LLM integration and tool orchestration
- **Flask**: RESTful API server with CORS support

### **AI & Language Models**
- **LangGraph**: Advanced conversation state management and flow control
- **LangChain**: LLM integration and tool orchestration
- **OpenAI-Compatible API**: Flexible model integration (Qwen2.5-14B-Instruct)
- **Tool Integration**: Web search, human assistance, and extensible tool framework

### **Frontend & Deployment**
- **React**: Modern component-based user interface
- **Tailwind CSS**: Utility-first styling framework
- **Railway**: Scalable cloud hosting with automatic deployments
- **Vercel**: Optional frontend deployment platform

### **Infrastructure**
- **RunPod GPU**: High-performance model serving with A40 GPU
- **vLLM**: Optimized inference serving for large language models  
- **Memory Checkpointing**: Persistent conversation state across sessions
- **DuckDuckGo Search**: Privacy-focused web search integration

## **🚀 Quick Start**

### **Prerequisites**
```bash
# Python 3.8+ required
python --version

# Git for cloning repository
git --version
