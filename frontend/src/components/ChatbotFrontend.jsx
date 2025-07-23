// ChatbotFrontend.jsx
import React, { useState, useEffect, useRef } from 'react';
import { Send, Bot, User, Search, Brain, Pause, Play, MessageSquare, Zap, Loader2 } from 'lucide-react';

const ChatbotFrontend = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const [threadId, setThreadId] = useState('1');
  const [isInterrupted, setIsInterrupted] = useState(false);
  const [interruptData, setInterruptData] = useState(null);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Test connection on load
  useEffect(() => {
    testConnection();
  }, []);

  const testConnection = async () => {
    try {
      const response = await fetch('/api/test-connection', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      
      if (response.ok) {
        setConnectionStatus('connected');
      } else {
        setConnectionStatus('error');
      }
    } catch (error) {
      setConnectionStatus('error');
    }
  };

  const sendMessage = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage = {
      id: Date.now(),
      type: 'user',
      content: input,
      timestamp: new Date().toLocaleTimeString()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: input,
          thread_id: threadId
        })
      });

      const data = await response.json();

      if (data.status === 'INTERRUPTED') {
        setIsInterrupted(true);
        setInterruptData(data.interrupt_data);
        
        const interruptMessage = {
          id: Date.now() + 1,
          type: 'system',
          content: `🚨 AI needs human assistance: ${data.interrupt_data?.query || 'Waiting for input'}`,
          timestamp: new Date().toLocaleTimeString()
        };
        
        setMessages(prev => [...prev, interruptMessage]);
      } else if (data.response) {
        const botMessage = {
          id: Date.now() + 1,
          type: 'assistant',
          content: data.response,
          timestamp: new Date().toLocaleTimeString(),
          tools_used: data.tools_used || []
        };
        
        setMessages(prev => [...prev, botMessage]);
      }
    } catch (error) {
      const errorMessage = {
        id: Date.now() + 1,
        type: 'error',
        content: 'Sorry, I encountered an error. Please try again.',
        timestamp: new Date().toLocaleTimeString()
      };
      
      setMessages(prev => [...prev, errorMessage]);
    }

    setIsLoading(false);
  };

  const resumeExecution = async () => {
    if (!input.trim() || !isInterrupted) return;

    const resumeMessage = {
      id: Date.now(),
      type: 'user',
      content: `Resume: ${input}`,
      timestamp: new Date().toLocaleTimeString()
    };

    setMessages(prev => [...prev, resumeMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetch('/api/resume', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          response: input,
          thread_id: threadId
        })
      });

      const data = await response.json();
      
      if (data.response) {
        const botMessage = {
          id: Date.now() + 1,
          type: 'assistant',
          content: data.response,
          timestamp: new Date().toLocaleTimeString()
        };
        
        setMessages(prev => [...prev, botMessage]);
      }

      setIsInterrupted(false);
      setInterruptData(null);
    } catch (error) {
      const errorMessage = {
        id: Date.now() + 1,
        type: 'error',
        content: 'Error resuming conversation.',
        timestamp: new Date().toLocaleTimeString()
      };
      
      setMessages(prev => [...prev, errorMessage]);
    }

    setIsLoading(false);
  };

  const newThread = () => {
    const newId = String(parseInt(threadId) + 1);
    setThreadId(newId);
    setMessages([]);
    setIsInterrupted(false);
    setInterruptData(null);
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (isInterrupted) {
        resumeExecution();
      } else {
        sendMessage();
      }
    }
  };

  const getMessageIcon = (type) => {
    switch (type) {
      case 'user': return <User className="w-5 h-5" />;
      case 'assistant': return <Bot className="w-5 h-5" />;
      case 'system': return <Brain className="w-5 h-5" />;
      case 'error': return <Zap className="w-5 h-5" />;
      default: return <MessageSquare className="w-5 h-5" />;
    }
  };

  const getMessageStyle = (type) => {
    const baseStyle = "flex gap-3 p-4 rounded-lg max-w-4xl mx-auto";
    
    switch (type) {
      case 'user':
        return `${baseStyle} bg-blue-50 border-l-4 border-blue-400`;
      case 'assistant':
        return `${baseStyle} bg-green-50 border-l-4 border-green-400`;
      case 'system':
        return `${baseStyle} bg-yellow-50 border-l-4 border-yellow-400`;
      case 'error':
        return `${baseStyle} bg-red-50 border-l-4 border-red-400`;
      default:
        return `${baseStyle} bg-gray-50`;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-6xl mx-auto p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-blue-600 rounded-lg">
                <Bot className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-800">
                  AI Assistant with Human-in-the-Loop
                </h1>
                <p className="text-sm text-gray-600">
                  Powered by Qwen2.5-14B • Web Search • Memory • RunPod GPU
                </p>
              </div>
            </div>
            
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <div className={`w-3 h-3 rounded-full ${
                  connectionStatus === 'connected' ? 'bg-green-400' : 
                  connectionStatus === 'error' ? 'bg-red-400' : 'bg-yellow-400'
                }`} />
                <span className="text-sm text-gray-600 capitalize">
                  {connectionStatus}
                </span>
              </div>
              
              <button
                onClick={newThread}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors text-sm font-medium"
              >
                New Thread
              </button>
            </div>
          </div>
          
          {/* Features */}
          <div className="flex gap-4 mt-3 text-xs">
            <div className="flex items-center gap-1 text-gray-600">
              <Search className="w-3 h-3" />
              <span>Web Search</span>
            </div>
            <div className="flex items-center gap-1 text-gray-600">
              <Brain className="w-3 h-3" />
              <span>Memory</span>
            </div>
            <div className="flex items-center gap-1 text-gray-600">
              <Pause className="w-3 h-3" />
              <span>Human Assistance</span>
            </div>
          </div>
        </div>
      </div>

      {/* Chat Area */}
      <div className="max-w-6xl mx-auto p-4 pb-32">
        {messages.length === 0 ? (
          <div className="text-center py-12">
            <Bot className="w-16 h-16 text-gray-400 mx-auto mb-4" />
            <h2 className="text-xl font-semibold text-gray-700 mb-2">
              Welcome to AI Assistant
            </h2>
            <p className="text-gray-600 mb-6 max-w-2xl mx-auto">
              This AI assistant can search the web, remember our conversation, and request human assistance when needed. 
              Try asking complex questions or requesting expert advice!
            </p>
            
            <div className="grid md:grid-cols-2 gap-4 max-w-4xl mx-auto text-left">
              <div className="p-4 bg-white rounded-lg border hover:shadow-md transition-shadow cursor-pointer"
                   onClick={() => setInput("I need expert advice on career change")}>
                <h3 className="font-semibold text-gray-800 mb-2">Expert Assistance</h3>
                <p className="text-gray-600 text-sm">Request human expert help for complex decisions</p>
              </div>
              
              <div className="p-4 bg-white rounded-lg border hover:shadow-md transition-shadow cursor-pointer"
                   onClick={() => setInput("What's the latest news in AI?")}>
                <h3 className="font-semibold text-gray-800 mb-2">Web Search</h3>
                <p className="text-gray-600 text-sm">Get real-time information from the web</p>
              </div>
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            {messages.map((message) => (
              <div key={message.id} className={getMessageStyle(message.type)}>
                <div className="flex-shrink-0 p-2 rounded-full bg-white shadow-sm">
                  {getMessageIcon(message.type)}
                </div>
                
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="font-medium text-gray-800 capitalize">
                      {message.type === 'assistant' ? 'AI Assistant' : message.type}
                    </span>
                    <span className="text-xs text-gray-500">{message.timestamp}</span>
                    {message.tools_used && message.tools_used.length > 0 && (
                      <span className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded-full">
                        Used: {message.tools_used.join(', ')}
                      </span>
                    )}
                  </div>
                  
                  <div className="text-gray-700 whitespace-pre-wrap break-words">
                    {message.content}
                  </div>
                </div>
              </div>
            ))}
            
            {isLoading && (
              <div className="flex gap-3 p-4 rounded-lg max-w-4xl mx-auto bg-gray-50">
                <div className="flex-shrink-0 p-2 rounded-full bg-white shadow-sm">
                  <Bot className="w-5 h-5" />
                </div>
                <div className="flex items-center gap-2">
                  <Loader2 className="w-4 h-4 animate-spin text-blue-500" />
                  <span className="text-gray-600 text-sm">Thinking...</span>
                </div>
              </div>
            )}
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="fixed bottom-0 left-0 right-0 bg-white border-t shadow-lg">
        <div className="max-w-6xl mx-auto p-4">
          {isInterrupted && (
            <div className="mb-3 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
              <div className="flex items-center gap-2">
                <Pause className="w-4 h-4 text-yellow-600" />
                <span className="text-sm font-medium text-yellow-800">
                  Execution Paused - AI needs human input
                </span>
              </div>
              {interruptData?.query && (
                <p className="text-sm text-yellow-700 mt-1">
                  Question: {interruptData.query}
                </p>
              )}
            </div>
          )}
          
          <div className="flex gap-3">
            <div className="flex-1 relative">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder={isInterrupted ? "Provide your response to resume..." : "Type your message..."}
                className="w-full px-4 py-3 border rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                rows="3"
                disabled={isLoading}
              />
            </div>
            
            <button
              onClick={isInterrupted ? resumeExecution : sendMessage}
              disabled={isLoading || !input.trim()}
              className={`px-6 py-3 rounded-lg font-medium transition-colors flex items-center gap-2 ${
                isLoading || !input.trim()
                  ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                  : isInterrupted
                  ? 'bg-green-600 text-white hover:bg-green-700'
                  : 'bg-blue-600 text-white hover:bg-blue-700'
              }`}
            >
              {isLoading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Processing
                </>
              ) : isInterrupted ? (
                <>
                  <Play className="w-4 h-4" />
                  Resume
                </>
              ) : (
                <>
                  <Send className="w-4 h-4" />
                  Send
                </>
              )}
            </button>
          </div>
          
          <div className="flex justify-between items-center mt-2 text-xs text-gray-500">
            <span>Thread: {threadId} • Press Enter to send, Shift+Enter for new line</span>
            <span>{connectionStatus === 'connected' ? 'Connected to RunPod GPU' : 'Connection issues'}</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChatbotFrontend;