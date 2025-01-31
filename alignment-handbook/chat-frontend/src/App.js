import React, { useState, useRef, useEffect } from 'react';

// 아이콘 컴포넌트들
const SendIcon = () => (
  <svg 
    xmlns="http://www.w3.org/2000/svg" 
    width="24" 
    height="24" 
    viewBox="0 0 24 24" 
    fill="none" 
    stroke="currentColor" 
    strokeWidth="2" 
    strokeLinecap="round" 
    strokeLinejoin="round"
  >
    <line x1="22" y1="2" x2="11" y2="13"></line>
    <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
  </svg>
);

const ResetIcon = () => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width="24"
    height="24"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"></path>
    <path d="M3 3v5h5"></path>
  </svg>
);

// 채팅 메시지 컴포넌트
const ChatMessage = ({ role, content }) => (
  <div className={`flex w-full ${role === 'user' ? 'justify-end' : 'justify-start'} mb-4`}>
    <div className={`max-w-[80%] rounded-lg p-4 ${
      role === 'user' 
        ? 'bg-blue-500 text-white' 
        : 'bg-gray-200 text-gray-800'
    }`}>
      <p className="whitespace-pre-wrap">{content}</p>
    </div>
  </div>
);

// 로딩 인디케이터 컴포넌트
const LoadingIndicator = () => (
  <div className="flex justify-center my-4">
    <div className="animate-pulse flex space-x-2">
      <div className="w-2 h-2 bg-gray-400 rounded-full"></div>
      <div className="w-2 h-2 bg-gray-400 rounded-full animation-delay-200"></div>
      <div className="w-2 h-2 bg-gray-400 rounded-full animation-delay-400"></div>
    </div>
  </div>
);

const ChatApp = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const chatContainerRef = useRef(null);

  // 새 메시지가 추가될 때마다 스크롤을 아래로 이동
  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [messages]);

  // 초기화 핸들러
  const handleReset = async () => {
    if (isLoading) return;
    
    if (messages.length > 0) {
      if (window.confirm('대화 내용을 모두 삭제하시겠습니까?')) {
        setIsLoading(true);
        try {
          const response = await fetch('http://localhost:8000/reset', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            credentials: 'include',
          });

          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }

          const result = await response.json();
          if (result.status === 'success') {
            setMessages([]);
            setInput('');
          } else {
            throw new Error(result.message || '초기화 중 오류가 발생했습니다.');
          }
        } catch (error) {
          console.error('Reset error:', error);
          alert('초기화 중 오류가 발생했습니다. 다시 시도해주세요.');
        } finally {
          setIsLoading(false);
        }
      }
    }
  };

  // 메시지 전송 핸들러
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage = { role: 'user', content: input.trim() };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'text/event-stream',
        },
        body: JSON.stringify({
          messages: [...messages, userMessage]
        }),
        credentials: 'include',
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        
        if (done) {
          console.log('Stream complete');
          break;
        }

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6).trim();
            
            if (data === '[DONE]') {
              setIsLoading(false);
              break;
            }

            try {
              const parsed = JSON.parse(data);
              setMessages(prev => {
                const newMessages = [...prev];
                const lastMessage = newMessages[newMessages.length - 1];

                if (lastMessage && lastMessage.role === 'assistant') {
                  newMessages[newMessages.length - 1] = {
                    ...lastMessage,
                    content: parsed.full_response
                  };
                } else {
                  newMessages.push({
                    role: 'assistant',
                    content: parsed.full_response
                  });
                }

                return newMessages;
              });
            } catch (e) {
              console.error('Failed to parse chunk:', e);
            }
          }
        }
      }
    } catch (error) {
      console.error('Error:', error);
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: '죄송합니다. 오류가 발생했습니다. 다시 시도해주세요.' 
      }]);
      setIsLoading(false);
    }
  };

  // Enter 키 핸들러 (Shift+Enter는 줄바꿈)
  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gray-100">
      {/* 헤더 */}
      <div className="bg-white shadow-sm p-4 flex justify-between items-center">
        <h1 className="text-xl font-bold">AI Chat</h1>
        <button
          onClick={handleReset}
          disabled={isLoading || messages.length === 0}
          className={`p-2 rounded-full hover:bg-gray-100 transition-colors
            ${(isLoading || messages.length === 0) ? 'text-gray-400 cursor-not-allowed' : 'text-gray-600'}`}
          title="대화 내용 초기화"
        >
          <ResetIcon />
        </button>
      </div>
      
      {/* 채팅 영역 */}
      <div 
        ref={chatContainerRef}
        className="flex-1 overflow-y-auto p-4 space-y-4"
      >
        {messages.length === 0 ? (
          <div className="text-center text-gray-500 mt-8">
            새로운 대화를 시작해보세요!
          </div>
        ) : (
          <>
            {messages.map((message, index) => (
              <ChatMessage
                key={index}
                role={message.role}
                content={message.content}
              />
            ))}
            {isLoading && <LoadingIndicator />}
          </>
        )}
      </div>

      {/* 입력 폼 */}
      <form 
        onSubmit={handleSubmit} 
        className="p-4 bg-white border-t"
      >
        <div className="flex space-x-4">
          <textarea
            rows="1"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="메시지를 입력하세요... (Enter: 전송, Shift+Enter: 줄바꿈)"
            className="flex-1 p-2 border rounded-lg focus:outline-none focus:border-blue-500 resize-none"
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={isLoading || !input.trim()}
            className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:bg-gray-400 disabled:cursor-not-allowed flex items-center justify-center"
          >
            <SendIcon />
          </button>
        </div>
      </form>
    </div>
  );
};

export default ChatApp;