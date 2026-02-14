"use client";

import React, { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { Avatar, Button, TextInput, Spinner } from "flowbite-react";
import { HiPaperAirplane, HiOutlineRefresh, HiPaperClip, HiX } from "react-icons/hi";
import { propagateServerField } from "next/dist/server/lib/render-server";
import { threadId } from "worker_threads";

type Message = {
  role: "user" | "ai";
  content: string;
  images?: string[];
  time: string;
};

export default function Home() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [currentThreadId, setCurrentThreadId] = useState("");

  useEffect(() => {
    if (typeof window !== 'undefined') {
      setCurrentThreadId(self.crypto.randomUUID());
    }
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  const getCurrentTime = () =>
    new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

  const handleIconClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setSelectedImage(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const clearImage = () => {
    setSelectedImage(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const handleSend = async () => {
    if (!input.trim() && !selectedImage) return;
    const userText = input;
    const userImage = selectedImage;

    setInput("");
    setSelectedImage(null);
    setMessages((prev) => [...prev, { role: "user", content: userText, images: userImage ? [userImage] : [], time: getCurrentTime() }]);
    setIsLoading(true);

    try {
      const response = await fetch("http://127.0.0.1:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: userText, image: userImage, thread_id: currentThreadId }),
      });
      if (!response.ok) throw new Error("Network error");
      const data = await response.json();
      setMessages((prev) => [...prev, { role: "ai", content: data.response, images: data.images || [], time: getCurrentTime() }]);
    } catch (error) {
      setMessages((prev) => [...prev, { role: "ai", content: "! System Error.", time: getCurrentTime() }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = async () => {
    try {
      await fetch("http://127.0.0.1:8000/reset", { 
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({thread_id: currentThreadId})
      });
      setMessages([]);
    } catch (error) {
      console.error("Reset failed", error);
    }
  };

  // Custom Theme to remove borders from the library Input to match your design
  const inputTheme = {
    field: {
      input: {
        base: "w-full border-none focus:ring-0 bg-transparent text-gray-800 placeholder-gray-400 shadow-none",
        sizes: {
            lg: "p-4 sm:text-md"
        }
      }
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gradient-to-br from-purple-100 via-pink-100 to-blue-100">
      
      {/* --- HEADER --- */}
      <header className="flex-none px-6 py-4 bg-white shadow-sm flex items-center justify-between z-10 sticky top-0">
        <div className="flex items-center gap-3">
          {/* LIBRARY COMPONENT: Avatar */}
          <Avatar rounded placeholderInitials="JA" color="purple" />
          <span className="text-lg font-bold text-gray-800 tracking-tight">Jewellery Assistant</span>
        </div>
        
        {/* ✅ LIBRARY COMPONENT: Button */}
        <Button size="xs" color="gray" pill onClick={handleReset}>
           <HiOutlineRefresh className="mr-2 h-4 w-4" />
           Reset
        </Button>
      </header>

      {/* --- CHAT AREA --- */}
      <main className="flex-1 overflow-y-auto p-6 space-y-6">
        {messages.length === 0 && (
          <div className="h-full flex flex-col items-center justify-center opacity-40">
             {/* Using a library icon or standard SVG here */}
             <div className="w-16 h-16 bg-gray-200 rounded-full flex items-center justify-center mb-4">
                <HiPaperAirplane className="w-8 h-8 text-gray-400 rotate-90 ml-1" />
             </div>
             <p className="text-gray-500 text-xl font-medium">Start a conversation...</p>
          </div>
        )}

        {messages.map((msg, idx) => (
          <ChatBubble key={idx} message={msg} />
        ))}

        {isLoading && <LoadingBubble />}
        <div ref={messagesEndRef} />
      </main>

      {/* --- FOOTER --- */}
      {/* --- FOOTER --- */}
<footer className="flex-none p-4 bg-transparent z-20">
  <div className="max-w-4xl mx-auto w-full flex flex-col gap-3"> {/* gap-3 adds space between image and bar */}

    {/* 1. IMAGE PREVIEW AREA (Floating Above) */}
    {/* Only renders if an image is selected */}
    {selectedImage && (
      <div className="flex justify-start px-4 animate-fade-in-up">
        <div className="relative w-20 h-20 group">
          <img
            src={selectedImage}
            alt="Preview"
            className="w-full h-full object-cover rounded-xl border border-gray-200 shadow-sm"
          />
          {/* Close Button (X) */}
          <button
            onClick={clearImage}
            className="absolute -top-2 -right-2 bg-gray-100 hover:bg-gray-200 text-gray-600 rounded-full p-1 border border-gray-300 transition-transform hover:scale-110"
          >
            <HiX className="w-3 h-3" />
          </button>
        </div>
      </div>
    )}

    {/* 2. INPUT BAR (The White Pill) */}
    <div className="w-full bg-white rounded-[26px] shadow-lg border border-gray-100 px-4 py-2 flex flex-row items-center gap-3">
      
      {/* Hidden File Input */}
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileChange}
        accept="image/*"
        className="hidden"
      />

      {/* Paperclip Icon (Left) */}
      <button
        onClick={handleIconClick}
        className={`p-2 rounded-full transition-all ${
          selectedImage 
            ? 'text-purple-600 bg-purple-50' 
            : 'text-gray-400 hover:text-gray-600 hover:bg-gray-100'
        }`}
        title="Upload Image"
      >
        <HiPaperClip className="w-6 h-6" />
      </button>

      {/* Text Input (Middle) */}
      <div className="flex-1 min-w-0">
        <TextInput
          id="chat-input"
          sizing="md"
          placeholder={selectedImage ? "Ask about this image..." : "Write your message..."}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSend()}
          disabled={isLoading}
          theme={{
            field: {
              input: {
                base: "w-full border-none focus:ring-0 bg-transparent text-gray-900 placeholder-gray-400 p-0 text-lg",
                sizes: { md: "p-0" } // Removes default padding so it centers perfectly
              }
            }
          }}
          className="focus:ring-0"
        />
      </div>

      {/* Send Button (Right) */}
      <button
        onClick={handleSend}
        disabled={(!input.trim() && !selectedImage) || isLoading}
        className={`p-2 rounded-full transition-all flex items-center justify-center ${
          (!input.trim() && !selectedImage) || isLoading
            ? "bg-gray-100 text-gray-300 cursor-not-allowed"
            : "bg-[#00BFA5] text-white hover:bg-[#00a892] shadow-md hover:shadow-lg active:scale-95"
        }`}
      >
        {isLoading ? (
          <Spinner size="xs" color="white" />
        ) : (
          <HiPaperAirplane className="w-5 h-5 rotate-90 ml-0.5" />
        )}
      </button>
      
    </div>
  </div>
</footer>
    </div>
  );
}

// --- BUBBLE COMPONENT ---
const injectImagesIntoMarkdown = (text: string, images?: string[]) => {
  if (!images || images.length === 0) return text;
  
  console.log("images: ", images);
  return text.replace(/\(image_(\d+)\)/g, (match, indexStr) => {
    const index = parseInt(indexStr, 10);
    return images[index] ? `(${images[index]})` : match;
  });
};

function ChatBubble({ message }: { message: Message }) {
  const isUser = message.role === "user";

  // 1. Prepare AI Content
  const displayContent = isUser
    ? message.content
    : injectImagesIntoMarkdown(message.content, message.images);

  // 2. Define Custom Image for AI
  const CustomImage = ({ node, ...props }: any) => {
    const src = props.src;
    if (!src || src === "" || src.startsWith("image_")) return null;
    return (
      <img
        {...props}
        className="rounded-xl shadow-md max-h-[250px] w-auto my-3 border border-gray-100 object-cover bg-white"
        alt={props.alt || "Product Image"}
        loading="lazy"
      />
    );
  };

  return (
    <div className={`flex gap-4 max-w-[85%] md:max-w-[75%] ${isUser ? "ml-auto flex-row-reverse" : ""}`}>
      <Avatar
        rounded
        img={isUser ? undefined : "/avatar_ai.jpg"}
        placeholderInitials={isUser ? "ME" : undefined}
        color={isUser ? "gray" : "transparent"}
        className="mt-1 flex-shrink-0"
      />

      <div className={`flex flex-col w-full min-w-0 ${isUser ? "items-end" : ""}`}>
         <span className={`text-xs text-gray-500 mb-1 ${isUser ? "text-right" : "text-left"}`}>
            {isUser ? "You" : "Assistant"} • {message.time}
         </span>

         <div className={`px-5 py-3 shadow-lg backdrop-blur-xl border text-[15px] leading-relaxed break-words
            ${isUser
              ? "w-fit bg-gradient-to-br from-pink-400/70 via-indigo-400/70 to-blue-400/70 text-white border-2 border-white/100 ring-2 ring-white/30 rounded-[22px] rounded-tr-sm"
              : "bg-white bg-opacity-30 backdrop-blur-lg rounded-[22px] rounded-tl-sm"
            }`}>

            {isUser ? (
               <div className="flex flex-col gap-2 items-start">
                  {message.images && message.images.map((img, idx) => (
                    <img
                      key={idx}
                      src={img}
                      alt="My Upload"
                      className="rounded-lg max-h-[250px] w-auto h-auto max-w-full object-contain border border-white/30 shadow-sm"
                    />
                  ))}
                  {message.content && <span>{message.content}</span>}
               </div>
            ) : (
               <ReactMarkdown
                 remarkPlugins={[remarkGfm]}
                 urlTransform={(value) => value}
                 components={{
                   img: CustomImage,
                   a: ({node, ...props}) => (
                      <a {...props} className="text-blue-600 underline font-medium" target="_blank" />
                   )
                 }}
               >
                 {displayContent}
               </ReactMarkdown>
            )}
         </div>
      </div>
    </div>
  );
}

function LoadingBubble() {
  return (
    <div className="flex gap-4 max-w-[80%]">
      <Avatar rounded img="avatar_ai.jpg" className="mt-1 flex-shrink-0" />
      <div className="flex flex-col">
         <span className="text-xs text-gray-500 mb-1">Assistant is typing...</span>
         <div className="px-5 py-3 bg-white rounded-[20px] rounded-tl-none shadow-sm w-fit flex items-center gap-2">
            <Spinner size="sm" color="purple" />
            <span className="text-xs text-gray-500">Thinking...</span>
         </div>
      </div>
    </div>
  );
}