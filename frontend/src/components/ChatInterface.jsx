import React, { useState } from "react";
import "./ChatInterface.css";

const API = "http://127.0.0.1:8000";

const ChatInterface = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [file, setFile] = useState(null);
  const [rawPreview, setRawPreview] = useState(null);
  const [isUploading, setIsUploading] = useState(false);

  // ======================
  // Upload CSV
  // ======================
  const uploadCSV = async () => {
    if (!file) return;

    setIsUploading(true);
    setMessages([]);
    setRawPreview(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      // 1️⃣ Upload CSV
      await fetch(`${API}/upload_csv`, {
        method: "POST",
        body: formData,
      });

      // 2️⃣ Fetch raw preview
      const previewRes = await fetch(`${API}/raw_preview`);
      const previewData = await previewRes.json();
      setRawPreview(previewData);

      // 3️⃣ Get initial assistant message
      const chatRes = await fetch(`${API}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: "" }),
      });
      const chatData = await chatRes.json();

      setMessages([{ sender: "bot", text: chatData.reply }]);
    } catch {
      setMessages([{ sender: "bot", text: "❌ CSV upload failed." }]);
    } finally {
      setIsUploading(false);
    }
  };

  // ======================
  // Send Chat
  // ======================
  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMsg = { sender: "user", text: input };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setIsLoading(true);

    try {
      const res = await fetch(`${API}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userMsg.text }),
      });

      const data = await res.json();
      setMessages((prev) => [...prev, { sender: "bot", text: data.reply }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="chat-root">
      {/* TOP BAR */}
      <div className="top-bar">
        <div className="upload-controls">
          <input type="file" accept=".csv" onChange={(e) => setFile(e.target.files[0])} />
          <button onClick={uploadCSV}>Upload CSV</button>
        </div>
        <div className="title">AI Assistant</div>
      </div>

      {/* CHAT */}
      <div className="chat-body">
        <div className="messages-area">

          {isUploading && (
            <div className="system-hint">Uploading dataset… ⏳</div>
          )}

          {/* RAW DATA PREVIEW TABLE */}
          {rawPreview && (
            <div className="table-wrapper">
              <div className="table-title">Raw Data Preview</div>
              <table className="data-table">
                <thead>
                  <tr>
                    {rawPreview.columns.map((col) => (
                      <th key={col}>{col}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {rawPreview.rows.map((row, i) => (
                    <tr key={i}>
                      {rawPreview.columns.map((col) => (
                        <td key={col}>{row[col]}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {/* CHAT MESSAGES */}
          {messages.map((msg, i) => (
            <div key={i} className={`message ${msg.sender}`}>
              {msg.text}
            </div>
          ))}

          {isLoading && <div className="message bot">Typing…</div>}
        </div>

        {/* INPUT */}
        <div className="input-area">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type a message..."
            onKeyDown={(e) => e.key === "Enter" && sendMessage()}
          />
          <button onClick={sendMessage}>Send</button>
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;
