// ChatInterface.jsx
import React, { useState, useRef, useEffect } from "react";
import "./ChatInterface.css";

const API = "http://127.0.0.1:8000";

const ChatInterface = () => {
  const [messages, setMessages] = useState([
    {
      text: "Hi 👋 Upload a CSV to get started.",
      sender: "bot",
      timestamp: new Date(),
    },
  ]);

  const [inputText, setInputText] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);

  const [rawPreview, setRawPreview] = useState(null);
  const [trainResult, setTrainResult] = useState(null);
  const [forecastResult, setForecastResult] = useState(null);

  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, rawPreview, trainResult, forecastResult]);

  // ---------------- Upload ----------------
  const handleUpload = async () => {
    if (!selectedFile) return;

    setIsUploading(true);
    setRawPreview(null);
    setTrainResult(null);
    setForecastResult(null);

    try {
      const fd = new FormData();
      fd.append("file", selectedFile);

      const res = await fetch(`${API}/upload_csv`, {
        method: "POST",
        body: fd,
      });

      const data = await res.json();

      const botMessages = data.messages
        .filter((m) => m.role === "assistant")
        .map((m) => ({
          text: m.content,
          sender: "bot",
          timestamp: new Date(),
        }));

      setMessages([
        {
          text: `📄 Uploaded file: ${selectedFile.name}`,
          sender: "bot",
          timestamp: new Date(),
        },
        ...botMessages,
      ]);

      const previewRes = await fetch(`${API}/raw_preview`);
      const previewData = await previewRes.json();
      setRawPreview(previewData);
    } finally {
      setIsUploading(false);
    }
  };

  // ---------------- Chat ----------------
  const handleSend = async (e) => {
    e.preventDefault();
    if (!inputText.trim()) return;

    const userMsg = {
      text: inputText,
      sender: "user",
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMsg]);
    setInputText("");
    setIsLoading(true);

    try {
      const res = await fetch(`${API}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userMsg.text }),
      });

      const data = await res.json();

      setMessages((prev) => [
        ...prev,
        {
          text: data.reply,
          sender: "bot",
          timestamp: new Date(),
        },
      ]);

      // ---- TRY TRAIN RESULTS ----
      try {
        const tr = await fetch(`${API}/train_results`);
        if (tr.ok) {
          const tData = await tr.json();
          setTrainResult(tData);
        }
      } catch {}

      // ---- TRY FORECAST RESULTS ----
      try {
        const fr = await fetch(`${API}/forecast_results`);
        if (fr.ok) {
          const fData = await fr.json();
          setForecastResult(fData);
        }
      } catch {}
    } finally {
      setIsLoading(false);
    }
  };

  // ---------------- Table Renderer ----------------
  const renderTable = (columns, rows) => (
    <table>
      <thead>
        <tr>
          {columns.map((c) => (
            <th key={c}>{c}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map((r, i) => (
          <tr key={i}>
            {columns.map((c) => (
              <td key={c}>{r[c]}</td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );

  return (
    <div className="chat-container">
      <header className="chat-header">
        <h1>AI Assistant</h1>
      </header>

      {/* Upload */}
      <div className="upload-bar">
        <input
          type="file"
          accept=".csv"
          onChange={(e) => setSelectedFile(e.target.files[0])}
          disabled={isUploading || isLoading}
        />
        <button onClick={handleUpload} disabled={!selectedFile || isUploading}>
          Upload CSV
        </button>
      </div>

      <div className="messages-area">
        {messages.map((m, i) => (
          <div key={i} className={`message-bubble ${m.sender}`}>
            {m.text}
          </div>
        ))}

        {/* RAW PREVIEW */}
        {rawPreview && (
          <div className="table-block">
            <h3>Raw data preview (first 15 rows)</h3>
            {renderTable(rawPreview.columns, rawPreview.rows)}
          </div>
        )}

        {/* TRAIN RESULTS */}
        {trainResult && (
          <div className="table-block">
            <h3>Training leaderboard</h3>
            {renderTable(
              Object.keys(trainResult.leaderboard[0]),
              trainResult.leaderboard
            )}
            <p>{trainResult.explanation}</p>
          </div>
        )}

        {/* FORECAST RESULTS */}
        {forecastResult && (
          <div className="table-block">
            <h3>
              Forecast Results ({forecastResult.meta.ds_col} →{" "}
              {forecastResult.meta.y_col})
            </h3>
            {renderTable(
              ["Date", "Forecast", "Lower", "Upper"],
              forecastResult.preview
            )}
          </div>
        )}

        {isLoading && <div className="message-bubble bot">Typing…</div>}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <form className="input-area" onSubmit={handleSend}>
        <input
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          placeholder="Type a message..."
        />
        <button type="submit">▶</button>
      </form>
    </div>
  );
};

export default ChatInterface;
