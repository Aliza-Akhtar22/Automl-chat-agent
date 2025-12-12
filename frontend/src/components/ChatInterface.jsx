// ChatInterface.jsx
import React, { useState, useRef, useEffect } from "react";
import "./ChatInterface.css";

const ChatInterface = () => {
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);

  const [rawPreview, setRawPreview] = useState(null);
  const [cleanPreview, setCleanPreview] = useState(null);

  // training results (one entry per training run)
  const [trainResults, setTrainResults] = useState([]);

  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, rawPreview, cleanPreview, trainResults]);

  // ---------- Simple markdown formatting ----------
  const renderText = (text) => {
    if (!text) return "";

    // strip special markers so user doesn't see them
    let safe = text
      .replace(/__show_preprocessed_preview__/gi, "")
      .replace(/__show_training_results__/gi, "");

    return safe
      .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\n/g, "<br />");
  };

  // ---------- CSV upload ----------
  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      alert("Please choose a CSV file first.");
      return;
    }

    setIsUploading(true);
    setRawPreview(null);
    setCleanPreview(null);
    setTrainResults([]); // reset ALL old training results on new upload

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const res = await fetch("http://127.0.0.1:8000/upload_csv", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) throw new Error("Upload failed");

      const data = await res.json();

      setMessages((prev) => [
        ...prev,
        {
          text: `📄 Uploaded file: ${selectedFile.name}`,
          sender: "bot",
          timestamp: new Date(),
        },
        {
          text: data.reply,
          sender: "bot",
          timestamp: new Date(),
        },
      ]);

      await fetchRawPreview();
    } catch (err) {
      console.error(err);
      setMessages((prev) => [
        ...prev,
        {
          text: "Sorry, I couldn't upload the CSV.",
          sender: "bot",
          isError: true,
          timestamp: new Date(),
        },
      ]);
    } finally {
      setIsUploading(false);
    }
  };

  // ---------- RAW preview ----------
  const fetchRawPreview = async () => {
    try {
      const res = await fetch("http://127.0.0.1:8000/raw_preview");
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setRawPreview(data);
    } catch (err) {
      console.error("Error fetching raw preview:", err);
    }
  };

  // ---------- PREPROCESSED preview ----------
  const fetchCleanPreview = async () => {
    try {
      const res = await fetch("http://127.0.0.1:8000/preview");
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setCleanPreview(data);
    } catch (err) {
      console.error("Error fetching preprocessed preview:", err);
    }
  };

  // ---------- TRAINING RESULTS (leaderboard) ----------
  const fetchTrainResults = async () => {
    try {
      const res = await fetch("http://127.0.0.1:8000/train_results");
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();

      // data = { leaderboard: [...], explanation: "..." }  OR just [...]
      const rows = Array.isArray(data)
        ? data
        : Array.isArray(data.leaderboard)
        ? data.leaderboard
        : [];

      const columns = rows.length ? Object.keys(rows[0]) : [];
      const explanation = data.explanation || null;

      // append this training run's results
      setTrainResults((prev) => [...prev, { columns, rows, explanation }]);
    } catch (err) {
      console.error("Error fetching training results:", err);
    }
  };

  // ---------- DOWNLOADS ----------
  const PREPROCESSED_DOWNLOAD_URL =
    "http://127.0.0.1:8000/download_preprocessed";
  const MODEL_DOWNLOAD_URL =
    "http://127.0.0.1:8000/download_best_model";

  const handleDownloadPreprocessed = async () => {
    try {
      const res = await fetch(PREPROCESSED_DOWNLOAD_URL, {
        method: "GET",
      });
      if (!res.ok) throw new Error("Failed to download preprocessed data");

      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "preprocessed_data.csv";
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      console.error(err);
      alert("Could not download the preprocessed data. Please try again.");
    }
  };

  const handleDownloadBestModel = async () => {
    try {
      const res = await fetch(MODEL_DOWNLOAD_URL, {
        method: "GET",
      });
      if (!res.ok) throw new Error("Failed to download best model");

      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "best_model.pkl";
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      console.error(err);
      alert("Could not download the best model. Please try again.");
    }
  };

  // ---------- Chat ----------
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
      const res = await fetch("http://127.0.0.1:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userMsg.text }),
      });

      if (!res.ok) throw new Error("Network error");

      const data = await res.json();
      let reply = data.reply || data.message || "Received response";

      const lowerReply = reply.toLowerCase();
      const lowerUser = userMsg.text.toLowerCase();

      // Add bot reply to chat
      setMessages((prev) => [
        ...prev,
        { text: reply, sender: "bot", timestamp: new Date() },
      ]);

      // ----- PREPROCESSED PREVIEW -----
      if (lowerReply.includes("__show_preprocessed_preview__")) {
        await fetchCleanPreview();
      }

      // also allow user-triggered preview
      if (
        lowerUser.includes("preview") ||
        lowerUser.includes("show data") ||
        lowerUser.includes("show table")
      ) {
        await fetchCleanPreview();
      }

      // ----- TRAINING RESULTS / LEADERBOARD -----
      if (lowerReply.includes("__show_training_results__")) {
        await fetchTrainResults();
      }

      // optional: if user explicitly asks for leaderboard / metrics
      if (
        lowerUser.includes("leaderboard") ||
        lowerUser.includes("metrics") ||
        lowerUser.includes("training results")
      ) {
        await fetchTrainResults();
      }
    } catch (err) {
      console.error(err);
      setMessages((prev) => [
        ...prev,
        {
          text: "Sorry, I couldn't reach the server.",
          sender: "bot",
          isError: true,
          timestamp: new Date(),
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  // ---------- Render table (generic) ----------
  const renderTable = (preview) => {
    if (!preview) return null;
    return (
      <table>
        <thead>
          <tr>
            {preview.columns.map((col) => (
              <th key={col}>{col}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {preview.rows.map((row, idx) => (
            <tr key={idx}>
              {preview.columns.map((col) => (
                <td key={col}>{row[col]}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    );
  };

  // ---------- Map each training message to its own result ----------
  const getTrainingResultIndex = (messageIndex) => {
    let count = 0;
    for (let i = 0; i <= messageIndex; i++) {
      const t = (messages[i].text || "").toLowerCase();
      if (t.includes("__show_training_results__")) {
        count += 1;
      }
    }
    return count - 1; // 0-based index into trainResults
  };

  return (
    <div className="chat-container">
      <header className="chat-header">
        <h1>AI Assistant</h1>
      </header>

      {/* Upload bar */}
      <div className="upload-bar">
        <input
          type="file"
          accept=".csv"
          onChange={handleFileChange}
          disabled={isUploading || isLoading}
        />
        <button
          className="upload-btn"
          onClick={handleUpload}
          disabled={isUploading || !selectedFile}
        >
          {isUploading ? "Uploading…" : "Upload CSV"}
        </button>
      </div>

      {/* Chat + tables */}
      <div className="messages-area">
        {messages.length === 0 && (
          <div className="empty-state">
            <p>👋 Upload a CSV to get started.</p>
          </div>
        )}

        {messages.map((msg, index) => {
          const text = msg.text || "";
          const isUploadMsg = text.startsWith("📄 Uploaded file:");

          const lowerText = text.toLowerCase();
          const isTrainingMessage =
            lowerText.includes("__show_training_results__");
          const trainingIdx = isTrainingMessage
            ? getTrainingResultIndex(index)
            : -1;
          const thisTrainResult =
            trainingIdx >= 0 && trainingIdx < trainResults.length
              ? trainResults[trainingIdx]
              : null;

          return (
            <React.Fragment key={index}>
              {/* Message bubble */}
              <div
                className={`message-bubble ${msg.sender} ${
                  msg.isError ? "error" : ""
                }`}
              >
                <div
                  className="message-content"
                  dangerouslySetInnerHTML={{ __html: renderText(text) }}
                />
                <div className="message-time">
                  {msg.timestamp.toLocaleTimeString([], {
                    hour: "2-digit",
                    minute: "2-digit",
                  })}
                </div>
              </div>

              {/* Raw Preview after upload */}
              {isUploadMsg && rawPreview && (
                <div className="table-block">
                  <div className="table-title">
                    Raw data preview (first 15 rows)
                  </div>
                  <div className="table-container">
                    {renderTable(rawPreview)}
                  </div>
                </div>
              )}

              {/* Preprocessed Preview only when explicitly triggered */}
              {lowerText.includes("__show_preprocessed_preview__") &&
                cleanPreview && (
                  <div className="table-block">
                    <div className="table-title">
                      Preprocessed data preview (first 15 rows)
                    </div>
                    <div className="table-container">
                      {renderTable(cleanPreview)}
                    </div>

                    {/* download button for preprocessed data */}
                    <button
                      className="download-button"
                      onClick={handleDownloadPreprocessed}
                    >
                      Download preprocessed data (CSV)
                    </button>
                  </div>
                )}

              {/* Training leaderboard for THIS message's training run */}
              {isTrainingMessage && thisTrainResult && (
                <div className="table-block">
                  <div className="table-title">Training leaderboard</div>
                  <div className="table-container">
                    {renderTable(thisTrainResult)}
                  </div>

                  {/* download button for best model */}
                  <button
                    className="download-button"
                    onClick={handleDownloadBestModel}
                  >
                    Download best model (.pkl)
                  </button>

                  {thisTrainResult.explanation && (
                    <div
                      className="training-explanation"
                      dangerouslySetInnerHTML={{
                        __html: renderText(thisTrainResult.explanation),
                      }}
                    />
                  )}
                </div>
              )}
            </React.Fragment>
          );
        })}

        {isLoading && (
          <div className="message-bubble bot loading">
            <div className="typing-dots">
              <span></span>
              <span></span>
              <span></span>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <form className="input-area" onSubmit={handleSend}>
        <input
          type="text"
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          placeholder="Type a message..."
          disabled={isLoading}
        />
        <button type="submit" disabled={isLoading || !inputText.trim()}>
          <svg viewBox="0 0 24 24" width="24" height="24" fill="currentColor">
            <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
          </svg>
        </button>
      </form>
    </div>
  );
};

export default ChatInterface;
