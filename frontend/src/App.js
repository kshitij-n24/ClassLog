import React, { useState } from "react";
import "./App.css";

const App = () => {
  const [isProcessing, setIsProcessing] = useState(false);
  const [responseMessage, setResponseMessage] = useState(null);

  const handleFileChange = async (event) => {
    const videoFile = event.target.files[0];
    if (!videoFile) return;

    setIsProcessing(true);

    try {
      const formData = new FormData();
      formData.append("file", videoFile);

      // Send POST request to backend
      const response = await fetch("http://127.0.0.1:5000/upload", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Error: ${response.statusText}`);
      }

      const data = await response.json();
      setResponseMessage(data.message || "File uploaded successfully!");

      // If the transcript file is available, fetch and save it
      if (data.transcript_file) {
        const transcriptResponse = await fetch(
          "http://127.0.0.1:5000/download_transcript"
        );

        if (!transcriptResponse.ok) {
          throw new Error("Failed to fetch the transcript file.");
        }

        const transcriptText = await transcriptResponse.text();

        // Save the transcript as a text file
        const blob = new Blob([transcriptText], { type: "text/plain" });
        const link = document.createElement("a");
        link.href = URL.createObjectURL(blob);
        link.download = "transcript.txt";
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);

        setResponseMessage(
          `Transcription completed. Transcript saved as "transcript.txt".`
        );
      }
    } catch (error) {
      console.error("Error uploading video:", error);
      setResponseMessage("Failed to upload video. Please try again.");
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="container">
      <h1 className="title">ClassLog</h1>
      <h2 className="subtitle">Algebra</h2>
      <h3 className="lecture">Lecture - 1</h3>
      <p className="grade">Grade Level: 4th Standard</p>

      <div className="upload-box">
        <input
          type="file"
          accept="video/*"
          id="fileInput"
          className="file-input"
          onChange={handleFileChange}
        />
        <label htmlFor="fileInput" className="file-label">
          <i className="fas fa-upload"></i> Drag and drop or browse files
        </label>
      </div>

      <button
        className="process-button"
        disabled={isProcessing}
        onClick={() => document.getElementById("fileInput").click()}
      >
        {isProcessing ? "Processing..." : "Upload Video"}
      </button>

      {responseMessage && (
        <div className="response-message">
          <p>{responseMessage}</p>
        </div>
      )}
    </div>
  );
};

export default App;
