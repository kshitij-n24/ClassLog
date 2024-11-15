import React, { useState } from "react";
import { FFmpeg } from "@ffmpeg/ffmpeg";
import "./App.css";

const App = () => {
  const [isProcessing, setIsProcessing] = useState(false);
  const [audioUrl, setAudioUrl] = useState(null);

  const handleFileChange = async (event) => {
    const videoFile = event.target.files[0];
    if (!videoFile) return;

    setIsProcessing(true);

    try {
      const ffmpeg = new FFmpeg();
      await ffmpeg.load();

      const videoData = await videoFile.arrayBuffer();

      ffmpeg.FS("writeFile", "input.mp4", new Uint8Array(videoData));
      await ffmpeg.run(
        "-i",
        "input.mp4",
        "-q:a",
        "0",
        "-map",
        "a",
        "output.mp3"
      );

      const audioData = ffmpeg.FS("readFile", "output.mp3");
      const audioBlob = new Blob([audioData.buffer], { type: "audio/mpeg" });
      const audioUrl = URL.createObjectURL(audioBlob);

      setAudioUrl(audioUrl);
    } catch (error) {
      console.error("Error processing video:", error);
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
        {isProcessing ? "Processing..." : "Generate Analysis"}
      </button>

      {audioUrl && (
        <div className="output">
          <audio controls>
            <source src={audioUrl} type="audio/mpeg" />
          </audio>
          <a href={audioUrl} download="output.mp3">
            Download Audio
          </a>
        </div>
      )}
    </div>
  );
};

export default App;
