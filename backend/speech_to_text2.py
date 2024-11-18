import os
import torch
import whisper
import multiprocessing
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from pydub import AudioSegment
from pydub.utils import make_chunks
from moviepy.editor import VideoFileClip
from concurrent.futures import ProcessPoolExecutor
import json

# Ensure multiprocessing compatibility
if multiprocessing.get_start_method(allow_none=True) != 'spawn':
    multiprocessing.set_start_method('spawn')

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def extract_audio(video_path, audio_path="output_audio.mp3"):
    print(f"Extracting audio from {video_path}...")
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path)
    clip.close()
    print(f"Audio saved to {audio_path}")
    return audio_path

def split_audio(audio_path, output_dir="chunks", chunk_duration=600):
    print(f"Splitting audio into chunks of {chunk_duration} seconds...")
    os.makedirs(output_dir, exist_ok=True)

    audio = AudioSegment.from_file(audio_path)
    chunks = make_chunks(audio, chunk_duration * 1000)

    chunk_paths = []
    for i, chunk in enumerate(chunks):
        chunk_path = os.path.join(output_dir, f"chunk_{i}.mp3")
        chunk.export(chunk_path, format="mp3")
        chunk_paths.append(chunk_path)

    print(f"Audio split into {len(chunk_paths)} chunks.")
    return chunk_paths

def transcribe_chunk_with_timestamps(chunk_path, model_name="base"):
    print(f"Transcribing {chunk_path}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(model_name).to(device)
    result = model.transcribe(chunk_path)

    segments = []
    for segment in result['segments']:
        segments.append({
            "start": segment['start'],
            "end": segment['end'],
            "text": segment['text']
        })

    print(f"Completed transcription for {chunk_path}")
    return segments

def transcribe_audio_chunks_with_timestamps(chunk_paths, model="base", use_gpu=True):
    print("Starting transcription of audio chunks with timestamps...")
    transcripts = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(transcribe_chunk_with_timestamps, chunk, model_name=model) for chunk in chunk_paths]
        for future in futures:
            transcripts.extend(future.result())
    return transcripts

def save_transcripts_with_timestamps(transcripts, output_file="transcript_with_timestamps.json"):
    print(f"Saving transcripts with timestamps to {output_file}...")
    with open(output_file, "w") as f:
        json.dump(transcripts, f, indent=4)
    print(f"Transcripts saved to {output_file}")
    return output_file

@app.route('/upload_with_timestamps', methods=['POST'])
def upload_with_timestamps():
    try:
        print("Received a POST request to /upload_with_timestamps")

        video = request.files.get('file')
        if not video:
            return jsonify({"error": "No video file uploaded"}), 400

        video_path = "temp_video.mp4"
        audio_path = "temp_audio.mp3"

        # Save the uploaded video
        video.save(video_path)

        # Step 1: Extract audio
        audio_path = extract_audio(video_path)

        # Step 2: Split audio into chunks
        chunk_paths = split_audio(audio_path)

        # Step 3: Transcribe chunks with timestamps
        transcripts_with_timestamps = transcribe_audio_chunks_with_timestamps(chunk_paths, model="base", use_gpu=True)

        # Step 4: Save transcripts with timestamps
        transcript_file = save_transcripts_with_timestamps(transcripts_with_timestamps)

        # Cleanup temporary files
        os.remove(audio_path)
        os.remove(video_path)
        for chunk in chunk_paths:
            os.remove(chunk)

        return jsonify({"message": "Transcription completed", "transcript_file": transcript_file}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/download_transcript_with_timestamps', methods=['GET'])
def download_transcript_with_timestamps():
    transcript_file = "transcript_with_timestamps.json"
    if os.path.exists(transcript_file):
        return send_file(transcript_file, as_attachment=True)
    return jsonify({"error": "Transcript file not found"}), 404

if __name__ == "__main__":
    app.run(debug=True)
