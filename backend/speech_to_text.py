import os
import torch
import whisper
import multiprocessing
from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS  # Import flask-cors
from pydub import AudioSegment
from pydub.utils import make_chunks
from moviepy.editor import VideoFileClip
from concurrent.futures import ProcessPoolExecutor

# Ensure multiprocessing compatibility
if multiprocessing.get_start_method(allow_none=True) != 'spawn':
    multiprocessing.set_start_method('spawn')

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

def extract_audio(video_path, audio_path="output_audio.mp3"):
    """
    Extracts audio from a video file and saves it as an MP3 file.
    """
    print(f"Extracting audio from {video_path}...")
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path)
    clip.close()
    print(f"Audio saved to {audio_path}")
    return audio_path


def split_audio(audio_path, output_dir="chunks", chunk_duration=600):
    """
    Splits the audio file into smaller chunks of a specified duration.
    """
    print(f"Splitting audio into chunks of {chunk_duration} seconds...")
    os.makedirs(output_dir, exist_ok=True)

    audio = AudioSegment.from_file(audio_path)
    chunks = make_chunks(audio, chunk_duration * 1000)  # chunk_duration in milliseconds

    chunk_paths = []
    for i, chunk in enumerate(chunks):
        chunk_path = os.path.join(output_dir, f"chunk_{i}.mp3")
        chunk.export(chunk_path, format="mp3")
        chunk_paths.append(chunk_path)

    print(f"Audio split into {len(chunk_paths)} chunks.")
    return chunk_paths


def transcribe_chunk(chunk_path, model_name="base"):
    """
    Transcribes a single audio chunk using Whisper, loading the model in each subprocess.
    """
    print(f"Transcribing {chunk_path}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(model_name).to(device)
    result = model.transcribe(chunk_path)
    print(f"Completed transcription for {chunk_path}")
    return result['text']


def transcribe_audio_chunks(chunk_paths, model, use_gpu=True):
    """
    Transcribes audio chunks in parallel using Whisper.
    """
    print("Starting transcription of audio chunks...")
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    model = whisper.load_model(model).to(device)

    transcripts = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(transcribe_chunk, chunk, model_name="base") for chunk in chunk_paths]
        for future in futures:
            transcripts.append(future.result())

    return transcripts


def save_transcripts(transcripts, output_file="transcript.txt"):
    """
    Saves the transcripts to a file incrementally.
    """
    print(f"Saving transcripts to {output_file}...")
    with open(output_file, "w") as f:
        for transcript in transcripts:
            f.write(transcript + "\n")
    print(f"Transcripts saved to {output_file}")
    return output_file


def stream_transcripts(file_path):
    """
    Streams a large transcript file in chunks to the client.
    """
    def generate():
        with open(file_path, "r") as f:
            while chunk := f.read(1024):  # Stream 1 KB at a time
                yield chunk
    return Response(generate(), content_type="text/plain")


@app.route('/upload', methods=['POST'])
def upload_lctrec():
    """
    Endpoint to handle video uploads and process transcripts.
    """
    try:
        print("Received a POST request to /upload")  # Log message when a POST request is received

        video = request.files.get('file')
        if not video:
            print("No video file uploaded")  # Log error if no video file is provided
            return jsonify({"error": "No video file uploaded"}), 400

        video_path = "temp_video.mp4"
        audio_path = "temp_audio.mp3"

        # Save the uploaded video
        video.save(video_path)
        print(f"Video saved to {video_path}")

        # Step 1: Extract audio
        audio_path = extract_audio(video_path)

        # Step 2: Split audio into chunks
        chunk_paths = split_audio(audio_path)

        # Step 3: Transcribe chunks
        transcripts = transcribe_audio_chunks(chunk_paths, model="base", use_gpu=True)

        # Step 4: Save transcripts
        transcript_file = save_transcripts(transcripts)

        # Cleanup temporary files
        os.remove(audio_path)
        os.remove(video_path)
        for chunk in chunk_paths:
            os.remove(chunk)

        print("Transcription process completed successfully")  # Log success message
        return jsonify({"message": "Transcription completed", "transcript_file": transcript_file}), 200

    except Exception as e:
        print(f"Error processing request: {str(e)}")  # Log error message
        return jsonify({"error": str(e)}), 500


@app.route('/download_transcript', methods=['GET'])
def download_transcript():
    """
    Endpoint to download the full transcript file.
    """
    transcript_file = "transcript.txt"
    if os.path.exists(transcript_file):
        return send_file(transcript_file, as_attachment=True)
    return jsonify({"error": "Transcript file not found"}), 404


@app.route('/stream_transcript', methods=['GET'])
def stream_transcript():
    """
    Endpoint to stream the transcript file in chunks.
    """
    transcript_file = "transcript.txt"
    if os.path.exists(transcript_file):
        return stream_transcripts(transcript_file)
    return jsonify({"error": "Transcript file not found"}), 404


if __name__ == "__main__":
    app.run(debug=True)