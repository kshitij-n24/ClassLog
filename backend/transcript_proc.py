import os
import cv2
import torch
import json
import whisper
import os
from collections import defaultdict
from inference_sdk import InferenceHTTPClient
from torchvision import transforms
from dotenv import load_dotenv
import multiprocessing
from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS  # Import flask-cors
from pydub import AudioSegment
from pydub.utils import make_chunks
from moviepy.editor import VideoFileClip
from concurrent.futures import ProcessPoolExecutor
import google.generativeai as genai
from google.generativeai.types import RequestOptions
from google.api_core import retry
from pymongo import MongoClient
from bson.json_util import dumps
from bson.objectid import ObjectId

# Ensure multiprocessing compatibility
if multiprocessing.get_start_method(allow_none=True) != 'spawn':
    multiprocessing.set_start_method('spawn')

load_dotenv()

# Initialize Flask app
app = Flask(__name__)

mongo_client = MongoClient(os.getenv("MONGO_URI"))
db = mongo_client['video_processing_db']
transcripts_collection = db['transcripts']
results_collection = db['results']

robo_key = os.getenv("ROBOFLOW_KEY")

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=robo_key
)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload and processed folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Parameters for hand raise detection and question analysis
THRESHOLD_CORRECT = 0.7
THRESHOLD_WRONG = 0.3

# Define transformations if required by the model
transform = transforms.Compose([
    transforms.ToTensor(),
])

genai.configure(api_key=os.getenv("GEMINI_KEY"))

gemini_model = genai.GenerativeModel("gemini-1.5-flash")

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


# def transcribe_chunk(chunk_path, model_name="base"):
#     """
#     Transcribes a single audio chunk using Whisper, loading the model in each subprocess.
#     """
#     print(f"Transcribing {chunk_path}..., running on device: {device}")
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model = whisper.load_model(model_name).to(device)
#     result = model.transcribe(chunk_path)
#     print(f"Completed transcription for {chunk_path}")
#     return result['text']


# def transcribe_audio_chunks(chunk_paths, model, use_gpu=True):
#     """
#     Transcribes audio chunks in parallel using Whisper.
#     """
#     print("Starting transcription of audio chunks...")
#     device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
#     model = whisper.load_model(model).to(device)

#     transcripts = []
#     with ProcessPoolExecutor(max_workers=4) as executor:
#         futures = [executor.submit(transcribe_chunk, chunk, model_name="base") for chunk in chunk_paths]
#         for future in futures:
#             transcripts.append(future.result())

#     return transcripts

def transcribe_chunk_with_timestamps(chunk_path, model_name="base"):
    print(f"Transcribing {chunk_path}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(model_name).to(device)
    result = model.transcribe(chunk_path)
    chunk_number = int(chunk_path.rsplit('_', 1)[-1].rsplit('.',1)[0])
    print(f"Chunk number {chunk_number}")

    segments = []
    for segment in result['segments']:
        segments.append({
            "start": (chunk_number*600)+segment['start'],
            "end": (chunk_number*600)+segment['end'],
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

def save_transcripts_with_timestamps(transcripts, output_file_txt="transcript.txt", output_file_json="transcript_with_timestamps.json"):
    print(f"Saving transcripts with timestamps to {output_file_json}...")
    with open(output_file_json, "w") as f:
        json.dump(transcripts, f, indent=4)

    with open(output_file_txt, "w") as f:
            f.write(str(transcripts) + "\n")
    print(f"Transcripts saved to {output_file_txt} and {output_file_json}")
    return output_file_txt


def stream_transcripts(file_path):
    """
    Streams a large transcript file in chunks to the client.
    """
    def generate():
        with open(file_path, "r") as f:
            while chunk := f.read(1024):  # Stream 1 KB at a time
                yield chunk
    return Response(generate(), content_type="text/plain")

def process_frame(frame):
    """Process a single frame for hand raise detection."""
    input_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    print('Fetching results from API')
    result = CLIENT.infer(input_frame, model_id="hand-raise-v1m/20")
    print('Retrieved the results')

    hand_raised_count = 0
    for boxes in result['predictions']:
        if boxes['class_id'] == 0:
            hand_raised_count += 1

    return hand_raised_count


def process_video(video_path):
    """Processes the video to detect raised hands and analyze question responses."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Failed to open video file: {video_path}")
    
    total_students = 40    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize result dictionaries
    question_results = defaultdict(lambda: {'yes': 0, 'no': 0, 'not_answered': total_students})
    questions_for_revision = []
    questions_completed = []

    quiz_locs = analyze_transcript()

    # Process each question and timestamp range
    for question_data in quiz_locs["questions"]:
        for question, (start_time, end_time) in question_data.items():
            start_frame = int(fps * start_time)
            end_frame = int(fps * end_time)

            yes_count, no_count = 0, 0

            for frame_index in range(start_frame, end_frame + 1):
                if 0 <= frame_index < total_frames:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Replace this with the actual model call for hand-raising detection
                    hand_raised_count = process_frame(frame)  # Example function
                    # Assume half the class says "yes," and half "no" for demo purposes
                    yes_count += hand_raised_count
                    no_count += total_students - hand_raised_count

            # Adjust not_answered based on detected responses
            not_answered = max(0, total_students - yes_count - no_count)
            question_results[question] = {
                "yes": yes_count,
                "no": no_count,
                "not_answered": not_answered
            }

            # Categorize question
            if yes_count / total_students >= 0.8:  # Example threshold for completion
                questions_completed.append(question)
            else:
                questions_for_revision.append(question)

    cap.release()

    return {
        "questions_for_revision": questions_for_revision,
        "questions_completed": questions_completed,
        "detailed_results": dict(question_results)
    }


#TODO: generalize transcript names
def analyze_transcript():
    with open('transcript_with_timestamps.json', 'r', encoding='utf-8') as file:
        file_contents = file.read()

    with open('temp_analyze_transcript.txt', "w") as f:
            f.write(file_contents + "\n")

    # for f in genai.list_files():
    #     if f.name != 'temp_analyze_transcript.txt':
    temp_upload = genai.upload_file('temp_analyze_transcript.txt')
    # upload_transcript_file = genai.upload_file("transcript_with_timestamps.json")
    result = gemini_model.generate_content(
        [temp_upload, "\n\n\n\n", """Analyze this transcript and give me the time stamp in JSON format, 
        where questions are being asked.
        Use this JSON schema:


        QuestionTimestamps = Dict(str: List[Dict(str: List(int, int))])

        Example:
        QuestionTimestamps = \{'questions':  [\{'Is the 2+2=4?': [15.222, 17.222]\}, \{'Is the capital of France, Pairs?', [254.506, 258.990]\}\}

        Return: QuestionTimestamps


        (The first timestamp if for answering yes, and the second timestamp is for answering no).

        """], request_options=RequestOptions(retry=retry.Retry(initial=10, multiplier=2, maximum=60, timeout=300))
    )
    print(f"{result.text}")
    res_str = str(result.text)
    print(type(res_str))
    res_str = res_str.split('\n',1)[1][:-4]
    res_json = json.loads(res_str)


    os.remove('temp_analyze_transcript.txt')

    return res_json


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

        # video_path = "temp_video.mp4"
        # audio_path = "temp_audio.mp3"

        # Save the uploaded video
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video.filename)
        video.save(video_path)
        print(f"Video saved to {video_path}")

        # Step 1: Extract audio
        audio_path = extract_audio(video_path)

        # Step 2: Split audio into chunks
        chunk_paths = split_audio(audio_path)

        # Step 3: Transcribe chunks
        # transcripts = transcribe_audio_chunks(chunk_paths, model="base", use_gpu=True)
        timestamp_transcript = transcribe_audio_chunks_with_timestamps(chunk_paths, model="base", use_gpu=True)

        # Step 4: Save transcripts
        transcript_file = save_transcripts_with_timestamps(timestamp_transcript)

        ext = os.path.splitext(video.filename)[1].lower()
        if ext in ['.mp4', '.avi', '.mov', '.mkv']:
        # Process video
            result = process_video(video_path)

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

@app.route('/download_transcript_with_timestamps', methods=['GET'])
def download_transcript_with_timestamps():
    transcript_file = "transcript_with_timestamps.json"
    if os.path.exists(transcript_file):
        return send_file(transcript_file, as_attachment=True)
    return jsonify({"error": "Transcript file not found"}), 404

if __name__ == "__main__":
    app.run(debug=True)
