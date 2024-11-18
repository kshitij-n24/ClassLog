from inference_sdk import InferenceHTTPClient
from flask import Flask, request, jsonify, redirect, url_for
import cv2
import torch
from torchvision import transforms
from collections import defaultdict
import os
from dotenv import load_dotenv

app = Flask(__name__)

load_dotenv()

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
QUESTION_INTERVAL = 10
# model = torch.hub.load('WongKinYiu/yolov7', 'custom', '../../yolov7.pt', trust_repo=True)
# model.eval()

# Define transformations if required by the model
transform = transforms.Compose([
    transforms.ToTensor(),
])

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


def process_video(video_path, timestamps):
    """Processes the video to detect raised hands and analyze question responses."""
    cap = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise Exception(f"Failed to open video file: {video_path}")
    
    question_results = defaultdict(lambda: {'yes': 0, 'no': 0})
    frame_count = 0
    question_count = 1

    
    # Get video properties
    fps = video.get(cv2.CAP_PROP_FPS)  # Frames per second
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Map each timestamp to its corresponding frame index
    timestamp_to_frame = {}
    for timestamp in timestamps:
        frame_index = int(fps * timestamp)
        if 0 <= frame_index < total_frames:
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = video.read()
            if success:
                timestamp_to_frame[timestamp] = frame
            else:
                timestamp_to_frame[timestamp] = None  # Mark as None if frame extraction failed
        else:
            timestamp_to_frame[timestamp] = None 

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        

        # Process the questions_for_revision and questions_completed
        questions_for_revision = []
        questions_completed = []
        
        frame_count += 1

    cap.release()

    # Analyze results for revision or completion
    

    return {
        "questions_for_revision": questions_for_revision,
        "questions_completed": questions_completed
    }

@app.route('')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and process the input."""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Check if the file is a video or an image
    ext = os.path.splitext(file.filename)[1].lower()
    if ext in ['.mp4', '.avi', '.mov', '.mkv']:
        # Process video
        result = process_video(file_path)
    elif ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        # Process single image
        frame = cv2.imread(file_path)
        hand_raised_count = process_frame(frame)
        result = {"hand_raised_count": hand_raised_count}
    else:
        return jsonify({"error": "Unsupported file format"}), 400

    return jsonify(result)

@app.route('/get_transcript', methods=['GET'])
def 

if __name__ == '__main__':
    app.run(debug=True)
