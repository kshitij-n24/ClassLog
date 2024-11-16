from flask import Flask, request, jsonify, redirect, url_for
import cv2
import torch
from torchvision import transforms
from collections import defaultdict
import os

app = Flask(__name__)

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
model = torch.hub.load('WongKinYiu/yolov7', 'custom', '../../yolov7.pt', trust_repo=True)
model.eval()

# Define transformations if required by the model
transform = transforms.Compose([
    transforms.ToTensor(),
])


def process_frame(frame):
    """Process a single frame for hand raise detection."""
    input_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = transform(input_frame).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        outputs = model(input_tensor)

    print(input_tensor)

    # Ensure the outputs are in the expected format
    if isinstance(outputs, tuple):
        outputs = outputs[0]  # Extract the relevant data if wrapped in a tuple

    # Count the number of raised hands detected in the frame
    hand_raised_count = 0
    for output in outputs.xyxy[0]:  # Adjust according to actual output structure
        x1, y1, x2, y2, conf, cls = output
        if int(cls) == 0:  # Replace 0 with the correct class index for 'hand_raised'
            hand_raised_count += 1

    return hand_raised_count


def process_video(video_path):
    """Processes the video to detect raised hands and analyze question responses."""
    cap = cv2.VideoCapture(video_path)
    question_results = defaultdict(lambda: {'yes': 0, 'no': 0})
    frame_count = 0
    question_count = 1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        hand_raised_count = process_frame(frame)

        # Collect responses every QUESTION_INTERVAL frames
        if frame_count % QUESTION_INTERVAL == 0:
            total_people = 10  # Adjust this based on your use case
            if hand_raised_count / total_people >= THRESHOLD_CORRECT:
                question_results[question_count]['yes'] += 1
            elif hand_raised_count / total_people <= THRESHOLD_WRONG:
                question_results[question_count]['no'] += 1
            question_count += 1

        frame_count += 1

    cap.release()

    # Analyze results for revision or completion
    questions_for_revision = []
    questions_completed = []

    for question, results in question_results.items():
        total_responses = results['yes'] + results['no']
        if total_responses == 0:
            continue

        yes_ratio = results['yes'] / total_responses
        if yes_ratio >= THRESHOLD_CORRECT:
            questions_completed.append(question)
        elif yes_ratio <= THRESHOLD_WRONG:
            questions_for_revision.append(question)

    return {
        "questions_for_revision": questions_for_revision,
        "questions_completed": questions_completed
    }


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


if __name__ == '__main__':
    app.run(debug=True)
