import os
import json
import torch
import cv2
import matplotlib.pyplot as plt
import torchaudio
import torchaudio.transforms as transforms
import torch.nn as nn
import numpy as np
import asyncio
import websockets
import base64
import random
from torchvision import models
from ultralytics import YOLO
import simpleaudio as sa
from glob import glob

# Paths to the models
MODEL_PATHS = {
    "image": "Drone-UI/models/image.pt",
    "thermal": "Drone-UI/models/thermal.pt",
    "audio": "Drone-UI/models/screaming_detector_gpu.pth"
}

# Paths to test data
TEST_DATA_PATHS = {
    "image": {
        "Human Detected": "Drone-UI/test_data/image/human/",
        "No human detected": "Drone-UI/test_data/image/no_human/"
    },
    "thermal": {
        "Human Detected": "Drone-UI/test_data/thermal/human/",
        "No human detected": "Drone-UI/test_data/thermal/no_human/"
    },
    "audio": {
        "Human Detected": "Drone-UI/test_data/audio/human/",
        "No human detected": "Drone-UI/test_data/audio/no_human/"
    }
}

# Lazy-loaded models
yolo_models = {}
audio_model = None

# Load YOLO model dynamically
def load_yolo_model(model_type):
    global yolo_models
    if model_type not in yolo_models:
        print(f"🔄 Loading {model_type} model...")
        yolo_models[model_type] = YOLO(MODEL_PATHS[model_type])
    return yolo_models[model_type]

# Audio classification model
class AudioClassifier(nn.Module):
    def __init__(self):
        super(AudioClassifier, self).__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, 2)

    def forward(self, x):
        return self.resnet(x)

# Load audio model dynamically
def load_audio_model():
    global audio_model
    if audio_model is None:
        print("🔄 Loading audio model...")
        audio_model = AudioClassifier()
        audio_model.load_state_dict(torch.load(MODEL_PATHS["audio"]))
        audio_model.eval()
    return audio_model

# Function to preprocess audio
def preprocess_audio(file_path):
    waveform, sr = torchaudio.load(file_path)
    target_sample_rate = 22050
    num_samples = target_sample_rate * 3  # 3 seconds

    if sr != target_sample_rate:
        waveform = torchaudio.transforms.Resample(sr, target_sample_rate)(waveform)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if waveform.shape[1] > num_samples:
        waveform = waveform[:, :num_samples]
    else:
        pad = num_samples - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad))

    mel_spec = transforms.MelSpectrogram(sample_rate=target_sample_rate, n_mels=64, n_fft=1024, hop_length=512)(waveform)
    mel_spec = torch.log(mel_spec + 1e-9).squeeze(0)
    return mel_spec

# Function to classify audio
def classify_audio(file_path):
    model = load_audio_model()
    mel_spec = preprocess_audio(file_path).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        output = model(mel_spec)
        predicted_class = torch.argmax(output, dim=1).item()
    class_labels = ["No Human", "Human Detected"]
    print(f"🔊 Audio Classification Result: {class_labels[predicted_class]}")
    return class_labels[predicted_class]

# Function to detect and plot bounding boxes for images
def plot_bboxes(image_path, model):
    img = cv2.imread(image_path)
    results = model(image_path)
    detected_img = results[0].plot()

    # Encode image as base64 for WebSocket transmission
    _, buffer = cv2.imencode(".png", detected_img)
    encoded_img = base64.b64encode(buffer).decode()

    return encoded_img

# Function to select a random test file
def get_test_file(model_type, human_status):
    folder = TEST_DATA_PATHS[model_type][human_status]
    files = glob(os.path.join(folder, "*.wav" if model_type == "audio" else "*.jpg"))
    return random.choice(files) if files else None

# WebSocket handler
async def websocket_handler(websocket):
    print("🔌 WebSocket connection established")
    async for message in websocket:
        try:
            data = json.loads(message)  # Expecting JSON object
            model_type = data["model"]
            human_status = "Human Detected"

            print(f"📩 Received detection request: {model_type}")

            if model_type in ["image", "thermal"]:
                model = load_yolo_model(model_type)
                test_file = get_test_file(model_type, human_status)

                if test_file:
                    encoded_img = plot_bboxes(test_file, model)
                    response = json.dumps({"type": model_type, "image": encoded_img})
                    await websocket.send(response)  # Send JSON response

            elif model_type == "audio":
                test_file = get_test_file(model_type, human_status)

                if test_file:
                    result = classify_audio(test_file)
                    response = json.dumps({"type": "audio", "result": result})
                    await websocket.send(response)  # Send JSON response

        except Exception as e:
            print(f"❗ Error: {e}")

# Start WebSocket Server
async def main():
    server = await websockets.serve(websocket_handler, "localhost", 8765)
    print("✅ WebSocket server started on ws://localhost:8765")
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())
