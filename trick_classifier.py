import torch
import random

class TimeSformer:
    def __init__(self, model_path):
        self.model = torch.load(model_path)
        self.model.eval()

    def predict(self, video_frames):
        # Dummy function simulating prediction logic
        predictions = ["Kickflip", "Heelflip", "Ollie", "Pop Shuvit"]
        probabilities = [0.3, 0.2, 0.4, 0.1]
        predicted_trick = random.choices(predictions, probabilities)[0]
        return predicted_trick

def predict_flip(video_path):
    frames = extract_frames(video_path)
    model = TimeSformer('timesformer.pth')
    predicted_trick = model.predict(frames)
    flipped = predicted_trick in ["Kickflip", "Heelflip", "Treflip"]
    return predicted_trick, flipped

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frames.append(frame)
    cap.release()
    return frames
