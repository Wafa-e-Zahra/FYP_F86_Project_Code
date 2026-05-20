import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, models
from transformers import Wav2Vec2Model
from moviepy.video.io.VideoFileClip import VideoFileClip 

class FaceCropper:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def get_face(self, frame):
        if frame is None: return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
        if len(faces) > 0:
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            x, y, w, h = faces[0]
            return frame[y:y+h, x:x+w]
        return None

cropper = FaceCropper()

class VideoFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.features = vgg.features
        self.avgpool = vgg.avgpool
        self.vgg_classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        b, f, c, h, w = x.shape
        x = x.view(b * f, c, h, w)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.vgg_classifier(x)
        return x.view(b, f, -1)

class AudioFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        outputs = self.model(x).last_hidden_state
        return outputs.mean(dim=1)

class DeepfakeClassifier(nn.Module):
    def __init__(self, video_dim=4096, audio_dim=768):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(video_dim + audio_dim, 512), 
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2)
        )

    def forward(self, v, a):
        v_mean = v.mean(dim=1)
        combined = torch.cat((v_mean, a), dim=1)
        return self.fusion(combined)

# --- THE MISSING FUNCTION ---
def analyze_video_detailed(path, segments=5, frames_per_segment=8):
    """
    Splits the video into time-based segments so the report can say 
    EXACTLY which seconds are fake.
    """
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    analysis_data = []
    # Pick 5 different spots in the video to check
    indices = np.linspace(0, max(0, total_frames - frames_per_segment), segments, dtype=int)
    
    for start_idx in indices:
        frames = []
        timestamp = round(start_idx / fps, 2) if fps > 0 else 0
        
        for i in range(frames_per_segment):
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx + i)
            ret, frame = cap.read()
            if not ret: break
            
            face = cropper.get_face(frame)
            img = face if face is not None else frame
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(transform(img))
            
        if len(frames) == frames_per_segment:
            tensor = torch.stack(frames).unsqueeze(0)
            analysis_data.append({
                'time': f"{timestamp}s",
                'tensor': tensor
            })
            
    cap.release()
    return analysis_data

def process_audio(path, sample_rate=16000, duration=2):
    try:
        video = VideoFileClip(path)
        if video.audio is None: return torch.zeros((1, sample_rate * duration))
        audio_sub = video.audio.subclip(0, min(duration, video.duration))
        audio_data = audio_sub.to_soundarray(fps=sample_rate)
        if audio_data.ndim > 1: audio_data = audio_data.mean(axis=1)
        target_len = sample_rate * duration
        audio_data = np.pad(audio_data, (0, max(0, target_len - len(audio_data))))[:target_len]
        video.close()
        return torch.tensor(audio_data).float().unsqueeze(0)
    except:
        return torch.zeros((1, sample_rate * duration))