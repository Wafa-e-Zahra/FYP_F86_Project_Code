import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import models, transforms

# --- CONFLICT-FREE FACE DETECTION ---
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

# Standard ImageNet transforms
img_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def crop_face(frame):
    """Detects and crops face using OpenCV Haar Cascades."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        margin = 20
        ih, iw, _ = frame.shape
        nx, ny = max(0, x - margin), max(0, y - margin)
        nw, nh = min(iw - nx, w + 2*margin), min(ih - ny, h + 2*margin)
        
        face = frame[ny:ny+nh, nx:nx+nw]
        if face is not None and face.size != 0:
            return face
    return frame

def process_image(img_path):
    """Loads an image, crops face, and returns a 4D tensor [1, 3, 224, 224]."""
    img = cv2.imread(img_path)
    if img is None:
        return torch.zeros(1, 3, 224, 224)
    
    face = crop_face(img)
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    return img_transform(face_rgb).unsqueeze(0)

# --- MODEL DEFINITION ---
class DeepfakeImageModel(nn.Module):
    def __init__(self, num_classes=2):
        super(DeepfakeImageModel, self).__init__()
        # Using EfficientNet-B0 for high-accuracy feature extraction
        base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        
        # Keep only the feature extraction layers
        self.feature_extractor = base.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # Simple classifier for single image (1280 is EfficientNet-B0's output size)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        # Input shape: [Batch, 3, 224, 224]
        x = self.feature_extractor(x)
        x = self.avgpool(x).flatten(1) 
        return self.classifier(x)