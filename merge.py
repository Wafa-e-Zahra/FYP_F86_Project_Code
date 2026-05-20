import os, base64, sqlite3, cv2, torch, datetime
import numpy as np
import torch.nn.functional as F
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from PIL import Image
from torchvision import transforms

# --- IMPORT ARCHITECTURES FROM YOUR LOCAL FILES ---
from model_image import DeepfakeImageModel
from model2 import (VideoFeatureExtractor, AudioFeatureExtractor, 
                    DeepfakeClassifier, analyze_video_detailed, process_audio)

app = Flask(__name__)
app.secret_key = 'forensic_lab_secure_key_2026'
UPLOAD_FOLDER = 'static/uploads'
REPORTS_FOLDER = 'static/reports'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. MODEL LOADING ---
image_model = DeepfakeImageModel().to(DEVICE)
if os.path.exists("Image_best_robust.pth"):
    image_model.load_state_dict(torch.load("Image_best_robust.pth", map_location=DEVICE))
    image_model.eval()

v_net = VideoFeatureExtractor().to(DEVICE)
a_net = AudioFeatureExtractor().to(DEVICE)
v_classifier = DeepfakeClassifier().to(DEVICE)

if os.path.exists("video_model_best.pth"):
    checkpoint = torch.load("video_model_best.pth", map_location=DEVICE)
    v_net.load_state_dict(checkpoint['v_net'])
    a_net.load_state_dict(checkpoint['a_net'])
    v_classifier.load_state_dict(checkpoint['classifier'])
    v_net.eval(); a_net.eval(); v_classifier.eval()

# --- 2. FORENSIC UTILITIES ---
def get_db_connection():
    conn = sqlite3.connect('users.db')
    conn.row_factory = sqlite3.Row
    return conn

def generate_forensic_report(filename, result, confidence, a_type, timeline=None):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_name = f"report_{secure_filename(filename)}.txt"
    report_path = os.path.join(REPORTS_FOLDER, report_name)
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("┌──────────────────────────────────────────────────────────┐\n")
        f.write("│          DEEPFAKEGUARD  FORENSIC ANALYSIS          │\n")
        f.write("│                PROPRIETARY DATA - SECURE                 │\n")
        f.write("└──────────────────────────────────────────────────────────┘\n\n")
        f.write(f"TIMESTAMP: {timestamp} | FILE: {filename}\n")
        f.write(f"MODE:      {a_type.upper()}_SCAN\n")
        f.write("-" * 60 + "\n")
        f.write(f"VERDICT:    [{result}]\n")
        f.write(f"CONFIDENCE: {confidence}%\n")
        f.write("-" * 60 + "\n\n")

        if timeline:
            f.write("TIMELINE ANALYSIS:\n")
            f.write(f"{'SEC':<8} | {'PROBABILITY':<15} | {'STATUS'}\n")
            f.write("-" * 60 + "\n")
            for entry in timeline:
                status_icon = "[!] ALERT" if entry['status'] == 'FAKE' else "[✓] CLEAN"
                f.write(f"{entry['time']:<8} | {entry['fake_score']:<14}% | {status_icon}\n")
            f.write("-" * 60 + "\n\n")
        f.write("EOF (End of Forensic Log)")
    return report_name

# --- 3. MAIN APP ROUTES ---
@app.route('/', methods=['GET', 'POST'])
def index():
    analysis = None
    if request.method == 'POST' and 'user_id' in session:
        file = request.files.get('file')
        a_type = request.form.get('analysis_type')
        if file:
            filename = secure_filename(file.filename)
            path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(path)

            if a_type == 'image':
                img = Image.open(path).convert('RGB')
                tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), 
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                tensor = tf(img).unsqueeze(0).to(DEVICE)
                out = image_model(tensor)
                prob = torch.softmax(out, dim=1)
                conf, pred = torch.max(prob, 1)
                
                # Logic for both REAL and FAKE
                res = 'REAL' if pred.item() == 1 else 'FAKE'
                conf_val = round(conf.item()*100, 2)
                
                generate_forensic_report(filename, res, conf_val, 'image')
                analysis = {'type': 'image', 'result': res, 'confidence': conf_val, 'path': path}

            else: # Video Mode
                try:
                    detailed_frames = analyze_video_detailed(path)
                    a_t = process_audio(path).to(DEVICE)
                    audio_features = a_net(a_t)
                    
                    timeline, scores = [], []
                    for segment in detailed_frames:
                        v_feat = v_net(segment['tensor'].to(DEVICE))
                        out = v_classifier(v_feat, audio_features)
                        prob_fake = F.softmax(out, dim=1)[0][0].item() # index 0 = Fake
                        
                        timeline.append({
                            'time': segment['time'],
                            'fake_score': round(prob_fake * 100, 2),
                            'status': 'FAKE' if prob_fake > 0.5 else 'REAL'
                        })
                        scores.append(prob_fake)

                    avg_fake = sum(scores)/len(scores)
                    # Logic for both REAL and FAKE
                    res = 'FAKE' if avg_fake > 0.5 else 'REAL'
                    conf_val = round(avg_fake * 100 if res == 'FAKE' else (1 - avg_fake) * 100, 2)
                    
                    generate_forensic_report(filename, res, conf_val, 'video', timeline=timeline)
                    analysis = {'type': 'video', 'result': res, 'confidence': conf_val, 'path': path, 'timeline': timeline}
                except Exception as e:
                    flash(f"Forensic Error: {str(e)}", "danger")

    if 'user_id' in session:
        return render_template('merge.html', view='dashboard', username=session.get('username'), analysis=analysis)
    return render_template('merge.html', view='entry', mode=request.args.get('mode', 'login'))

# --- 4. AUTHENTICATION (Simplified) ---
@app.route('/login', methods=['POST'])
def login():
    email, pw = request.form.get('email'), request.form.get('password')
    with get_db_connection() as conn:
        user = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
    if user and check_password_hash(user['password'], pw):
        session['user_id'], session['username'] = user['id'], user['username']
        return redirect(url_for('index'))
    flash("Invalid credentials.", "danger")
    return redirect(url_for('index', mode='login'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index', mode='login'))

if __name__ == '__main__':
    app.run(debug=True)