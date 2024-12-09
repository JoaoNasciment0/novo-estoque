import os
import sqlite3
from flask import Flask, render_template, request, redirect, Response, jsonify
import torch
from PIL import Image
import base64
import numpy as np
import cv2
import gc  
from pathlib import Path
import sys

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/output'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Caminho para o repositório YOLOv5 clonado
YOLOV5_PATH = Path("train/yolov5")

# Adicionar o YOLOv5 ao PYTHONPATH
sys.path.append(str(YOLOV5_PATH))

# Carregar o modelo YOLOv5
model = torch.hub.load(str(YOLOV5_PATH), 'custom', path='IA-Treinada/best.pt', source='local', device='cpu')
model.eval()

def init_db():
    conn = sqlite3.connect('db/database.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_name TEXT NOT NULL,
            image_path TEXT NOT NULL,
            count INTEGER NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

@app.route('/')
def index():
    return render_template('index.html')

def generate_video_feed():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img)

        with torch.no_grad():
            results = model([pil_img])

        detections = results.xyxy[0].cpu().numpy()
        filtered_detections = [d for d in detections if d[4] >= 0.85]

        for det in filtered_detections:
            x1, y1, x2, y2, conf, cls = det
            label = f"{results.names[int(cls)]} {conf:.2f}"
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_video_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict_camera', methods=['POST'])
def predict_camera():
    try:
        data = request.get_json()
        image_data = data['image']
        img_data = base64.b64decode(image_data.split(',')[1])
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        with torch.no_grad():
            results = model([pil_img])

        detections = results.xyxy[0].cpu().numpy()
        filtered_detections = [d for d in detections if d[4] >= 0.85]

        bigbags = []
        bigbag_count = 0
        for det in filtered_detections:
            cls = int(det[5])
            if cls == 0:
                bigbag_count += 1
                x1, y1, x2, y2, _, _ = det
                bigbags.append({
                    "x": int(x1),
                    "y": int(y1),
                    "width": int(x2 - x1),
                    "height": int(y2 - y1)
                })

        gc.collect()
        return jsonify({"status": "success", "count": bigbag_count, "bigbags": bigbags})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/history')
def history():
    try:
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute('SELECT id, image_name, count FROM detections')
        records = cursor.fetchall()
        conn.close()
        return render_template('history.html', records=records)

    except Exception as e:
        return "Ocorreu um erro ao acessar o histórico."

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
