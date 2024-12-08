import os
import sqlite3
from flask import Flask, render_template, request, redirect, Response, jsonify
import onnxruntime as ort
from PIL import Image, ImageDraw
import base64
import numpy as np
import cv2

# Inicializar o Flask
app = Flask(__name__)

# Configurações de diretórios
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/output'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Carregar o modelo ONNX
ort_session = ort.InferenceSession('best.onnx')

def preprocess_image(img):
    """Pré-processar a imagem para entrada no modelo ONNX."""
    img = img.resize((320, 320))  # Reduzir resolução
    img = np.array(img).astype(np.float32) / 255.0  # Normalizar para [0,1]
    img = np.transpose(img, (2, 0, 1))  # Alterar para formato [C, H, W]
    return np.expand_dims(img, axis=0)

# Função para inicializar o banco de dados
def init_db():
    conn = sqlite3.connect('database.db')
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

# Inicializar o banco de dados
init_db()

@app.route('/')
def index():
    return render_template('index.html')

# Função para gerar o feed de vídeo
def generate_video_feed():
    # Abrir a câmera
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Converter para RGB (OpenCV usa BGR por padrão)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img)

        # Pré-processar a imagem e realizar a predição
        input_tensor = preprocess_image(pil_img)
        results = ort_session.run(None, {"images": input_tensor})[0]

        # Desenhar as caixas na imagem
        for det in results:
            x1, y1, x2, y2, conf, cls = det
            if conf >= 0.85:  # Filtrar predições com confiança >= 0.85
                label = f"Classe {int(cls)} {conf:.2f}"
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Converter para JPEG e enviar via Response
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
        # Receber a imagem em base64
        data = request.get_json()
        image_data = data['image']
        
        # Converter base64 para imagem
        img_data = base64.b64decode(image_data.split(',')[1])
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Converter para RGB (OpenCV usa BGR por padrão)
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # Pré-processar e realizar a predição
        input_tensor = preprocess_image(pil_img)
        results = ort_session.run(None, {"images": input_tensor})[0]

        # Contar e coletar as coordenadas dos BigBags detectados (classe 0, BigBag)
        bigbags = []
        bigbag_count = 0
        for det in results:
            x1, y1, x2, y2, conf, cls = det
            if conf >= 0.85 and int(cls) == 0:  # Filtrar BigBags com confiança >= 0.85
                bigbag_count += 1
                bigbags.append({
                    "x": int(x1),
                    "y": int(y1),
                    "width": int(x2 - x1),
                    "height": int(y2 - y1)
                })
        
        # Retornar a contagem e as coordenadas dos BigBags
        return jsonify({
            "status": "success", 
            "count": bigbag_count, 
            "bigbags": bigbags
        })

    except Exception as e:
        print(f"Erro durante a predição: {e}")
        return jsonify({"status": "error", "message": str(e)})

@app.route('/history')
def history():
    try:
        # Recuperar os dados do banco de dados
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute('SELECT id, image_name, count FROM detections')
        records = cursor.fetchall()
        conn.close()

        return render_template('history.html', records=records)

    except Exception as e:
        print(f"Erro ao acessar o histórico: {e}")
        return "Ocorreu um erro ao acessar o histórico. Verifique o console para mais detalhes."

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
