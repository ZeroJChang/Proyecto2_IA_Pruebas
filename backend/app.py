from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import os
import json
import subprocess
import platform
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'model', 'sign_model.h5')
LABELS_PATH = os.path.join(BASE_DIR, '..', 'model', 'labels.npy')
DICT_PATH = os.path.join(BASE_DIR, 'word_dict.json')

model = load_model(MODEL_PATH)
labels = np.load(LABELS_PATH)

if os.path.exists(DICT_PATH):
    with open(DICT_PATH, 'r') as f:
        word_dict = json.load(f)
else:
    word_dict = {}

IMG_SIZE = 128

def preprocess_image(file_storage):
    file_bytes = np.frombuffer(file_storage.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    normalized = resized / 255.0
    reshaped = normalized.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    return reshaped

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No se encontró el archivo de imagen"}), 400

    file = request.files['file']
    processed = preprocess_image(file)
    if processed is None:
        return jsonify({"error": "No se pudo procesar la imagen"}), 400

    prediction = model.predict(processed, verbose=0)
    class_idx = np.argmax(prediction)
    label = str(labels[class_idx])
    confidence = float(prediction[0][class_idx])

    return jsonify({"letter": label, "confidence": confidence})

@app.route('/execute', methods=['POST'])
def execute_word():
    data = request.get_json()
    word = data.get('word', '').upper()
    if not word:
        return jsonify({"error": "No se proporcionó ninguna palabra"}), 400

    ruta = word_dict.get(word)
    try:
        system = platform.system()

        if ruta:
            if system == 'Windows':
                os.startfile(ruta)
            elif system == 'Darwin':
                subprocess.Popen(['open', ruta])
            else:
                subprocess.Popen(['xdg-open', ruta])
            return jsonify({"message": f"Ejecutando comando para '{word}'"})

        else:
            url = f"https://www.google.com/search?q={word}"
            if system == 'Windows':
                os.startfile(url)
            elif system == 'Darwin':
                subprocess.Popen(['open', url])
            else:
                subprocess.Popen(['xdg-open', url])
            return jsonify({"message": f"No se encontró comando para '{word}', buscando en Google..."})

    except Exception as e:
        return jsonify({"error": f"Error al ejecutar: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True)
