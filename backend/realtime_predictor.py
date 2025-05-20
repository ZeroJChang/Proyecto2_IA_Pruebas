import cv2
import numpy as np
import os
import pyttsx3
import subprocess
import json
import mediapipe as mp
import joblib
from datetime import datetime
from tensorflow.keras.models import load_model

# Rutas de modelos y etiquetas
MODEL_LANDMARK_PATH = '../model/landmark_model.pkl'
LABELS_PATH = '../model/labels.npy'
MODEL_IMAGE_PATH = '../model/best_model.h5'
DICT_PATH = '../model/word_dict.json'

# Cargar modelos
model_landmark = joblib.load(MODEL_LANDMARK_PATH)
model_image = load_model(MODEL_IMAGE_PATH)
labels = np.load(LABELS_PATH)

# Cargar diccionario de acciones
if os.path.exists(DICT_PATH):
    with open(DICT_PATH, 'r') as f:
        word_dict = json.load(f)
else:
    print(f"No se encontró el archivo {DICT_PATH}")
    word_dict = {}

# Inicializar voz
engine = pyttsx3.init()

# Inicializar MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Función de preprocesamiento para el modelo por imagen
def preprocess_image(roi, size=128):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (size, size))
    normalized = resized / 255.0
    reshaped = normalized.reshape(1, size, size, 1)
    return reshaped

cap = cv2.VideoCapture(0)
built_word = ""
label = "-"
confidence = 0.0

print("🎥 Traductor activo con fusión de modelos (imagen + landmarks)")
print("Teclas disponibles:")
print(" S: agregar letra actual a la palabra")
print(" G: guardar palabra actual en archivo")
print("␣  Espacio: ejecutar palabra")
print(" C: limpiar palabra")
print(" Q: salir")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    hand_detected = False

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            hand_detected = True
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            vector = []
            for lm in hand_landmarks.landmark:
                vector.extend([lm.x, lm.y, lm.z])
            pred_landmark = model_landmark.predict_proba([vector])[0]

            # === Imagen (ROI) ===
            x_coords = [lm.x * frame.shape[1] for lm in hand_landmarks.landmark]
            y_coords = [lm.y * frame.shape[0] for lm in hand_landmarks.landmark]
            min_x, max_x = int(min(x_coords)), int(max(x_coords))
            min_y, max_y = int(min(y_coords)), int(max(y_coords))
            margin = 30
            min_x = max(0, min_x - margin)
            max_x = min(frame.shape[1], max_x + margin)
            min_y = max(0, min_y - margin)
            max_y = min(frame.shape[0], max_y + margin)
            roi = frame[min_y:max_y, min_x:max_x]

            if roi.size > 0:
                pred_image = model_image.predict(preprocess_image(roi), verbose=0)[0]
            else:
                pred_image = np.zeros(len(labels))

            # === Fusión ===
            combined = (0.5 * pred_landmark) + (0.5 * pred_image)
            class_idx = np.argmax(combined)
            label = labels[class_idx]
            confidence = combined[class_idx]

            # Mostrar letra actual
            cv2.putText(frame, f'Letra actual: {label} ({confidence:.2f})',
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.putText(frame, f'Palabra: {built_word}', (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    estado = "Mano detectada" if hand_detected else "Buscando mano..."
    cv2.putText(frame, estado, (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow('Traductor de Señas (Modelo Combinado)', frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s') and hand_detected:
        built_word += label
        print(f" Letra '{label}' añadida → Palabra: {built_word}")
        engine.say(label)
        engine.runAndWait()

    elif key == ord('g'):
        if built_word:
            with open("palabras_guardadas.txt", "a") as f:
                f.write(f"{datetime.now()}: {built_word}\n")
            print(f" Palabra '{built_word}' guardada en palabras_guardadas.txt")
        else:
            print(" No hay palabra para guardar.")

    elif key == ord('c'):
        built_word = ""
        print(" Palabra limpiada.")

    elif key == 32:
        palabra = built_word.upper()
        if palabra in word_dict:
            ruta = word_dict[palabra]
            print(f"⚙️ Ejecutando '{palabra}': {ruta}")
            try:
                subprocess.Popen(ruta, shell=True)
            except Exception as e:
                print(f" Error al ejecutar: {e}")
        else:
            print(f" No se reconoce el comando '{palabra}'")

        engine.say(built_word)
        engine.runAndWait()
        built_word = ""

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if built_word:
    print(f" Diciendo la palabra final: {built_word}")
    engine.say(built_word)
    engine.runAndWait()
