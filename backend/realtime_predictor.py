import cv2
import numpy as np
import os
import pyttsx3
import subprocess
import json
import mediapipe as mp
import joblib
from datetime import datetime

# Cargar modelo entrenado con landmarks
MODEL_PATH = '../model/landmark_model.pkl'
LABELS_PATH = '../model/landmark_labels.npy'
MODEL_PATHP = '../model/best_model.h5'
LABELS_PATHP = '../model/labels.npy'
DICT_PATH = '../model/word_dict.json'

model = joblib.load(MODEL_PATH)
labels = np.load(LABELS_PATH)

# Diccionario de acciones
if os.path.exists(DICT_PATH):
    with open(DICT_PATH, 'r') as f:
        word_dict = json.load(f)
else:
    print(f"No se encontró el archivo {DICT_PATH}")
    word_dict = {}

engine = pyttsx3.init()

# Configurar MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
built_word = ""
label = "-"
confidence = 0.0

print("🎥 Traductor activo con ingreso manual")
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

            # Extraer 63 valores de landmarks
            vector = []
            for lm in hand_landmarks.landmark:
                vector.extend([lm.x, lm.y, lm.z])

            # Predicción
            pred = model.predict([vector])[0]
            proba = model.predict_proba([vector])[0]
            conf = np.max(proba)

            label = pred
            confidence = conf

            # Mostrar letra actual pero no la agrega automáticamente
            cv2.putText(frame, f'Letra actual: {label} ({confidence:.2f})',
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.putText(frame, f'Palabra: {built_word}', (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    estado = "Mano detectada" if hand_detected else "Buscando mano..."
    cv2.putText(frame, estado, (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow('Traductor de Señas con Landmarks', frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s') and hand_detected:
        built_word += label
        print(f"### Letra '{label}' añadida → Palabra: {built_word}")

    elif key == ord('g'):
        if built_word:
            with open("palabras_guardadas.txt", "a") as f:
                f.write(f"{datetime.now()}: {built_word}\n")
            print(f"### Palabra '{built_word}' guardada en palabras_guardadas.txt")
        else:
            print("### No hay palabra para guardar.")

    elif key == ord('c'):
        built_word = ""
        print("### Palabra limpiada.")

    elif key == 32:
        palabra = built_word.upper()
        if palabra in word_dict:
            ruta = word_dict[palabra]
            print(f"### Ejecutando '{palabra}': {ruta}")
            try:
                subprocess.Popen(ruta)
            except Exception as e:
                print(f"Error al ejecutar: {e}")
        else:
            print(f"### No se reconoce el comando '{palabra}'")

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
