import cv2
import numpy as np
import uuid
import os
import pyttsx3
import subprocess
import json
import mediapipe as mp
from tensorflow.keras.models import load_model

# Configuración
IMG_SIZE = 128
MODEL_PATH = '../model/sign_model.h5'
LABELS_PATH = '../model/labels.npy'
DICT_PATH = 'word_dict.json'

# Cargar modelo y etiquetas
model = load_model(MODEL_PATH)
labels = np.load(LABELS_PATH)

# Cargar diccionario de palabras/comandos
if os.path.exists(DICT_PATH):
    with open(DICT_PATH, 'r') as f:
        word_dict = json.load(f)
else:
    print(f"No se encontró el archivo {DICT_PATH}")
    word_dict = {}

# Inicializar motor de texto a voz
engine = pyttsx3.init()

# Configurar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    normalized = resized / 255.0
    reshaped = normalized.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    return reshaped

def get_hand_roi(frame, hand_landmarks):
    # Obtener coordenadas de la mano
    x_coords = [landmark.x * frame.shape[1] for landmark in hand_landmarks.landmark]
    y_coords = [landmark.y * frame.shape[0] for landmark in hand_landmarks.landmark]
    
    min_x, max_x = int(min(x_coords)), int(max(x_coords))
    min_y, max_y = int(min(y_coords)), int(max(y_coords))
    
    # Añadir margen alrededor de la mano
    margin = 30
    min_x = max(0, min_x - margin)
    max_x = min(frame.shape[1], max_x + margin)
    min_y = max(0, min_y - margin)
    max_y = min(frame.shape[0], max_y + margin)
    
    # Asegurar relación de aspecto cuadrada
    width = max_x - min_x
    height = max_y - min_y
    size = max(width, height)
    
    # Centrar el ROI
    center_x = (min_x + max_x) // 2
    center_y = (min_y + max_y) // 2
    
    min_x = max(0, center_x - size//2)
    max_x = min(frame.shape[1], center_x + size//2)
    min_y = max(0, center_y - size//2)
    max_y = min(frame.shape[0], center_y + size//2)
    
    return min_x, min_y, max_x, max_y

cap = cv2.VideoCapture(0)
built_word = ""
hand_detected = False

print("🎥 S: guardar letra | D: borrar última | C: limpiar palabra | Espacio: ejecutar si aplica | Q: salir")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convertir a RGB para MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    hand_detected = False
    roi_coords = None
    
    if results.multi_hand_landmarks:
        hand_detected = True
        for hand_landmarks in results.multi_hand_landmarks:
            # Dibujar landmarks de la mano
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Obtener ROI alrededor de la mano
            roi_coords = get_hand_roi(frame, hand_landmarks)
            min_x, min_y, max_x, max_y = roi_coords
            
            # Dibujar rectángulo alrededor de la mano
            cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
            
            # Preprocesar y predecir solo si hay mano detectada
            roi = frame[min_y:max_y, min_x:max_x]
            if roi.size > 0:  # Asegurarse que el ROI no está vacío
                processed = preprocess_frame(roi)
                prediction = model.predict(processed, verbose=0)
                class_idx = np.argmax(prediction)
                label = labels[class_idx]
                confidence = prediction[0][class_idx]
                
                # Mostrar predicción
                cv2.putText(frame, f'Letra: {label} ({confidence:.2f})', 
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    # Mostrar palabra construida
    cv2.putText(frame, f'Palabra: {built_word}', (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    
    # Mostrar estado de detección
    status_text = "Mano detectada" if hand_detected else "Buscando mano..."
    cv2.putText(frame, status_text, (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    cv2.imshow('Traductor de Señas', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s') and hand_detected:
        built_word += label
        print(f"Letra '{label}' añadida → Palabra: {built_word}")
        os.makedirs('capturas', exist_ok=True)
        filename = f"capturas/{label}_{uuid.uuid4().hex[:8]}.jpg"
        cv2.imwrite(filename, roi)

    elif key == ord('d'):
        if built_word:
            built_word = built_word[:-1]
            print(f"Última letra eliminada → Palabra: {built_word}")

    elif key == ord('c'):
        built_word = ""
        print("Palabra limpiada.")

    elif key == 32:  # Tecla espacio
        upper_word = built_word.upper()
        if upper_word in word_dict:
            ruta = word_dict[upper_word]
            print(f"Ejecutando '{upper_word}': {ruta}")
            try:
                subprocess.Popen(ruta)
            except Exception as e:
                print(f"Error al ejecutar: {e}")
        else:
            print(f"No se reconoce el comando '{upper_word}'")
        built_word = ""

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if built_word:
    print(f"Diciendo la palabra: {built_word}")
    engine.say(built_word)
    engine.runAndWait()