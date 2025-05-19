import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import uuid
import os
import pyttsx3
import subprocess
import json
import mediapipe as mp
from tensorflow.keras.models import load_model

# Configuracion
IMG_SIZE = 128
MODEL_PATH = '../model/best_model.h5'
LABELS_PATH = '../model/labels.npy'
DICT_PATH = 'word_dict.json'

model = load_model(MODEL_PATH)
labels = np.load(LABELS_PATH)
word_dict = json.load(open(DICT_PATH)) if os.path.exists(DICT_PATH) else {}
engine = pyttsx3.init()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

built_word = ""
label = ""
confidence = 0
roi = None

# Procesamiento de imagen
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    normalized = resized / 255.0
    reshaped = normalized.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    return reshaped

def get_hand_roi(frame, hand_landmarks):
    x_coords = [landmark.x * frame.shape[1] for landmark in hand_landmarks.landmark]
    y_coords = [landmark.y * frame.shape[0] for landmark in hand_landmarks.landmark]
    min_x, max_x = int(min(x_coords)), int(max(x_coords))
    min_y, max_y = int(min(y_coords)), int(max(y_coords))
    margin = 30
    min_x = max(0, min_x - margin)
    max_x = min(frame.shape[1], max_x + margin)
    min_y = max(0, min_y - margin)
    max_y = min(frame.shape[0], max_y + margin)
    width, height = max_x - min_x, max_y - min_y
    size = max(width, height)
    center_x, center_y = (min_x + max_x) // 2, (min_y + max_y) // 2
    min_x = max(0, center_x - size//2)
    max_x = min(frame.shape[1], center_x + size//2)
    min_y = max(0, center_y - size//2)
    max_y = min(frame.shape[0], center_y + size//2)
    return min_x, min_y, max_x, max_y

# GUI
root = tk.Tk()
root.title("Traductor de Lenguaje de Señas")

video_label = tk.Label(root)
video_label.pack()

status_label = tk.Label(root, text="Estado: Esperando...", font=("Arial", 12))
status_label.pack()

word_label = tk.Label(root, text="Palabra: ", font=("Arial", 16), fg="blue")
word_label.pack()

prediction_label = tk.Label(root, text="Letra: - (Confianza: -)", font=("Arial", 14))
prediction_label.pack()

def update_video():
    global label, confidence, roi, built_word
    ret, frame = cap.read()
    if not ret:
        return
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    hand_detected = False

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            hand_detected = True
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            min_x, min_y, max_x, max_y = get_hand_roi(frame, hand_landmarks)
            roi = frame[min_y:max_y, min_x:max_x]
            if roi.size > 0:
                processed = preprocess_frame(roi)
                prediction = model.predict(processed, verbose=0)
                class_idx = np.argmax(prediction)
                label = labels[class_idx]
                confidence = prediction[0][class_idx]
                cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)

    status_label.config(text="Estado: Mano detectada" if hand_detected else "Estado: Buscando mano...")
    prediction_label.config(text=f"Letra: {label} (Confianza: {confidence:.2f})" if hand_detected else "Letra: - (Confianza: -)")
    img = Image.fromarray(frame_rgb)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)
    word_label.config(text=f"Palabra: {built_word}")
    root.after(10, update_video)

def guardar_letra():
    global built_word, label, roi
    if label:
        built_word += label
        os.makedirs('capturas', exist_ok=True)
        filename = f"capturas/{label}_{uuid.uuid4().hex[:8]}.jpg"
        if roi is not None:
            cv2.imwrite(filename, roi)

def borrar_letra():
    global built_word
    built_word = built_word[:-1]

def limpiar_palabra():
    global built_word
    built_word = ""

def ejecutar_comando():
    global built_word
    palabra = built_word.upper()
    if palabra in word_dict:
        ruta = word_dict[palabra]
        try:
            subprocess.Popen(ruta)
        except Exception as e:
            print(f"Error al ejecutar: {e}")
    else:
        print(f"No se reconoce el comando '{palabra}'")
    built_word = ""

def salir():
    root.destroy()
    cap.release()
    cv2.destroyAllWindows()
    if built_word:
        engine.say(built_word)
        engine.runAndWait()

frame_buttons = tk.Frame(root)
frame_buttons.pack(pady=10)

buttons = [
    ("Guardar Letra (S)", guardar_letra),
    ("Borrar Última (D)", borrar_letra),
    ("Limpiar Palabra (C)", limpiar_palabra),
    ("Ejecutar (Espacio)", ejecutar_comando),
    ("Salir (Q)", salir)
]

for i, (text, cmd) in enumerate(buttons):
    btn = ttk.Button(frame_buttons, text=text, command=cmd)
    btn.grid(row=0, column=i, padx=5)

update_video()
root.mainloop()
