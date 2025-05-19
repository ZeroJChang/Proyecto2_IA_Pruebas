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

# Configuración
IMG_SIZE = 128
MODEL_PATH = '../model/best_model.h5'
LABELS_PATH = '../model/labels.npy'

model = load_model(MODEL_PATH)
labels = np.load(LABELS_PATH)
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
root.geometry("850x650")

main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True)

frame_top = tk.Frame(main_frame)
frame_top.pack(side=tk.TOP, fill=tk.X, pady=5)

frame_video = tk.Frame(main_frame)
frame_video.pack(side=tk.TOP, pady=5)

frame_info = tk.Frame(main_frame)
frame_info.pack(side=tk.TOP, pady=5)

frame_bottom = tk.Frame(main_frame, bg="#f0f0f0")
frame_bottom.pack(side=tk.TOP, fill=tk.X, pady=10)

# Componentes
video_label = tk.Label(frame_video)
video_label.pack()

status_label = tk.Label(frame_info, text="Estado: Esperando...", font=("Arial", 12))
status_label.pack(pady=2)

word_label = tk.Label(frame_info, text="Palabra: ", font=("Arial", 16), fg="blue")
word_label.pack(pady=2)

prediction_label = tk.Label(frame_info, text="Letra: - (Confianza: -)", font=("Arial", 14))
prediction_label.pack(pady=2)

roi_label = tk.Label(frame_info)
roi_label.pack(pady=5)

# Botones
for text, cmd in [("🧹 Limpiar", lambda: limpiar_palabra()), ("⚙️ Ejecutar", lambda: ejecutar_comando()), ("❌ Salir", lambda: salir())]:
    btn = ttk.Button(frame_bottom, text=text, command=cmd)
    btn.pack(side=tk.LEFT, expand=True, padx=20, pady=5)

added_letters = set()

def update_video():
    global label, confidence, roi, built_word
    ret, frame = cap.read()
    if not ret:
        return
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    hand_detected = False
    hand_roi_display = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

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
                hand_roi_display = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
                if confidence >= 0.8:
                    built_word += label
                    print(f"Letra añadida automáticamente: {label} → Palabra: {built_word}")

    status_label.config(text="Estado: Mano detectada" if hand_detected else "Estado: Buscando mano...")
    prediction_label.config(text=f"Letra: {label} (Confianza: {confidence:.2f})" if hand_detected else "Letra: - (Confianza: -)")

    img = Image.fromarray(frame_rgb)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    roi_img = Image.fromarray(cv2.cvtColor(hand_roi_display, cv2.COLOR_BGR2RGB))
    roi_imgtk = ImageTk.PhotoImage(image=roi_img)
    roi_label.imgtk = roi_imgtk
    roi_label.configure(image=roi_imgtk)

    word_label.config(text=f"Palabra: {built_word}")
    root.after(10, update_video)

def limpiar_palabra():
    global built_word
    built_word = ""
    print("Palabra limpiada.")

def ejecutar_comando():
    global built_word
    palabra = built_word.upper()
    if palabra:
        if palabra == "O":
            print("Ejecutando Microsoft Word")
            subprocess.Popen("winword")
        elif palabra == "H":
            print("Ejecutando Bloc de Notas")
            subprocess.Popen("notepad")
        else:
            print(f"No hay acción asignada para '{palabra}'")
        engine.say(built_word)
        engine.runAndWait()
        built_word = ""
        print("Palabra ejecutada y limpiada.")

def salir():
    root.destroy()
    cap.release()
    cv2.destroyAllWindows()
    if built_word:
        engine.say(built_word)
        engine.runAndWait()

update_video()
root.mainloop()