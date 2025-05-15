import cv2
import numpy as np
import uuid
import os
import pyttsx3
from tensorflow.keras.models import load_model

IMG_SIZE = 64
MODEL_PATH = '../model/sign_model.h5'
LABELS_PATH = '../model/labels.npy'

engine = pyttsx3.init()

model = load_model(MODEL_PATH)
labels = np.load(LABELS_PATH)

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    normalized = resized / 255.0
    reshaped = normalized.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    return reshaped

cap = cv2.VideoCapture(0)
built_word = ""

print("🎥 Presiona 's' para guardar letra, 'c' para limpiar palabra, 'q' para salir")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    x1, y1, x2, y2 = 100, 100, 300, 300
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    roi = frame[y1:y2, x1:x2]

    processed = preprocess_frame(roi)
    prediction = model.predict(processed, verbose=0)
    class_idx = np.argmax(prediction)
    label = labels[class_idx]
    confidence = prediction[0][class_idx]

    cv2.putText(frame, f'Letra: {label} ({confidence:.2f})', (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    cv2.putText(frame, f'Palabra: {built_word}', (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    cv2.imshow('Traductor de Señas', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        built_word += label
        print(f"🔠 Letra '{label}' añadida → Palabra: {built_word}")

        os.makedirs('capturas', exist_ok=True)
        filename = f"capturas/{label}_{uuid.uuid4().hex[:8]}.jpg"
        cv2.imwrite(filename, roi)
        print(f"📸 Imagen guardada: {filename}")

    elif key == ord('c'):
        built_word = ""
        print("🧹 Palabra limpiada.")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if built_word:
    print(f"🔊 Diciendo la palabra: {built_word}")
    engine.say(built_word)
    engine.runAndWait()
