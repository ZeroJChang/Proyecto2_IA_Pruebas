# capture_landmarks.py
import cv2
import mediapipe as mp
import numpy as np
import os
import csv

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

output_dir = "dataset_landmarks"
os.makedirs(output_dir, exist_ok=True)

current_label = "C"
print(f"Etiqueta actual: {current_label}")
print("Presiona 'g' para guardar, 'c' para cambiar letra, 'q' para salir.")

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            vector = []
            for lm in hand_landmarks.landmark:
                vector.extend([lm.x, lm.y, lm.z])
            # Mostrar que se detectaron 21 puntos
            cv2.putText(frame, f"Landmarks detectados: {len(vector)//3}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.putText(frame, f"Letra actual: {current_label}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow("Captura de Landmarks", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('c'):
        current_label = input("Cambiar a letra: ").upper()
        print(f"Etiqueta actual: {current_label}")
    elif key == ord('g') and results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            vector = []
            for lm in hand_landmarks.landmark:
                vector.extend([lm.x, lm.y, lm.z])
            file_path = os.path.join(output_dir, f"{current_label}.csv")
            with open(file_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(vector)
            print(f"Guardado en {file_path}")

cap.release()
cv2.destroyAllWindows()
