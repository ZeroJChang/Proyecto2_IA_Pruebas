# Traductor de Lenguaje de Señas – Proyecto Rafaelitos

Este proyecto implementa una aplicación de reconocimiento de letras en lenguaje de señas (ASL) utilizando visión por computadora con **MediaPipe** y **TensorFlow**. Permite capturar señas en tiempo real a través de una cámara web, mostrar la letra detectada y construir palabras, las cuales pueden ejecutarse como comandos (ej. abrir Word o Notepad).

---

## Uso

### 1. El usuario posiciona su mano frente a la cámara.
### 2. El sistema detecta la mano y predice la letra.
### 3. Si la confianza ≥ 0.8, se agrega automáticamente a la palabra.
### 4. El usuario puede:
- Limpiar la palabra.
- Ejecutar una acción (abrir Word, Notepad, etc.).
---

## Descripción de archivos y carpetas

### `backend/`
- `realtime_predictor.py`: Script principal de predicción en vivo con OpenCV + MediaPipe + Tkinter.
- `train_model.py`: Entrena el modelo CNN a partir de imágenes de lenguaje de señas.
- `realtime_predictorfront.py`: Versión alternativa o frontal para pruebas.
- `word_dict.json`: Diccionario que asocia palabras con comandos de sistema.

### `model/`
- `best_model.h5`: Modelo entrenado con mayor precisión.
- `labels.npy`: Archivo con el orden de etiquetas (letras A-Z).
- `confusion_matrix.png`: Imagen con la matriz de confusión del modelo.

---

## Tecnologías utilizadas

- **Python 3.10**
- **OpenCV**
- **MediaPipe Hands**
- **TensorFlow / Keras**
- **Tkinter**
- **pyttsx3** (Texto a Voz)
- **subprocess** (ejecución de comandos del sistema)
- **React** 

---

## Requisitos técnicos

- Python 3.8 o superior
- Paquetes necesarios:

```bash
pip install opencv-python mediapipe tensorflow pyttsx3
```
## Evaluación del modelo

![image](https://github.com/user-attachments/assets/8d1720b7-23d3-416d-9638-8d21c38fa0e0)
---
## Matriz de confusión
![confusion_matrix](https://github.com/user-attachments/assets/87212d42-976b-4c08-8296-2edf97e3f3a9)
---

## Dataset
Se utilizó un dataset de lenguaje de señas americano (ASL) procesado con MediaPipe, con ~700 imágenes por letra.
El preprocesamiento incluye:

- **Recorte centrado de la mano**
- **Escala a escala de grises de 128x128 px**
- **Normalización de pixeles**

---

## Ejecucción del modelo

![image](https://github.com/user-attachments/assets/8f08a42e-ded6-431a-b56e-d816f3794da6)

