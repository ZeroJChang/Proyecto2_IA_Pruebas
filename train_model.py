import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SeparableConv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# --- Configuración ---
DATA_DIR = 'dataset'         # Carpeta con las imágenes
IMG_SIZE = 128                # Reducir resolución para acelerar
BATCH_SIZE = 128             # Tamaño de lote grande para mejor uso de GPU/CPU
EPOCHS = 30                  # Épocas moderadas con EarlyStopping y callbacks
AUTOTUNE = tf.data.AUTOTUNE  # Paralelismo en tf.data

# --- Detectar GPU y mixed precision ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    mixed_precision.set_global_policy('mixed_float16')
    print(f"GPUs detectadas: {gpus}. Usando mixed precision.")
else:
    print("No se detectó GPU. Entrenando en float32.")

# --- Leer rutas y etiquetas ---
all_paths = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR)
             if f.lower().endswith(('.jpg','.png','.jpeg'))]
labels = [os.path.basename(p)[0].upper() for p in all_paths]
classes = sorted(set(labels))
label_to_index = {label: idx for idx, label in enumerate(classes)}
label_indices = np.array([label_to_index[l] for l in labels])

# --- Calcular pesos de clase (balanceo) ---
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.arange(len(classes)),
    y=label_indices
)
class_weight_dict = {i: w for i, w in enumerate(class_weights)}

# --- Crear dataset con tf.data ---
paths_ds = tf.data.Dataset.from_tensor_slices(all_paths)
labels_ds = tf.data.Dataset.from_tensor_slices(label_indices)
dataset = tf.data.Dataset.zip((paths_ds, labels_ds))
dataset = dataset.shuffle(buffer_size=len(all_paths), seed=42)

# Funciones de preprocesamiento y aumento

def parse_and_preprocess(path, label):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=1)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = img / 255.0
    return img, label


def augment(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, 0.1)
    img = tf.image.random_contrast(img, 0.9, 1.1)
    return img, label

# Dividir en entrenamiento y validación
dataset_size = len(all_paths)
train_size = int(0.8 * dataset_size)
train_ds = (dataset.take(train_size)
            .map(parse_and_preprocess, num_parallel_calls=AUTOTUNE)
            .map(augment, num_parallel_calls=AUTOTUNE)
            .cache()
            .batch(BATCH_SIZE)
            .prefetch(AUTOTUNE))

val_ds = (dataset.skip(train_size)
          .map(parse_and_preprocess, num_parallel_calls=AUTOTUNE)
          .cache()
          .batch(BATCH_SIZE)
          .prefetch(AUTOTUNE))

# --- Definir el modelo CNN eficiente ---
model = Sequential([
    tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
    SeparableConv2D(32, 3, activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),

    SeparableConv2D(64, 3, activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),

    SeparableConv2D(128, 3, activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(classes), activation='softmax', dtype='float32')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# --- Callbacks ---
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
    ModelCheckpoint('model/best_model.h5', save_best_only=True)
]

# --- Entrenamiento ---
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weight_dict
)

# --- Guardar modelo y etiquetas ---
os.makedirs('model', exist_ok=True)
model.save('model/sign_model.h5')
with open('model/labels.txt', 'w') as f:
    for c in classes:
        f.write(f"{c}\n")
# Guardar etiquetas en numpy para la UI
np.save(os.path.join('model','labels.npy'), np.array(classes))
print("✔️ Etiquetas guardadas en 'model/labels.npy'.")

print("✔️ Entrenamiento completado. Modelo guardado en 'model/sign_model.h5'.")

# --- Evaluación en el set de validación ---
print("\n🔍 Evaluación en validación:")
y_true = []
y_pred = []
for imgs, labels in val_ds:
    preds = model.predict(imgs)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(labels.numpy())

print("\n📊 Reporte de Clasificación:")
print(classification_report(y_true, y_pred, target_names=classes, digits=4))

cm = confusion_matrix(y_true, y_pred)
print("\n🗂 Matriz de Confusión:")
print(cm)

# Graficar y guardar la matriz de confusión
plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Matriz de Confusión')
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=90)
plt.yticks(tick_marks, classes)
plt.ylabel('Verdaderos')
plt.xlabel('Predichos')
plt.tight_layout()
plt.savefig('model/confusion_matrix.png')
plt.close()
print("✔️ Matriz de confusión guardada en 'model/confusion_matrix.png'.")
