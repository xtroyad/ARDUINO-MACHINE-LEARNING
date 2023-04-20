# Uso de la GPU para aumentar la velocidad del entreno. NO SE USA POR AHOR
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" # usar GPU 0

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
import PIL
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # usar GPU 0


# -----------------------------------------------------------------------------------------
AUTOTUNE = tf.data.AUTOTUNE
batch_size = 32
img_height = 100
img_width = 100
epochs = 100


#-----------------------------------------------------------------------------------------
#FUNCIONES AUXILIARES PARA PODER HACER UN DATASET DE UNA CARPETA DE FOTOS
#-----------------------------------------------------------------------------------------

def get_label(file_path):
    # Convierte la ruta del archivo en una lista de componentes de ruta
    parts = tf.strings.split(file_path, os.path.sep)
    # La penúltima parte es el directorio de clases
    one_hot = parts[-2] == class_names
    # Codifica la etiqueta como un número entero
    return tf.argmax(one_hot)


def decode_img(img):
    # Convierte la cadena comprimida en un tensor uint8 de 3D
    img = tf.io.decode_jpeg(img, channels=3)
    # Cambia el tamaño de la imagen al tamaño deseado
    return tf.image.resize(img, [img_height, img_width])


def process_path(file_path):
    label = get_label(file_path)
    # Carga los datos en crudo desde el archivo como una cadena
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label

#-----------------------------------------------------------------------------------------
# PROCESANDO IMAGENES PARA CREAR UN DATASET
#-----------------------------------------------------------------------------------------

# Definimos la ruta de nuestro directorio de imágenes
data_dir = pathlib.Path("imagenes")

# Contamos el número de imágenes que tenemos en nuestro directorio
image_count = len(list(data_dir.glob('*/*.jpg')))

# Creamos un Dataset a partir de las rutas de las imágenes, sin mezclar el orden de las imágenes
list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'), shuffle=False)

# Mezclamos aleatoriamente las rutas de las imágenes
list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)

# -----------------------------------------------------------------------------------------

# Creamos el conjunto de clases que puede tener nuestro modelo
class_names = np.array(sorted(
    [item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"]))

# -----------------------------------------------------------------------------------------

# Divida el conjunto de datos en conjuntos de entrenamiento y validación.
val_size = int(image_count * 0.2)
train_ds = list_ds.skip(val_size)
val_ds = list_ds.take(val_size)

# -----------------------------------------------------------------------------------------

# Usamos Daset.map para crear un conjunto de datos de pares (imagen, label)
train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)

# -----------------------------------------------------------------------------------------


def configure_for_performance(ds):
    # Almacenamos en caché para acelerar el proceso de lectura.
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)  # Mezclamos las imágenes aleatoriamente.
    # Agrupamos las imágenes por lotes para un procesamiento eficiente.
    ds = ds.batch(batch_size)
    # Preprocesamos los datos por lotes mientras se ejecuta el modelo en paralelo.
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


# Aplicamos la función configure_for_performance al conjunto de entrenamiento y de validación
train_ds = configure_for_performance(train_ds)
val_ds = configure_for_performance(val_ds)

#-----------------------------------------------------------------------------------------
# NORMALIZACION DE DATOS Y TRANFORMACION A LA ESCALA DE UN COLOR
#-----------------------------------------------------------------------------------------

normalization_layer = tf.keras.layers.Rescaling(1./255)
preprocessing_layer = tf.keras.layers.Lambda(
    lambda x: tf.image.rgb_to_grayscale(x))

# Aplicamos la capa de preprocesamiento y normalización al conjunto de entrenamiento
train_ds = train_ds.map(lambda x, y: (preprocessing_layer(x), y))

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

# Aplicamos la capa de preprocesamiento y normalización al conjunto de validación
val_ds = val_ds.map(lambda x, y: (preprocessing_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

#-----------------------------------------------------------------------------------------
# AUMENTO DE DATOS. ESTE ARGUMENTO COGE LAS FOTOS QUE YA TENEMOS Y LAS ROTA PARA DAR VARIEDAD
#-----------------------------------------------------------------------------------------

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1)
    ]
)

#-----------------------------------------------------------------------------------------
# POSIBLES MODELOS PARA LA PLACA. CNN2 PINTA SER LA MEJOR DE MOMENTO
#-----------------------------------------------------------------------------------------

num_classes = len(class_names)
model = Sequential([
    data_augmentation,
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])
# -----------------------------------------------------------------------------------------
model2 = Sequential([
    data_augmentation,
    layers.Flatten(),
    layers.Dense(150, activation='relu'),
    layers.Dense(150, activation='relu'),
    layers.Dense(4, activation='softmax')
])
# -----------------------------------------------------------------------------------------

modelCNN = Sequential([
    data_augmentation,
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(100, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])
# -----------------------------------------------------------------------------------------

modelCNN2 = tf.keras.models.Sequential([
    data_augmentation,
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.5),
    layers.Flatten(),
    layers.Dense(250, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

#-----------------------------------------------------------------------------------------
# INICIO DE ENTRENAMIENTO. PARA PROBAR REDES DIFERENTES CAMBIAR : "nombreModelo".compile y "nombreModelo".fit
#-----------------------------------------------------------------------------------------

modelCNN2.compile(
    optimizer='adam',
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy'])

history = modelCNN2.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

#-----------------------------------------------------------------------------------------
# VISUALIZAMOS LA GRAFICA DEL ENTRENAMIENTO
#-----------------------------------------------------------------------------------------

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


#-----------------------------------------------------------------------------------------
# EXPORTAMOS A VERSION TENSORFLOWLITE
#-----------------------------------------------------------------------------------------

# Convertimos el modelo a formato TensorFlow Lite
# converter = tf.lite.TFLiteConverter.from_keras_model(modelCNN)
# tflite_model = converter.convert()

# # Guardamos el modelo en un archivo
# with open('model.tflite', 'wb') as f:
#     f.write(tflite_model)
