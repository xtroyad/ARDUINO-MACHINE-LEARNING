import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import numpy as np
import os

# Cargar el modelo guardado
modelo = load_model('ML.h5')

# Obtener la lista de archivos en la carpeta de pruebas
folder_path = '/Pruebas'
files = os.listdir(folder_path)

# Loop a través de cada archivo de imagen y hacer predicciones
for file in files:
    # Cargar la imagen
    img_path = os.path.join(folder_path, file)
    img = keras.preprocessing.image.load_img(
        img_path,
        target_size=(180,180)
    )

    # Convertir la imagen en un array numpy
    img_array = keras.preprocessing.image.img_to_array(img)

    # Redimensionar la imagen a la forma de entrada esperada por el modelo
    img_array = tf.image.resize(img_array, [180, 180])
    img_array = tf.reshape(img_array, [1, -1])  # aplanar en un solo vector

    # Normalizar los valores de píxel (igual que en el entrenamiento)
    img_array = img_array / 255.0

    # Realizar la predicción con el modelo cargado
    predictions = modelo.predict(img_array)
    class_names = ['Arandela', 'Mariposa', 'Tornillo', 'Tuerca'] 
    predicted_class = class_names[np.argmax(predictions[0])]

    # Imprimir el nombre del archivo y la predicción
    print(f'{file}: {predicted_class}')
