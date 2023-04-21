import tensorflow as tf
from tensorflow import keras

# Cargar el modelo guardado en formato h5
model = keras.models.load_model('laMachin-v1.2.h5')

# Convertir el modelo a TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Guardar el modelo en formato TFLite
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

## sudo apt-get install xxd

## xxd -i model.tflite > model_data.cc
