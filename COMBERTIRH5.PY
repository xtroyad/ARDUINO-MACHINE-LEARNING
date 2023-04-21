
import tensorflow as tf
from tensorflow import keras

# Cargar el modelo guardado en formato h5
model = keras.models.load_model('laMachin-BETA.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.experimental_new_converter = True
converter.experimental_new_quantizer = True
quantized_model = converter.convert()

# Guardar el modelo cuantizado en un archivo
with open('modelo.tflite', 'wb') as f:
    f.write(quantized_model)

## sudo apt-get install xxd
## xxd -i model.tflite > model_data.cc
