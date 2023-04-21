import cv2
import numpy as np
from tensorflow import keras
import tensorflow as tf

# Cargar el modelo guardado
model = keras.models.load_model('laMachin.h5')

# Inicializar la cámara
cap = cv2.VideoCapture(0)

# Variable para indicar si se debe tomar la foto
take_photo = False

# Definir los nombres de las clases
class_names =['Arandelas', 'Mariposas', 'Tornillos', 'Tuercas']

# Definir el tamaño de las imágenes
img_height, img_width = 100, 100

while True:
    # Capturar un fotograma
    ret, frame = cap.read()

    # Preprocesar la imagen
    img = cv2.resize(frame, (img_height, img_width))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)

    # Realizar la predicción si se debe tomar la foto
    if take_photo:
        # Mostrar la imagen preprocesada
        cv2.imshow('processed_image', img[0])
        
        predictions = model.predict(img)
        score = tf.nn.softmax(predictions[0])

        print(
            "Esta imagen probablemente pertenece a {} con una confianza del {:.2f}%."
            .format(class_names[np.argmax(score)], 100 * np.max(score))
        )

        take_photo = False

    # Mostrar la imagen en una ventana
    cv2.imshow('frame', frame)

    # Detectar la tecla presionada
    key = cv2.waitKey(1)
    if key == ord('q'):
        # Salir si se pulsa la tecla 'q'
        break
    elif key == ord('p'):
        # Cambiar la variable a True si se presiona la tecla 'p'
        take_photo = True

# Liberar la cámara y cerrar la ventana
cap.release()
cv2.destroyAllWindows()
