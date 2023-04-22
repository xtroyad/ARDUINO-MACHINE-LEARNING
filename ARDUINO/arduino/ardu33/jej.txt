// Importamos las librerías de TensorFlow
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"


#include "tensorflow/lite/micro/examples/micro_speech/micro_features/no_micro_features_data.h"
#include "tensorflow/lite/micro/examples/micro_speech/micro_features/yes_micro_features_data.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model.h"


// Inicializamos la cámara
OV767X cam;


// Se crea el modelo
const tflite::Model* model = ::tflite::GetModel(g_model);
if (model->version() != TFLITE_SCHEMA_VERSION) {
  TF_LITE_REPORT_ERROR(error_reporter,
    "Model provided is schema version %d not equal "
    "to supported version %d.\n",
    model->version(), TFLITE_SCHEMA_VERSION);
}

// Especificamos cuales operaciones en las neuronas han sido utilizadas
tflite::MicroMutableOpResolver resolver;
tflite::MicroMutableOpResolver<6> micro_op_resolver(&resolver);
micro_op_resolver.AddConv2D();
micro_op_resolver.AddMaxPool2D();
micro_op_resolver.AddDropout();
micro_op_resolver.AddFlatten();
micro_op_resolver.AddFullyConnected();
micro_op_resolver.AddSoftmax();

// Creamos el área de memoria que se va a usar para la entrada, salida
// o otros arrays de TensorFlow. Se debe de ir ajustando
contexpr int tensor_arena_size = 1 * 1024;
unint8_t tensor_arena[tensor_arena_size];

// Intérprete
tflite::MicroInterpreter interpreter(model, resolver, tensor_arena,tensor_arena_size);

void setup() {


  Serial.begin(9600);
  // Inicializamos la cámara con el modo QVGA (320x240)
  cam.begin(OV767X_MODE_QVGA_RGB565);
  // Esperamos a que la cámara esté lista
  delay(500);


  // Asignamos memoria de tensor_arena a los tensores del modelo
  interpreter.AllocateTensors();
  // Devuelve un puntero al tensor de entrada del modelo con índice 0.
  // Un tensor es un array con datos y el tensor de entrada es donde se
  // proporcionan los datos de entrada al modelo
  TfLiteTensor* input = interpreter.input(0);
  // Probamos que la entrada tiene las propiedades que esperábamos
  TF_LITE_MICRO_EXPECT_NE(nullptr, input);
  // La propiedad "dims" nos dice la forma del tensor. Tiene un elemento por
  // cada dimensión. Nuestra entrada es un tensor 2D que contiene 1 elemento,
  // por lo que "dims" debería tener un tamaño de 2.
  TF_LITE_MICRO_EXPECT_EQ(2, input->dims->size);
  // The value of each element gives the length of the corresponding tensor.
  // We should expect two single element tensors (one is contained within the
  // other).
  TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[1]);
  // La entrada es un valor de un puntero float de 32 bit
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteFloat32, input->type);
}

void loop() {

  // Capturamos la imagen de la cámara
  cam.getImage();
  // Creamos un objeto Image de la librería TFT_eSPI para mostrar la imagen en la pantalla
  TFT_eSPI_Image image = TFT_eSPI_Image(cam.getBuffer(), cam.width(), cam.height(), 0);
  // Redimensionamos la imagen a 64x64
  image.resize(64, 64);
  // Convertimos la imagen a escala de grises
  image.toGrayscale();
  // Obtenemos un puntero al array de bytes que contiene los valores de la imagen
  const uint8_t* image_data = image.getBitmap();
  // Asignamos los valores de la imagen al tensor de entrada del modelo
  for (int i = 0; i < 64*64; i++) {
    input->data.f[i] = ((float)image_data[i]);
  }



  // Ejecutamos el modelo
  // kTfLiteOk -> la inferencia se ejecutó correctamente
  // kTfLiteError -> la inferencia no se ejecutó
  TfLiteStatus invoke_status = interpreter.Invoke();
  if (invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed\n");
    
  }

  // Obtenemos el valor del tensor de salida mediante un float
  TfLiteTensor* output = interpreter.output(0);


  // Obtenemos el valor de salida del tensor
  float value = output->data.f[0];

  // Imprimimos el resultado
  Serial.print("Valor de salida: ");
  Serial.println(value);

  // Esperamos unos segundos antes de tomar otra imagen
  delay(2000);

}








