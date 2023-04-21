// Importamos las librerías de TensorFlow
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#include "model.h"

void setup() {
  // Variable que se pasará al intérprete para escribir registros
  tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  // Se crea el modelo 
  const tflite::Model* model = ::tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
        "Model provided is schema version %d not equal "
        "to supported version %d.\n",
        model->version(), TFLITE_SCHEMA_VERSION);
  }

  // Especificamos cuales operaciones en las neuronas han sido utilizadas
  tflite::MicroMutableOpResolver  resolver;
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
  tflite::MicroInterpreter interpreter(model, resolver, tensor_arena,
                                     tensor_arena_size, error_reporter);

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

  // Establecemos el contenido del tensor de entrada
  input->data.f[0] = 0.;
}

void loop() {
  // Ejecutamos el modelo
  // kTfLiteOk -> la inferencia se ejecutó correctamente
  // kTfLiteError -> la inferencia no se ejecutó
  TfLiteStatus invoke_status = interpreter.Invoke();
  if (invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed\n");
  }

  // Obtenemos el valor del tensor de salida mediante un float
  TfLiteTensor* output = interpreter.output(0);
  TF_LITE_MICRO_EXPECT_EQ(2, output->dims->size); // Verifica que el tensor de salida tenga dos dimensiones
  TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[0]); // Verifica que el tensor de entrada tenga dimension 1
  TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[1]); // Verifica que el tensor de entrada tenga dimension 1
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteFloat32, output->type); // Verifica que el tipo es float

  // Obtenemos el valor de salida del tensor
  float value = output->data.f[0];
  // Comprobamos si ese valor está en ese rango
  TF_LITE_MICRO_EXPECT_NEAR(0., value, 0.05);

}
