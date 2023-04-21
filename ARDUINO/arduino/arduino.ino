// Import TensorFlow stuff
#include "TensorFlowLite.h"
#include "tensorflow/lite/experimental/micro/kernels/micro_ops.h"
#include "tensorflow/lite/experimental/micro/micro_error_reporter.h"
#include "tensorflow/lite/experimental/micro/micro_interpreter.h"
#include "tensorflow/lite/experimental/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/version.h"

// Our model
#include "model_data.h"

// Figure out what's going on in our model
#define DEBUG 1

// TFLite globals, used for compatibility with Arduino-style sketches
namespace {
// 4. Configurar el registro
tflite::ErrorReporter* error_reporter = nullptr;
// 5. Cargue un modelo
const tflite::Model* model = nullptr;
// 8. Instanciar intérprete
tflite::MicroInterpreter* interpreter = nullptr;
// 10. Validar forma de entrada
TfLiteTensor* model_input = nullptr;
// 13. Obtenga la salida
TfLiteTensor* model_output = nullptr;

// Create an area of memory to use for input, output, and other TensorFlow
// arrays. You'll need to adjust this by combiling, running, and looking
// for errors.

// 7. Asignar memoria
constexpr int kTensorArenaSize = 5 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

}  // namespace

void setup() {
  // put your setup code here, to run once:

#if DEBUG
  while (!Serial)
#else

  // 4. Configurar el registro
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // 5. Cargue un modelo
  model = tflite::GetModel(sine_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model version does not match Schema");
    while(1);
  }

  // 6. tflite::AllOpsResolver resolver; REVISAR
  tflite::MicroMutableOpResolver<4> micro_op_resolver;
  micro_op_resolver.AddDepthwiseConv2D();
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddSoftmax();

  // 8. Instanciar intérprete
  static tflite::MicroInterpreter static_interpreter( 
    model, 
    micro_mutable_op_resolver, 
    tensor_arena, 
    kTensorArenaSize, 
    error_reporter);

  interpreter = &static_interpreter;

  
  // Allocate memory from the tensor_arena for the model's tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    while(1);
  }

  // Assign model input and output buffers (tensors) to pointers
  model_input = interpreter->input(0);
  model_output = interpreter->output(0);
  


  

}

void loop() {
  // put your main code here, to run repeatedly:







}
