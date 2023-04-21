

# Write TFLite model to a C source (or header) file
with open(modelo + '.h', 'w') as file:
  file.write(hex_to_c_array(tflite_model, c_model_name))