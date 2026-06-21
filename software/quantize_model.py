import tensorflow as tf
import numpy as np

# 1. Load the dataset and the trained model
print("Loading data and model...")
X_train = np.load("X_features.npy")
model = tf.keras.models.load_model("chainsaw_detector.keras")

# 2. Create the Representative Dataset Generator
# TensorFlow needs to see a few real examples to calibrate the integer math correctly
def representative_data_gen():
    # We will use the first 100 samples from our training data for calibration
    for input_value in tf.data.Dataset.from_tensor_slices(X_train).batch(1).take(100):
        # Ensure the data is cast to float32 as expected by the converter
        yield [tf.cast(input_value, tf.float32)]

# 3. Setup the TFLite Converter
print("Configuring the TFLite Converter for Full Integer Quantization...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen

# Force the inputs and outputs to be INT8 (crucial for pure hardware accelerators)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# 4. Convert and Save
print("Converting the model... (this might take a few seconds)")
tflite_quant_model = converter.convert()

with open("chainsaw_detector_quantized.tflite", "wb") as f:
    f.write(tflite_quant_model)

print("\nSuccess! Model quantized and saved as 'chainsaw_detector_quantized.tflite'")

# Print a quick size comparison
import os
keras_size = os.path.getsize("chainsaw_detector.keras") / 1024
tflite_size = os.path.getsize("chainsaw_detector_quantized.tflite") / 1024
print(f"Original Model Size:  {keras_size:.2f} KB")
print(f"Quantized Model Size: {tflite_size:.2f} KB")