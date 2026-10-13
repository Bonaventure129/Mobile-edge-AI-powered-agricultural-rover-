import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"  # This must be at the very top

import tensorflow as tf
import tensorflow_hub as hub
import kagglehub

print("Downloading CropNet Cassava model...")
path = kagglehub.model_download("google/cropnet/tensorFlow2/classifier-cassava-disease-v1")
print(f"Model downloaded to: {path}")

print("Wrapping model in Keras and converting to TFLite...")
model = tf.keras.Sequential([
    hub.KerasLayer(path, input_shape=(224, 224, 3))
])

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

output_filename = 'crop_disease_quantized.tflite'
with open(output_filename, 'wb') as f:
    f.write(tflite_model)

print(f"Success! Saved as {output_filename}")
