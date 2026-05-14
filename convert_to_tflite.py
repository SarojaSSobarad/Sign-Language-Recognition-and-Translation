import tensorflow as tf

# Load the trained model (.h5)
model = tf.keras.models.load_model("model/keypoint_classifier/isl_word_model.h5")

# Convert it to .tflite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the converted model
with open("model/keypoint_classifier/isl_word_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Conversion successful! Saved as isl_word_model.tflite")
