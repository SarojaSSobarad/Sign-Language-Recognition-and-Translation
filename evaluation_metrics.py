import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score

# -----------------------------
# 1. PATHS
# -----------------------------
DATA_PATH = r"C:\SignAI-Sign-Language-Recognition-and-Translation-main\SignAI\model\keypoint_classifier\keypoint.csv"
MODEL_PATH = r"C:\SignAI-Sign-Language-Recognition-and-Translation-main\SignAI\model\keypoint_classifier\keypoint_classifier.tflite"

# -----------------------------
# 2. LOAD CSV DATA
# -----------------------------
data = pd.read_csv(DATA_PATH, header=None).values

# COLUMN 0 = LABEL
y = data[:, 0].astype(int)

# OTHER COLUMNS = FEATURES
X = data[:, 1:].astype(np.float32)

print("Data Loaded Successfully")
print("Samples:", len(X))
print("Features:", X.shape[1])

# -----------------------------
# 3. LOAD TFLITE MODEL
# -----------------------------
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

model_input_size = input_details[0]['shape'][1]

# Ensure correct number of features
X = X[:, :model_input_size]

# -----------------------------
# 4. RUN INFERENCE
# -----------------------------
predictions = []

for sample in X:
    sample = sample.reshape(1, model_input_size).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], sample)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    predictions.append(np.argmax(output))

predictions = np.array(predictions)

# -----------------------------
# 5. METRICS
# -----------------------------
accuracy = accuracy_score(y, predictions)
precision = precision_score(y, predictions, average="macro", zero_division=0)

print("\n-------- MODEL EVALUATION --------")
print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print("----------------------------------")
