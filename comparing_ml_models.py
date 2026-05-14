import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore")

# ---------------- CONFIG ----------------
DATA_PATH = "C:\\SignAI-Sign-Language-Recognition-and-Translation-main\\SignAI\\model\\keypoint_classifier\\keypoint.csv"
MAX_SAMPLES = 20000   
TEST_SIZE = 0.20
RANDOM_STATE = 42
PRINT_CM = False
# ----------------------------------------

# Safe CSV loading
data = pd.read_csv(DATA_PATH, header=None).values
y_all = data[:, 0].astype(int)          # labels
X_all = data[:, 1:].astype(np.float32)  # features

print(f"Loaded {X_all.shape[0]} samples, {X_all.shape[1]} features.")

# Subsample if dataset too large
if X_all.shape[0] > MAX_SAMPLES:
    print(f"Subsampling to {MAX_SAMPLES} samples (stratified)...")
    _, X_sub, _, y_sub = train_test_split(
        X_all, y_all, train_size=MAX_SAMPLES, stratify=y_all, random_state=RANDOM_STATE
    )
    X, y = X_sub, y_sub
    print(f"Subsampled shape: {X.shape}")
else:
    X, y = X_all, y_all

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

print(f"Train: {X_train.shape[0]}  Test: {X_test.shape[0]}")

results = []

def evaluate_and_report(name, y_true, y_pred, elapsed):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    print(f"\n{name} — Time: {elapsed:.1f}s  Accuracy: {acc*100:.2f}%  Precision: {prec*100:.2f}%")
    results.append((name, acc, prec, elapsed))


# ---------------- Model 1: MLP (sklearn) ----------------
t0 = time.time()
mlp = MLPClassifier(hidden_layer_sizes=(128,), max_iter=300, tol=1e-4, random_state=RANDOM_STATE)
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)
evaluate_and_report("MLP (sklearn)", y_test, y_pred, time.time()-t0)


# ---------------- Model 2: Random Forest ----------------
t0 = time.time()
rf = RandomForestClassifier(n_estimators=120, n_jobs=-1, random_state=RANDOM_STATE)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
evaluate_and_report("Random Forest", y_test, y_pred, time.time()-t0)


# ---------------- Model 3: Logistic Regression ----------------
t0 = time.time()
logreg = LogisticRegression(
    solver='saga',
    max_iter=500,
    n_jobs=-1,
    multi_class='multinomial',
    random_state=RANDOM_STATE
)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
evaluate_and_report("Logistic Regression", y_test, y_pred, time.time()-t0)


# ---------------- Summary ----------------
print("\n\n====== SUMMARY ======")
print(f"Dataset used: {X.shape[0]} samples | features: {X.shape[1]}")
for name, acc, prec, t in results:
    print(f"{name:25} → Acc: {acc*100:.2f}% | Prec: {prec*100:.2f}% | Time: {t:.1f}s")
print("=====================")
