import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, roc_curve, auc

# =========================
# SETTINGS
# =========================
MODEL_PATH = "colon_cancer_model_final.h5"
TEST_DIR = "Dataset/test"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
THRESHOLD = 0.35   # 🔥 NEW THRESHOLD

# =========================
# LOAD MODEL
# =========================
model = load_model(MODEL_PATH)
print("✅ Model loaded")

# =========================
# TEST DATA
# =========================
datagen = ImageDataGenerator(rescale=1./255)

test_gen = datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

# =========================
# PREDICTIONS
# =========================
y_true = test_gen.classes
y_pred_prob = model.predict(test_gen).ravel()

# Apply NEW threshold
y_pred = (y_pred_prob >= THRESHOLD).astype(int)

# =========================
# CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Normal", "Cancer"],
    yticklabels=["Normal", "Cancer"]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Threshold = 0.35)")
plt.savefig("confusion_matrix_t035.png")
plt.show()

# =========================
# CLASSIFICATION REPORT
# =========================
print("\n📊 Classification Report (Threshold = 0.35)")
print(classification_report(
    y_true, y_pred,
    target_names=["Normal", "Cancer"]
))

# =========================
# PRECISION–RECALL CURVE
# =========================
precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)

plt.figure(figsize=(6,5))
plt.plot(recall, precision, marker='.')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve")
plt.grid(True)
plt.savefig("precision_recall_curve_t035.png")
plt.show()

# =========================
# ROC CURVE
# =========================
fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.savefig("roc_curve_t035.png")
plt.show()

print("\n✅ NEW GRAPHS GENERATED (Threshold = 0.35)")