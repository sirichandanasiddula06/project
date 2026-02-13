from flask import Flask, render_template, request
import os
import gdown
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# ------------------------------
# Create uploads folder
# ------------------------------
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ------------------------------
# Model Download & Loading
# ------------------------------
MODEL_PATH = "colon_cancer_model.h5"
MODEL_URL = "https://drive.google.com/uc?id=14TKeAwyN13K9A0oz4XhbvdmgBFE4Kg18"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

# ------------------------------
# Routes
# ------------------------------
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return render_template("index.html", prediction="No file uploaded")

    img_file = request.files["file"]

    if img_file.filename == "":
        return render_template("index.html", prediction="No file selected")

    img_path = os.path.join(UPLOAD_FOLDER, img_file.filename)
    img_file.save(img_path)

    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0

    pred = model.predict(x)

    result = "CANCER DETECTED" if pred[0][0] > 0.5 else "NORMAL"

    return render_template("index.html", prediction=result)

# ------------------------------
# Run App
# ------------------------------
if __name__ == "__main__":
    app.run(debug=True)