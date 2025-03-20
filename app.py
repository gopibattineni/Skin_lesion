import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from celery import Celery
import time

# Initialize Flask app
app = Flask(__name__)

# Configure Celery for background tasks
app.config['CELERY_BROKER_URL'] = os.getenv('REDIS_URL', 'redis://localhost:6379/0')  # Use Render Redis URL
app.config['result_backend'] = os.getenv('REDIS_URL', 'redis://localhost:6379/0')  # Use Render Redis URL

celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

# Load the trained MobileNetV3 model (cached to avoid reloading)
MODEL_PATH = "my_model.keras"
model = load_model(MODEL_PATH)
print("MobileNetV3 Model loaded successfully!")

# Define class names (update based on your dataset)
classes = ["Actinic keratosis", "Basal cell carcinoma", "Benign keratosis", 
           "Chickenpox", "Cowpox", "Dermatofibroma", "HFMD", "Healthy", 
           "Measles", "Melanocytic nevus", "Melanoma", "Monkeypox", 
           "Squamous cell carcinoma", "Vascular lesion"]

# Function to make predictions on images
def model_predict(img_path):
    try:
        # Use OpenCV for faster image loading and preprocessing
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError("Unable to read image file.")
        
        img = cv2.resize(img, (224, 224))  # Resize to target size
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        x = np.expand_dims(img, axis=0)
        x = preprocess_input(x)

        # Make predictions
        preds = model.predict(x)

        # Get top 3 predictions
        top_3_indices = np.argsort(preds[0])[-3:][::-1]  # Sort and get top 3 indices
        top_3_scores = preds[0][top_3_indices] * 100  # Convert to percentage

        # Format predictions
        top_3_results = [(classes[i], top_3_scores[idx]) for idx, i in enumerate(top_3_indices)]
        return top_3_results
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

# Celery task for background prediction
@celery.task
def predict_task(img_path):
    try:
        result = model_predict(img_path)
        if result is None:
            raise ValueError("Prediction failed.")
        return result
    finally:
        # Clean up: Delete the uploaded file after processing
        if os.path.exists(img_path):
            os.remove(img_path)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def upload():
    if request.method == "POST":
        f = request.files["file"]

        # Ensure uploads directory exists
        basepath = os.path.dirname(__file__)
        upload_folder = os.path.join(basepath, "uploads")
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)

        # Check if the file is an image
        allowed_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}
        file_extension = os.path.splitext(f.filename)[1].lower()
        if file_extension not in allowed_extensions:
            return jsonify({"error": "Invalid file type. Please upload an image file."}), 400

        # Save the uploaded file
        file_path = os.path.join(upload_folder, secure_filename(f.filename))
        f.save(file_path)
        print(f"File saved at: {file_path}")

        # Start background prediction task
        task = predict_task.delay(file_path)
        return jsonify({"task_id": task.id}), 202

@app.route("/status/<task_id>", methods=["GET"])
def check_status(task_id):
    task = predict_task.AsyncResult(task_id)
    if task.ready():
        result = task.result()
        if result is None:
            return jsonify({"status": "failed", "error": "Prediction failed."}), 500
        return jsonify({"status": "completed", "result": result})
    else:
        return jsonify({"status": "pending"})

if __name__ == "__main__":
    app.run(debug=True)
