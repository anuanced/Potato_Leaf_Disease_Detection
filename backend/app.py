"""
Potato Disease Classification - Flask Backend
Academic Project: Model Comparison System
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from PIL import Image
import io
import time

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin requests

# Disease class names (must match training order)
CLASS_NAMES = ['Early Blight', 'Late Blight', 'Healthy']

# Global variables to store loaded models
custom_cnn_model = None
mobilenet_model = None

def load_models():
    """Load both models at startup to avoid repeated loading"""
    global custom_cnn_model, mobilenet_model
    
    print("Loading Custom CNN model...")
    custom_cnn_model = load_model('models/potatoes.h5')
    print("Custom CNN loaded successfully!")
    
    print("Loading MobileNetV2 model...")
    mobilenet_model = load_model('models/mobilenetv2_potato.h5')
    print("MobileNetV2 loaded successfully!")

def preprocess_image_custom_cnn(image):
    """
    Preprocess image for Custom CNN
    - Resize to 256x256
    - Rescale pixel values to [0, 1]
    """
    img = image.resize((256, 256))
    img_array = np.array(img)
    img_array = img_array / 255.0  # Rescale to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def preprocess_image_mobilenet(image):
    """
    Preprocess image for MobileNetV2
    - Resize to 224x224
    - Apply MobileNetV2 preprocessing
    """
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # MobileNetV2 preprocessing
    return img_array

def predict_with_model(model, img_array, model_name):
    """
    Run prediction and measure inference time
    Returns: predicted_class, confidence, inference_time
    """
    start_time = time.time()
    predictions = model.predict(img_array, verbose=0)
    inference_time = time.time() - start_time
    
    # Get predicted class and confidence
    predicted_idx = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0]) * 100)
    predicted_class = CLASS_NAMES[predicted_idx]
    
    # Get all class probabilities
    class_probabilities = {
        CLASS_NAMES[i]: float(predictions[0][i] * 100) 
        for i in range(len(CLASS_NAMES))
    }
    
    return predicted_class, confidence, inference_time, class_probabilities

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "message": "Backend is running"})

@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint
    Accepts image file and returns predictions from both models
    """
    try:
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        
        # Read and open image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary (handle PNG with transparency)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # === Custom CNN Prediction ===
        img_cnn = preprocess_image_custom_cnn(image)
        cnn_class, cnn_conf, cnn_time, cnn_probs = predict_with_model(
            custom_cnn_model, img_cnn, "Custom CNN"
        )
        
        # === MobileNetV2 Prediction ===
        img_mobile = preprocess_image_mobilenet(image)
        mobile_class, mobile_conf, mobile_time, mobile_probs = predict_with_model(
            mobilenet_model, img_mobile, "MobileNetV2"
        )
        
        # Prepare response
        response = {
            "custom_cnn": {
                "prediction": cnn_class,
                "confidence": round(cnn_conf, 2),
                "inference_time_ms": round(cnn_time * 1000, 2),
                "class_probabilities": {k: round(v, 2) for k, v in cnn_probs.items()}
            },
            "mobilenet": {
                "prediction": mobile_class,
                "confidence": round(mobile_conf, 2),
                "inference_time_ms": round(mobile_time * 1000, 2),
                "class_probabilities": {k: round(v, 2) for k, v in mobile_probs.items()}
            },
            "agreement": cnn_class == mobile_class
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Load models before starting server
    load_models()
    
    # Start Flask server
    print("\n" + "="*50)
    print("🥔 Potato Disease Classification Backend")
    print("="*50)
    print("Server running on http://localhost:5000")
    print("Ready to accept predictions!\n")
    
    app.run(debug=True, port=5000)