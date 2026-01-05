import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Disease class mapping
CLASS_NAMES = ['Early Blight', 'Late Blight', 'Healthy']

def validate_image(image):
    """
    Validate uploaded image
    Returns: True if valid, False otherwise
    """
    try:
        # Check if image can be opened
        if not isinstance(image, Image.Image):
            return False
        
        # Check image dimensions
        width, height = image.size
        if width < 50 or height < 50:
            return False
        
        # Check if image is too large (optional)
        if width > 4000 or height > 4000:
            return False
        
        return True
    except:
        return False

def preprocess_for_custom_cnn(image, target_size=(256, 256)):
    """
    Preprocess image for Custom CNN model
    
    Args:
        image: PIL Image object
        target_size: Target size tuple (width, height)
    
    Returns:
        Preprocessed numpy array ready for prediction
    """
    # Resize image
    img = image.resize(target_size)
    
    # Convert to array
    img_array = np.array(img)
    
    # Normalize to [0, 1] range
    img_array = img_array / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def preprocess_for_mobilenet(image, target_size=(224, 224)):
    """
    Preprocess image for MobileNetV2 model
    
    Args:
        image: PIL Image object
        target_size: Target size tuple (width, height)
    
    Returns:
        Preprocessed numpy array ready for prediction
    """
    # Resize image
    img = image.resize(target_size)
    
    # Convert to array
    img_array = np.array(img)
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    # Apply MobileNetV2 specific preprocessing
    img_array = preprocess_input(img_array)
    
    return img_array

def get_prediction_details(predictions, class_names=CLASS_NAMES):
    """
    Extract prediction details from model output
    
    Args:
        predictions: Model prediction output (numpy array)
        class_names: List of class names
    
    Returns:
        Dictionary with prediction details
    """
    # Get predicted class index
    predicted_idx = np.argmax(predictions[0])
    
    # Get confidence score
    confidence = float(np.max(predictions[0]) * 100)
    
    # Get predicted class name
    predicted_class = class_names[predicted_idx]
    
    # Get all class probabilities
    class_probabilities = {
        class_names[i]: float(predictions[0][i] * 100) 
        for i in range(len(class_names))
    }
    
    return {
        'predicted_class': predicted_class,
        'confidence': round(confidence, 2),
        'class_probabilities': class_probabilities
    }

def compare_predictions(pred1, pred2):
    """
    Compare two predictions and return agreement status
    
    Args:
        pred1: First prediction dictionary
        pred2: Second prediction dictionary
    
    Returns:
        Boolean indicating if predictions agree
    """
    return pred1['predicted_class'] == pred2['predicted_class']

def format_inference_time(time_seconds):
    """
    Format inference time for display
    
    Args:
        time_seconds: Time in seconds
    
    Returns:
        Formatted string with time in milliseconds
    """
    time_ms = time_seconds * 1000
    return f"{time_ms:.2f} ms"