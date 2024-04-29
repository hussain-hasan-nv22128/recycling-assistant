import cv2
import numpy as np

def resize_image(image, target_size=(224, 224)):
    resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    return resized_image

def normalize_image(image):
    normalized_image = image.astype('float32') / 255.0  # Normalize pixel values to the range [0, 1]
    return normalized_image

def preprocess_image(image_path):
    image = cv2.imread(image_path)  # Load image using OpenCV
    if image is None:
        return None
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image = resize_image(image)  # Resize image to target size
    image = normalize_image(image)  # Normalize pixel values
    
    return image