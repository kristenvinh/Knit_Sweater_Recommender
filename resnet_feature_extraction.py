# feature_extractor.py

import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.models import Sequential
from numpy.linalg import norm
import cv2
from YOLO_pose_crop import extract_and_crop_image

# --- Global Model Initialization ---
# Initialize the model once when this module is imported.
# This is efficient because each worker process will inherit this model.
print("Initializing model...")
model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = Sequential([model, GlobalMaxPooling2D()])
print("Model initialized.")

def extract_features(img_path):
    """
    Extracts features from an image using the globally defined model.
    On success, returns a tuple: (img_path, feature_vector)
    On failure, returns a tuple: (img_path, Exception)
    """
    try: 
        img = extract_and_crop_image(img_path)
        img = cv2.resize(img, (224, 224))  # Resize to model input size
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
        result = model.predict(preprocessed_img).flatten()
        normalized_result = result / norm(result)
        return (img_path, normalized_result)
    except Exception as e:
        print (f"Error processing crop at {img_path}: {e}")
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
        result = model.predict(preprocessed_img).flatten()
        normalized_result = result / norm(result)
        return (img_path, normalized_result)
    except Exception as e:
        return (img_path, e)