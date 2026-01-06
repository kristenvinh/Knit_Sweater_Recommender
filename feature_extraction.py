# feature_extractor.py

import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from numpy.linalg import norm
import cv2
from YOLO_crop import extract_and_crop_image

# Using a globally defined model to avoid reloading it for each function call
# Initialize base model
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
FEATURE_DIM = 1024

# Instead of the very last layer, grab the output of a middle block
# 'conv4_block6_out' is often a good sweet spot for texture vs object
intermediate_layer_model = Model(inputs=base_model.input, outputs=base_model.get_layer('conv4_block6_out').output)

model = Sequential([
    intermediate_layer_model,
    GlobalAveragePooling2D()
])

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