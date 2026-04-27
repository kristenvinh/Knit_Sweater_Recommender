# dino_feature_extraction.py
import torch
from transformers import AutoModel, AutoImageProcessor
from numpy.linalg import norm
import numpy as np
import os
import threading
from PIL import Image
from YOLO_pose_crop import extract_and_crop_image


# Use CUDA (GPU) if available, otherwise CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "facebook/dinov2-base"
FEATURE_DIM = 768

model = None
processor = None
_model_load_error = None
_model_lock = threading.Lock()

def get_dino_components():
    """Thread-safe lazy loader for DINOv2 model and processor."""
    global model, processor, _model_load_error, FEATURE_DIM
    if model is not None and processor is not None:
        return model, processor
    if _model_load_error is not None:
        raise RuntimeError(f"DINOv2 models are not initialized: {_model_load_error}")

    with _model_lock:
        if model is not None and processor is not None:
            return model, processor

        try:
            os.environ['HF_HOME'] = os.path.expanduser('~/.cache/huggingface')
            loaded_model = AutoModel.from_pretrained(MODEL_ID).to(DEVICE)
            loaded_processor = AutoImageProcessor.from_pretrained(MODEL_ID)
            FEATURE_DIM = loaded_model.config.hidden_size
            model = loaded_model
            processor = loaded_processor
            return model, processor
        except Exception as e:
            _model_load_error = e
            raise RuntimeError(f"DINOv2 models are not initialized: {e}") from e

# Function to extract DINOv2 features from an image
def extract_features(img_path, return_cropped_image=False):
    try:
        current_model, current_processor = get_dino_components()
    except Exception as e:
        if return_cropped_image:
            return (img_path, Exception(str(e)), None)
        return (img_path, Exception(str(e)))
        
    try:
        image = None
        try:
            # 1. Load and crop the image using YOLO
            image = extract_and_crop_image(img_path)
        except Exception as e:
            print(f"Error during YOLO cropping: {e}. Will fall back to full image.")

        if image is None or not hasattr(image, "shape") or image.size == 0:
            image = np.array(Image.open(img_path).convert("RGB"))
        

        # 2. Process the image
        inputs = current_processor(images=image, return_tensors="pt").to(DEVICE)
        
        # 3. Run the model
        with torch.no_grad():
            
            outputs = current_model(**inputs)
        # 4. Get the feature vector
            feature_vector = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            
        # 5. Normalize the vector
        normalized_vector = feature_vector / norm(feature_vector)

        if return_cropped_image:
            return (img_path, normalized_vector, image)
        return (img_path, normalized_vector)
        
    except Exception as e:
        print(f"Failed to extract DINOv2 features for {img_path}: {e}")
        if return_cropped_image:
            return (img_path, e, None)
        return (img_path, e)
