# dino_feature_extraction.py
import torch
from transformers import AutoModel, AutoImageProcessor
from numpy.linalg import norm
import os
from YOLO_pose_crop import extract_and_crop_image


# Use CUDA (GPU) if available, otherwise CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "facebook/dinov2-base"

# Load the DINOv2 model and processor
try:
    # Set cache directory for Hugging Face models
    os.environ['HF_HOME'] = os.path.expanduser('~/.cache/huggingface')
    model = AutoModel.from_pretrained(MODEL_ID).to(DEVICE)
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    FEATURE_DIM = model.config.hidden_size

except Exception as e:
    model = None
    processor = None

# Function to extract DINOv2 features from an image
def extract_features(img_path):
    if model is None or processor is None:
        return (img_path, Exception("DINOv2 models are not initialized."))
        
    try:
        try: 
            # 1. Load and crop the image using YOLO
            image = extract_and_crop_image(img_path)

        except Exception as e:
            print(f"Error during YOLO cropping: {e}. Will fall back to full image.")
        

        # 2. Process the image
        inputs = processor(images=image, return_tensors="pt").to(DEVICE)
        
        # 3. Run the model
        with torch.no_grad():
            
            outputs = model(**inputs)
        # 4. Get the feature vector
            feature_vector = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            
        # 5. Normalize the vector
        normalized_vector = feature_vector / norm(feature_vector)

        return (img_path, normalized_vector)
        
    except Exception as e:
        print(f"Failed to extract DINOv2 features for {img_path}: {e}")
        return (img_path, e)
