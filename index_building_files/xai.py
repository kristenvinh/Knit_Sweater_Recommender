import hnswlib
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import cv2
import torch
from PIL import Image
import matplotlib.pyplot as plt
import os
import cv2  # Used for visualization
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from dino_feature_extraction import extract_features
from transformers import AutoModel, AutoImageProcessor
from YOLO_pose_crop import extract_and_crop_image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "facebook/dinov2-base"  # DINOv2 Model

# --- Import your YOLO cropper ---
try:
    from YOLO_pose_crop import extract_and_crop_image
    YOLO_AVAILABLE = True
except ImportError:
    print("Warning: 'YOLO_crop.py' not found. Will fall back to resizing full images.")
    YOLO_AVAILABLE = False
except Exception as e:
    print(f"Error importing YOLO_crop: {e}. Will fall back to resizing full images.")
    YOLO_AVAILABLE = False

from PIL import Image  # <-- Make sure this is at the top of your file
import numpy as np     # <-- Make sure this is imported

# (Your other imports and YOLO_pose_crop functions...)

def preprocess_image_for_dino(img_path, processor):
    """
    Loads, YOLO-crops, and preprocesses an image for DINOv2.
    Returns the processed tensor and the original image (as a uint8 array) for plotting.
    """
    img_pil = None  # Initialize as None
    
    if YOLO_AVAILABLE:
        # try:
            # This function returns a NumPy array (or None if it fails badly)
            img_numpy_array = extract_and_crop_image(img_path) 
            
            # --- START FIX ---
            # Convert the NumPy array to a PIL Image
            if img_numpy_array is not None:
                img_pil = Image.fromarray(img_numpy_array)
            else:
                img_pil = None # It failed, so keep it None
            # --- END FIX ---

            # Now this check will work, because img_pil is a proper PIL Image
            if img_pil is not None and (img_pil.width == 0 or img_pil.height == 0):
                print(f"Warning: YOLO returned a zero-size image for {img_path}. Falling back.")
                img_pil = None

        # except Exception as e:
        #     ... (your exception block) ...
    
    if img_pil is None:
        if YOLO_AVAILABLE:
            print(f"Warning: YOLO found no object in {img_path} or failed. Using full image.")
        try:
            img_pil = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error opening image {img_path}: {e}")
            return None, None
    
    # ... (Rest of your function is fine) ...
    
    # 1. Process for the model
    try:
        inputs = processor(images=img_pil, return_tensors="pt").to(DEVICE)
        input_tensor = inputs['pixel_values']
    except Exception as e:
        print(f"Error during Hugging Face processing step for {img_path}: {e}")
        return None, None 
    
    # 2. Create the plotting image
    plot_size = processor.crop_size.get('height', 224)
    if plot_size is None:
        print(f"Error: Processor config missing 'crop_size'. Defaulting to 224.")
        plot_size = 224
        
    plot_img = img_pil.resize((int(plot_size), int(plot_size))) 
    plot_img_array = np.array(plot_img).astype(np.uint8) 
    
    return input_tensor, plot_img_array


# Custom target class for a specific feature index (IDENTICAL to CLIP version)
class FeatureTarget(ClassifierOutputTarget):
    def __init__(self, feature_index):
        super().__init__(feature_index)
        self.feature_index = feature_index

    def __call__(self, model_output):
        # model_output is a 1D tensor (e.g., shape [768])
        return model_output[self.feature_index]
    

os.environ['HF_HOME'] = os.path.expanduser('~/.cache/huggingface')
        # Use AutoImageProcessor to automatically get the correct processor
processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(MODEL_ID).to(DEVICE)
model.eval() # Set model to evaluation mode

# Reshape transform for Vision Transformers 
def reshape_transform_vit(tensor):
    """
    Reshapes the 3D ViT tensor to a 4D tensor for Grad-CAM.
    Input shape: (batch_size, num_tokens, embedding_dim)
    Output shape: (batch_size, embedding_dim, height, width)
    """
    if len(tensor.shape) == 3:
        # Assumes token 0 is the [CLS] token
        patch_tokens = tensor[:, 1:, :]
        num_patches = patch_tokens.shape[1]
        h = w = int(num_patches**0.5)
        
        # Reshape to (batch, H, W, C)
        reshaped_tensor = patch_tokens.reshape(-1, h, w, tensor.shape[-1])
        
        # Permute to (batch, C, H, W) as expected by pytorch-grad-cam
        reshaped_tensor = reshaped_tensor.permute(0, 3, 1, 2)
        return reshaped_tensor
    return tensor


def get_XAI_for_dino(TEST_IMAGE_PATH):
    # 1. Preprocess the image
    print(f"Processing image: {TEST_IMAGE_PATH}")
    input_tensor, plot_img_array = preprocess_image_for_dino(TEST_IMAGE_PATH, processor)
    
    if input_tensor is None:
        print(f"Skipping {TEST_IMAGE_PATH} due to processing error.")
        return

    # 2. Get the feature vector to find the top feature
    print("Getting feature vector...")
    with torch.no_grad():
        outputs = model(pixel_values=input_tensor)
        feature_vector = outputs.pooler_output # This is the [CLS] token output
    
    top_feature_index = torch.argmax(feature_vector[0]).item()
    top_feature_value = feature_vector[0, top_feature_index].item()
    print(f"Visualizing Grad-CAM for top feature: index {top_feature_index} (value: {top_feature_value:.4f})")

    # 3. Set up Grad-CAM
    
    # Wrap the model's feature extractor
    class DINOFeatureExtractor(torch.nn.Module):
        def __init__(self, model):
            super(DINOFeatureExtractor, self).__init__()
            self.model = model
        
        def forward(self, x):
            # Return the pooler_output, which is the feature vector
            return self.model(pixel_values=x).pooler_output

    feature_extractor_model = DINOFeatureExtractor(model)
    
    # Target layer for DINOv2 is the final layer norm

    target_layers = [model.encoder.layer[-1].norm1]    # --- End FIX ---

    cam = GradCAM(model=feature_extractor_model, 
                  target_layers=target_layers, 

                  reshape_transform=reshape_transform_vit) # Use the same ViT reshape

    targets = [FeatureTarget(top_feature_index)]

    # 4. Generate the heatmap
    print("Generating Grad-CAM heatmap...")
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    
    grayscale_cam = grayscale_cam[0, :] # Get the 2D heatmap

    # 5. Create and save the visualization
    
    # Resize the heatmap to match the image
    resized_heatmap = cv2.resize(grayscale_cam, (plot_img_array.shape[1], plot_img_array.shape[0]))

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 10))
    
    ax.imshow(plot_img_array) 
    im = ax.imshow(resized_heatmap, cmap='jet', alpha=0.5) 
    
    ax.set_title(f"DINOv2 Grad-CAM")
    ax.axis('off')

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Feature Activation Intensity')
    # --- End ADD COLOR BAR ---
    plt.tight_layout()
    output_path = os.path.splitext(TEST_IMAGE_PATH)[0] + "_dino_gradcam.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Grad-CAM visualization saved to: {output_path}")