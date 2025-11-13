import torch
import cv2
import numpy as np
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# --- Import your YOLO cropper ---
try:
    from YOLO_pose_crop import extract_and_crop_image
    YOLO_AVAILABLE = True
except ImportError:
    print("Warning: 'YOLO_pose_crop.py' not found. Will fall back to full images.")
    YOLO_AVAILABLE = False
except Exception as e:
    print(f"Error importing YOLO_pose_crop: {e}. Will fall back to full images.")
    YOLO_AVAILABLE = False


def _preprocess_for_xai(img_path, processor):
    """
    Loads, YOLO-crops, and preprocesses an image for DINOv2 XAI.
    Returns the processed tensor and the original image (as a uint8 array) 
    for plotting.
    """
    img_pil = None
    
    if YOLO_AVAILABLE:
        try:
            # This function returns a NumPy array (or None if it fails badly)
            img_numpy_array = extract_and_crop_image(img_path) 
            
            if img_numpy_array is not None:
                img_pil = Image.fromarray(img_numpy_array)
            else:
                img_pil = None # It failed, so keep it None

            if img_pil is not None and (img_pil.width == 0 or img_pil.height == 0):
                print(f"Warning: YOLO returned a zero-size image for {img_path}. Falling back.")
                img_pil = None
        except Exception as e:
            print(f"Warning: YOLO failed for {img_path} ({e}). Falling back.")
            img_pil = None
    
    if img_pil is None:
        if YOLO_AVAILABLE:
            print(f"Info: Using full image for {img_path}.")
        try:
            img_pil = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error opening image {img_path}: {e}")
            return None, None
    
    # 1. Process for the model
    try:
        inputs = processor(images=img_pil, return_tensors="pt")
        input_tensor = inputs['pixel_values']
    except Exception as e:
        print(f"Error during Hugging Face processing step for {img_path}: {e}")
        return None, None 
    
    # 2. Create the plotting image
    
    # --- THIS IS THE FIX ---
    # We must resize the plot image to match the processor's crop size (e.g., 224x224)
    # This ensures the heatmap and the image have matching dimensions.
    
    # Get the target size from the processor's config
    # Use .get() for safety, defaulting to 224
    plot_size = processor.crop_size.get('height', 224) 
    
    # Resize the cropped PIL image to the exact input size of the model
    plot_img_pil = img_pil.resize((plot_size, plot_size))
    
    # Convert this *resized* image to the float array for plotting
    # show_cam_on_image expects a float array between 0 and 1
    plot_img_array = np.array(plot_img_pil).astype(np.float32) / 255.0
    # --- END OF FIX ---
    
    return input_tensor, plot_img_array


# Custom target class for a specific feature index
class FeatureTarget(ClassifierOutputTarget):
    def __init__(self, feature_index):
        super().__init__(feature_index)
        self.feature_index = feature_index

    def __call__(self, model_output):
        # model_output is a 1D tensor (e.g., shape [768])
        return model_output[self.feature_index]
    

# Reshape transform for Vision Transformers 
def _reshape_transform_vit(tensor):
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


# --- MAIN FUNCTION TO BE CALLED BY main.py ---

def generate_xai_heatmap_bytes(image_path: str, model, processor):
    """
    Generates a Grad-CAM heatmap for the DINOv2 model and returns it as JPEG bytes.
    
    Args:
        image_path: Path to the user's uploaded image.
        model: The globally loaded DINOv2 model.
        processor: The globally loaded DINOv2 image processor.
        
    Returns:
        Bytes of the JPEG heatmap image, or None on failure.
    """
    
    # 1. Preprocess the image and get the tensor
    input_tensor, plot_img_array = _preprocess_for_xai(image_path, processor)
    
    if input_tensor is None:
        print(f"XAI Error: Preprocessing failed for {image_path}")
        return None
        
    input_tensor = input_tensor.to(model.device)
    
    # 2. Get the feature vector to find the top feature
    print("XAI: Getting feature vector...")
    with torch.no_grad():
        outputs = model(pixel_values=input_tensor)
        # We use the mean of patch tokens, matching dino_feature_extraction.py
        feature_vector = outputs.last_hidden_state.mean(dim=1)
    
    top_feature_index = torch.argmax(feature_vector[0]).item()
    print(f"XAI: Visualizing for top feature index {top_feature_index}")

    # 3. Set up Grad-CAM
    
    # Wrap the model's feature extractor
    class DINOFeatureExtractor(torch.nn.Module):
        def __init__(self, model):
            super(DINOFeatureExtractor, self).__init__()
            self.model = model
        
        def forward(self, x):
            # Return the mean of patch tokens, same as our feature vector
            return self.model(pixel_values=x).last_hidden_state.mean(dim=1)

    feature_extractor_model = DINOFeatureExtractor(model)
    
    # Target layer for DINOv2 is the final layer norm
    target_layers = [model.encoder.layer[-1].norm1]

    cam = GradCAM(model=feature_extractor_model, 
                  target_layers=target_layers, 
                  reshape_transform=_reshape_transform_vit)

    targets = [FeatureTarget(top_feature_index)]

    # 4. Generate the heatmap
    print("XAI: Generating Grad-CAM heatmap...")
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :] # Get the 2D heatmap

    # 5. Create and encode the visualization
    try:
        print("XAI: Encoding heatmap...")
        # Use show_cam_on_image to blend the heatmap and image
        # plot_img_array is now (224, 224, 3) and float (0-1)
        # grayscale_cam is (224, 224)
        visualization = show_cam_on_image(plot_img_array, grayscale_cam, use_rgb=True)
        
        # Convert from RGB (0-255 float) to BGR (0-255 int) for cv2
        visualization_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
        
        # Encode the final image to JPEG format *in memory*
        success, encoded_image = cv2.imencode('.jpg', visualization_bgr)
        
        if not success:
            raise Exception("Failed to encode XAI heatmap.")
            
        return encoded_image.tobytes()
        
    except Exception as e:
        print(f"XAI Error during visualization: {e}")
        return None