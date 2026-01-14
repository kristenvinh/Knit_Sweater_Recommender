# xaiutil.py
import matplotlib
# --- CRITICAL FIX: Force non-interactive backend ---
# This prevents macOS from crashing when plotting in a background thread.
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tf_keras_vis.utils.scores import CategoricalScore
from tf_keras_vis.gradcam import Gradcam
import io

# --- 1. Dynamic YOLO Import ---
YOLO_AVAILABLE = False
extract_and_crop_image = None

possible_yolo_modules = ['YOLO_crop', 'YOLO_pose_crop']

for module_name in possible_yolo_modules:
    try:
        mod = __import__(module_name, fromlist=['extract_and_crop_image'])
        extract_and_crop_image = mod.extract_and_crop_image
        YOLO_AVAILABLE = True
        print(f"XAI: Successfully imported YOLO from {module_name}")
        break
    except ImportError:
        continue

if not YOLO_AVAILABLE:
    print("XAI Warning: YOLO crop module not found. Will fall back to full images.")


# --- 2. Helper Functions ---

def find_last_conv_layer(model):
    """
    Finds the *layer object* of the last convolutional layer in a Keras model.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer
            
    raise ValueError(f"Could not find a Conv2D layer in model {model.name}.")

def _preprocess_for_resnet(img_path):
    """
    Loads, crops (YOLO), and preprocesses an image for ResNet50 Keras.
    """
    img = None
    
    # A. Try YOLO Crop
    if YOLO_AVAILABLE:
        try:
            img = extract_and_crop_image(img_path)
            
            if img is not None:
                img = cv2.resize(img, (224, 224))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
            if img is not None and (img.shape[0] == 0 or img.shape[1] == 0):
                img = None
        except Exception as e:
            print(f"XAI Warning: YOLO failed for {img_path} ({e}). Falling back.")
            img = None

    # B. Fallback to Full Image (PIL)
    if img is None:
        try:
            pil_img = image.load_img(img_path, target_size=(224, 224))
            img = image.img_to_array(pil_img).astype(np.uint8)
        except Exception as e:
            print(f"XAI Error loading image {img_path}: {e}")
            return None, None

    # C. Prepare for Model
    img_expanded = np.expand_dims(img, axis=0)
    input_tensor = preprocess_input(img_expanded.astype('float32'))
    
    return input_tensor, img

# --- 3. Main XAI Generator ---

def generate_xai_heatmap_bytes(image_path, model, processor=None):
    """
    Generates a Grad-CAM heatmap for ResNet50 and returns it as JPEG bytes.
    """
    try:
        # 1. Preprocess
        input_tensor, original_img_array = _preprocess_for_resnet(image_path)
        
        if input_tensor is None:
            return None

        # 2. Make Prediction
        feature_vector = model.predict(input_tensor)
        top_feature_index = np.argmax(feature_vector[0])
        print(f"XAI: Visualizing feature index {top_feature_index}")

        # --- KEY FIX: Extract Inner Functional Model ---
        xai_model = model
        manual_pooling_needed = False
        
        if isinstance(model, tf.keras.Sequential):
            for layer in model.layers:
                if isinstance(layer, tf.keras.Model):
                    print(f"XAI: Found inner functional model '{layer.name}'. Using it for gradients.")
                    xai_model = layer
                    manual_pooling_needed = True
                    break
        
        if not hasattr(xai_model, 'output_names'):
            xai_model.output_names = ['output']

        # 3. Setup Grad-CAM
        
        # Define score function
        if manual_pooling_needed:
            # Manually replicate GlobalAveragePooling
            def score(output):
                target_channel = output[..., top_feature_index]
                return tf.reduce_mean(target_channel, axis=(1, 2))
        else:
            def score(output):
                return output[:, top_feature_index]
        
        # Find target layer
        last_conv_layer = find_last_conv_layer(xai_model)
        
        # Create Gradcam object
        gradcam = Gradcam(xai_model, clone=False) 
        
        # Generate Heatmap
        cam = gradcam(score, input_tensor, penultimate_layer=last_conv_layer.name)
        
        if isinstance(cam, list):
            cam = cam[0]
            
        cam = np.squeeze(cam)

        # 4. Visualization
        # Because we set backend='Agg', this create a virtual figure, not a window
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
        
        ax.imshow(original_img_array)
        
        heatmap = np.uint8(cm.jet(cam)[..., :3] * 255)
        im = ax.imshow(heatmap, cmap='jet', alpha=0.5)
        
        ax.set_title(f"Texture Focus (Feature {top_feature_index})")
        ax.axis('off')
        
        # Save to memory buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='jpg', bbox_inches='tight')
        plt.close(fig) # Important to free memory
        
        buf.seek(0)
        return buf.getvalue()

    except Exception as e:
        print(f"XAI Error: {e}")
        return None