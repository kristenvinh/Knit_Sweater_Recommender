import numpy as np
import cv2  # Added for image loading and cropping
import os
import matplotlib.pyplot as plt
from ultralytics import YOLO  # Added for object detection

print("Initializing YOLOv8 models...")
try:
    yolo_model = YOLO('yolov8n-seg.pt')  # Load the segmentation model
    print("YOLOv8n model initialized.")
except Exception as e:
    print(f"Error initializing YOLO: {e}")
    print("Please ensure you have run 'pip install ultralytics'")
    yolo_model = None

import cv2
import numpy as np
# Make sure to import yolo_model and any other dependencies

def extract_and_crop_image(img_path):
    # 1. Load the image with OpenCV
    image = cv2.imread(img_path)
    
    # Check if image loaded correctly
    if image is None:
        print(f"  -> ERROR: Could not load image {img_path}.")
        return None 

    # --- DEFAULT FALLBACK ---
    cropped = image

    # Define minimum dimensions
    MIN_CROP_WIDTH = 50
    MIN_CROP_HEIGHT = 50

    # --- YOLO PROCESSING ---
    try:
        results = yolo_model(image, verbose=False)
        
        if results[0].masks and results[0].boxes:
            largest_area = 0
            best_mask_data = None

            for i, box in enumerate(results[0].boxes):
                if int(box.cls) == 0: # Class 0 is 'person'
                    area = (box.xyxy[0][2] - box.xyxy[0][0]) * (box.xyxy[0][3] - box.xyxy[0][1])
                    if area > largest_area:
                        largest_area = area
                        best_mask_data = results[0].masks.data[i]

            if best_mask_data is not None:
                best_mask = best_mask_data.cpu().numpy().astype(np.uint8)
                best_mask_resized = cv2.resize(best_mask, (image.shape[1], image.shape[0]))
                masked_image = cv2.bitwise_and(image, image, mask=best_mask_resized)
                y_indices, x_indices = np.where(best_mask_resized > 0)
                
                if y_indices.size > 0:
                    x1, x2 = x_indices.min(), x_indices.max()
                    y1, y2 = y_indices.min(), y_indices.max()
                    h = y2 - y1
                    crop_y1_new = max(y1, y1 + int(h * 0.1))
                    crop_y2_new = min(y2, y1 + int(h * 0.8))
                    
                    current_width = x2 - x1
                    current_height = crop_y2_new - crop_y1_new

                    if crop_y1_new < crop_y2_new and x1 < x2 and \
                       current_width >= MIN_CROP_WIDTH and current_height >= MIN_CROP_HEIGHT:
                        cropped = masked_image[crop_y1_new:crop_y2_new, x1:x2]
                    else:
                        print(f"  -> Cropped region too small. Falling back to masked image.")
                        cropped = masked_image
                else:
                    cropped = masked_image
            else:
                print(f"  -> No 'person' mask found. Falling back to original image.")
                cropped = image
        else:
            print(f"  -> No masks or boxes found. Falling back to original image.")
            cropped = image
    except Exception as yolo_e:
        print(f"  -> YOLO/Masking failed for {img_path}: {yolo_e}. Falling back to full image.")
        cropped = image
    
    # --- NEW: FINAL CHECK FOR MOSTLY-BLACK IMAGE ---
    # This check runs on whatever 'cropped' ended up as (torso, mask, or original)
    
    # Define your threshold (e.g., 95% black)
    FINAL_BLACK_THRESHOLD = 0.95 

    # Only run this check if we haven't already fallen back to the original image
    if cropped is not image: 
        try:
            gray_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            total_pixels = gray_cropped.size
            
            if total_pixels == 0:
                # This should be caught by MIN_CROP checks, but as a safeguard
                print(f"  -> Result was an empty (0 pixel) image. Falling back to original.")
                cropped = image
            else:
                # Count pixels that are NOT black (value > 0)
                non_black_pixels = np.count_nonzero(gray_cropped)
                
                # Calculate the percentage of black pixels
                percent_black = (total_pixels - non_black_pixels) / total_pixels
                
                if percent_black >= FINAL_BLACK_THRESHOLD:
                    print(f"  -> Result was {percent_black*100:.1f}% black. Falling back to original image.")
                    cropped = image # The final fallback
                    
        except cv2.error as e:
            # Handle potential errors if 'cropped' is somehow invalid
            print(f"  -> Error checking for black pixels: {e}. Falling back to original.")
            cropped = image

    # --- End of new check ---

    # Convert final BGR to RGB before returning
    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    print(f"  -> Final image shape: {cropped_rgb.shape}")
    
    return cropped_rgb