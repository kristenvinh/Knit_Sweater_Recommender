import numpy as np
import cv2  # Added for image loading and cropping
import os
import matplotlib.pyplot as plt
from ultralytics import YOLO  # Added for object detection

print("Initializing YOLOv8 models...")
try:
    # Load the SEGMENTATION model (for masks)
    yolo_seg_model = YOLO('yolov8n-seg.pt')
    print("YOLOv8n-seg model initialized.")
    
    # Load the POSE model (for keypoints)
    yolo_pose_model = YOLO('yolov8n-pose.pt')
    print("YOLOv8n-pose model initialized.")
    
except Exception as e:
    print(f"Error initializing YOLO: {e}")
    print("Please ensure you have run 'pip install ultralytics'")
    yolo_seg_model = None
    yolo_pose_model = None

import cv2
import numpy as np

KEYPOINT_CONF_THRESH = 0.5  # Confidence threshold for pose keypoints
TORSO_KEYPOINTS = [5, 6, 11, 12] # COCO indices: l_shoulder, r_shoulder, l_hip, r_hip


def _get_keypoint_crop(keypoints, confidences, img_shape):
    """
    Helper function to create a bounding box from torso keypoints,
    ensuring a minimum crop size.
    """
    h, w = img_shape
    
    # --- ADDED: Define minimum crop size ---
    # Adjust these values as needed
    MIN_CROP_WIDTH = .25 * w  # 25% of image width
    MIN_CROP_HEIGHT = .25 * h  # 25% of image height

    torso_points_xy = []
    
    for kpt_index in TORSO_KEYPOINTS:
        if confidences[kpt_index] > KEYPOINT_CONF_THRESH:
            torso_points_xy.append(keypoints[kpt_index])
            
    # Need at least 2 valid points to define a box
    if len(torso_points_xy) < 4:
        return None # Signal failure

    torso_points_xy = np.array(torso_points_xy)
    
    # Get the min/max x and y coordinates
    x1 = int(np.min(torso_points_xy[:, 0]))
    y1 = int(np.min(torso_points_xy[:, 1]))
    x2 = int(np.max(torso_points_xy[:, 0]))
    y2 = int(np.max(torso_points_xy[:, 1]))
    
    # Add 20% padding to the box
    box_h, box_w = y2 - y1, x2 - x1
    
    # Avoid division by zero or tiny boxes if h/w is 0
    if box_h <= 0 or box_w <= 0:
        return None # Not a valid box to pad
        
    padding_y = int(box_h * 0.20)
    padding_x = int(box_w * 0.20)
    
    # Apply padding and clamp to image boundaries
    x1 = max(0, x1 - padding_x)
    y1 = max(0, y1 - padding_y)
    x2 = min(w, x2 + padding_x)
    y2 = min(h, y2 + padding_y)
    
    # --- MODIFIED: Final check for validity AND minimum size ---
    
    # Calculate the final dimensions *after* padding and clamping
    final_width = x2 - x1
    final_height = y2 - y1
    
    # Check if the final crop meets the minimum size requirements
    if final_height >= MIN_CROP_HEIGHT and final_width >= MIN_CROP_WIDTH:
        print(f"Final crop size: width={final_width}, height={final_height}")
        print(f"Original size: width={w}, height={h}")
        return (y1, y2, x1, x2)
    else:
        # Fails if:
        # 1. The box is invalid (e.g., y1 >= y2)
        # 2. The box is valid but too small (e.g., width < MIN_CROP_WIDTH)
        return None
    

def _get_mask_crop(best_mask_resized):
    """Helper function to create a bounding box from a mask."""
    y_indices, x_indices = np.where(best_mask_resized > 0)
    
    if y_indices.size > 0:
        x1, x2 = x_indices.min(), x_indices.max()
        y1, y2 = y_indices.min(), y_indices.max()

        h = y2 - y1
        crop_y1_new = max(y1, y1 + int(h * 0.1)) 
        crop_y2_new = min(y2, y1 + int(h * 0.8))
        
        if crop_y1_new < crop_y2_new and x1 < x2:
            return (crop_y1_new, crop_y2_new, x1, x2)
    
    return None

def extract_and_crop_image(image_path):
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"  -> ERROR: Could not load image {image_path}.")
        # Return a copy of a blank image or handle as needed
        blank_image = np.zeros((100, 100, 3), dtype=np.uint8)
        return blank_image, blank_image 

    cropped_bgr = image # Default fallback
    img_height, img_width = image.shape[:2]

    # --- YOLO PROCESSING ---
    try:
        seg_results = yolo_seg_model(image, verbose=False)
        pose_results = yolo_pose_model(image, verbose=False)
        
        best_mask_resized = None
        best_keypoints = None
        best_keypoint_confs = None

        ### 1. Get Best MASK from Segmentation Model ###
        if seg_results[0].masks and seg_results[0].boxes:
            largest_area = 0
            best_mask_data = None
            for i, box in enumerate(seg_results[0].boxes):
                if int(box.cls) == 0: # Class 0 is 'person'
                    area = (box.xyxy[0][2] - box.xyxy[0][0]) * (box.xyxy[0][3] - box.xyxy[0][1])
                    if area > largest_area:
                        largest_area = area
                        best_mask_data = seg_results[0].masks.data[i]

            if best_mask_data is not None:
                best_mask = best_mask_data.cpu().numpy().astype(np.uint8)
                best_mask_resized = cv2.resize(best_mask, (img_width, img_height))

        ### 2. Get Best KEYPOINTS from Pose Model ###
        if pose_results[0].keypoints and pose_results[0].boxes:
            largest_area = 0
            for i, box in enumerate(pose_results[0].boxes):
                if int(box.cls) == 0: # Class 0 is 'person'
                    area = (box.xyxy[0][2] - box.xyxy[0][0]) * (box.xyxy[0][3] - box.xyxy[0][1])
                    if area > largest_area:
                        largest_area = area
                        best_keypoints = pose_results[0].keypoints[i].xy[0].cpu().numpy()
                        best_keypoint_confs = pose_results[0].keypoints[i].conf[0].cpu().numpy()

        
        ### 3. Combine and Crop ###
        if best_mask_resized is not None:
            masked_image = cv2.bitwise_and(image, image, mask=best_mask_resized)
            crop_box = None
            
            if best_keypoints is not None:
                crop_box = _get_keypoint_crop(best_keypoints, best_keypoint_confs, (img_height, img_width))
                if crop_box:
                    print(f"  -> Smart crop (Pose) for {image_path}") # Fixed: img_path -> image_path
            
            if crop_box is None:
                print(f"  -> Pose failed. Fallback crop (Mask) for {image_path}") # Fixed: img_path -> image_path
                crop_box = _get_mask_crop(best_mask_resized)
            
            if crop_box:
                y1, y2, x1, x2 = crop_box
                cropped_bgr = masked_image[y1:y2, x1:x2]
            else:
                # Fallback to the full masked image if no valid crop box
                cropped_bgr = masked_image
        
        # --- ADDED: FINAL CHECK FOR MOSTLY-BLACK IMAGE ---
        
        # Define your threshold (e.g., 95% black)
        FINAL_BLACK_THRESHOLD = 0.95 

        # Only run this check if we haven't already fallen back to the original image
        if cropped_bgr is not image: 
            try:
                gray_cropped = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2GRAY)
                total_pixels = gray_cropped.size
                
                if total_pixels == 0:
                    # This should be caught by MIN_CROP checks, but as a safeguard
                    print(f"  -> Result was an empty (0 pixel) image. Falling back to original.")
                    cropped_bgr = image
                else:
                    # Count pixels that are NOT black (value > 0)
                    non_black_pixels = np.count_nonzero(gray_cropped)
                    
                    # Calculate the percentage of black pixels
                    percent_black = (total_pixels - non_black_pixels) / total_pixels
                    
                    if percent_black >= FINAL_BLACK_THRESHOLD:
                        print(f"  -> Result was {percent_black*100:.1f}% black. Falling back to original image.")
                        cropped_bgr = image # The final fallback
                        
            except cv2.error as e:
                # Handle potential errors if 'cropped_bgr' is somehow invalid
                print(f"  -> Error checking for black pixels: {e}. Falling back to original.")
                cropped_bgr = image

        # --- End of new check ---

        return cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
    
    except Exception as yolo_e:
        print(f"  -> YOLO/Masking failed for {image_path}: {yolo_e}. Falling back to full image.") # Fixed: img_path -> image_path
        cropped_bgr = image
        return cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)