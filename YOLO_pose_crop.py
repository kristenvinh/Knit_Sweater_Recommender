import numpy as np
import cv2 
from ultralytics import YOLO  

yolo_seg_model = None
yolo_pose_model = None
yolo_init_error = None

import cv2
import numpy as np

KEYPOINT_CONF_THRESH = 0.5  # Confidence threshold for pose keypoints
TORSO_KEYPOINTS = [5, 6, 11, 12] # Indices for torso crop: l_shoulder, r_shoulder, l_hip, r_hip
MIN_KEYPOINTS_REQUIRED = 2  # Relax from 4 to 2+ keypoints for robustness


def _ensure_yolo_models():
    """Lazy-load YOLO models so module import stays lightweight."""
    global yolo_seg_model, yolo_pose_model, yolo_init_error
    if yolo_seg_model is not None and yolo_pose_model is not None:
        return True
    if yolo_init_error is not None:
        return False

    try:
        yolo_seg_model = YOLO('yolov8n-seg.pt')
        yolo_pose_model = YOLO('yolov8n-pose.pt')
        return True
    except Exception as e:
        yolo_init_error = e
        print(f"Error initializing YOLO: {e}")
        return False



def _get_keypoint_crop(keypoints, confidences, img_shape):
    h, w = img_shape
    MIN_CROP_WIDTH = .25 * w  # 25% of image width
    MIN_CROP_HEIGHT = .25 * h  # 25% of image height

    torso_points_xy = []
    detected_keypoint_count = 0
    
    for kpt_index in TORSO_KEYPOINTS:
        if confidences[kpt_index] > KEYPOINT_CONF_THRESH:
            torso_points_xy.append(keypoints[kpt_index])
            detected_keypoint_count += 1
    
    # Relax requirement: accept 2+ keypoints instead of all 4
    if detected_keypoint_count < MIN_KEYPOINTS_REQUIRED:
        print(f"  -> Insufficient keypoints ({detected_keypoint_count}). Falling back to segmentation.")
        return None  # Signal failure

    torso_points_xy = np.array(torso_points_xy)
    
    # Get the min/max x and y coordinates
    x1 = int(np.min(torso_points_xy[:, 0]))
    y1 = int(np.min(torso_points_xy[:, 1]))
    x2 = int(np.max(torso_points_xy[:, 0]))
    y2 = int(np.max(torso_points_xy[:, 1]))
    
    box_h, box_w = y2 - y1, x2 - x1
    
    # Avoid division by zero or tiny boxes
    if box_h <= 0 or box_w <= 0:
        return None
    
    # Smart padding: scale based on how many keypoints we have
    # More keypoints = confidence to use tighter padding; fewer keypoints = use more padding for safety
    if detected_keypoint_count >= 4:
        padding_factor = 0.20
    elif detected_keypoint_count == 3:
        padding_factor = 0.30
    else:  # 2 keypoints
        padding_factor = 0.40
    
    padding_y = int(box_h * padding_factor)
    padding_x = int(box_w * padding_factor)
    
    # Apply padding and clamp to image boundaries
    x1 = max(0, x1 - padding_x)
    y1 = max(0, y1 - padding_y)
    x2 = min(w, x2 + padding_x)
    y2 = min(h, y2 + padding_y)
    
    # Calculate the final dimensions after padding and clamping
    final_width = x2 - x1
    final_height = y2 - y1
    
    # Enforce minimum crop size to prevent losing sweater detail
    if final_height >= MIN_CROP_HEIGHT and final_width >= MIN_CROP_WIDTH:
        print(f"  -> Keypoint crop: {detected_keypoint_count} keypoints, size={final_width}x{final_height}")
        return (y1, y2, x1, x2)
    else:
        print(f"  -> Crop too small ({final_width}x{final_height}). Falling back to segmentation.")
        return None
    
#YOLO Segment fallback crop - improved to use full mask bounds with smart padding
def _get_mask_crop(best_mask_resized):
    """Extract crop from full segmentation mask with smart expansion to prevent too-small crops."""
    y_indices, x_indices = np.where(best_mask_resized > 0)
    
    if y_indices.size == 0:
        return None  # Empty mask
    
    # Get full bounding box from mask (improved from aggressive 10-80% vertical crop)
    x1, x2 = x_indices.min(), x_indices.max()
    y1, y2 = y_indices.min(), y_indices.max()
    
    box_h = y2 - y1
    box_w = x2 - x1
    
    if box_h <= 0 or box_w <= 0:
        return None
    
    # Apply smart padding: expand to ensure we capture full sweater context
    padding_y = int(box_h * 0.15)  # 15% vertical padding
    padding_x = int(box_w * 0.10)  # 10% horizontal padding
    
    img_h, img_w = best_mask_resized.shape
    
    # Apply padding and clamp to image boundaries
    crop_y1 = max(0, y1 - padding_y)
    crop_y2 = min(img_h, y2 + padding_y)
    crop_x1 = max(0, x1 - padding_x)
    crop_x2 = min(img_w, x2 + padding_x)
    
    if crop_y1 < crop_y2 and crop_x1 < crop_x2:
        print(f"  -> Segmentation crop: size={crop_x2-crop_x1}x{crop_y2-crop_y1}")
        return (crop_y1, crop_y2, crop_x1, crop_x2)
    
    return None

#Full function to extract and crop image
def extract_and_crop_image(image_path):
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"  -> ERROR: Could not load image {image_path}.")
        # Return a copy of a blank image
        blank_image = np.zeros((100, 100, 3), dtype=np.uint8)
        return blank_image, blank_image 

    cropped_bgr = image # Default fallback of full image
    img_height, img_width = image.shape[:2]

    if not _ensure_yolo_models():
        return cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)

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
                    print(f"  -> Smart crop (Pose) for {image_path}") 
            
            if crop_box is None:
                print(f"  -> Pose failed. Fallback crop (Segment) for {image_path}")
                crop_box = _get_mask_crop(best_mask_resized)
            
            if crop_box:
                y1, y2, x1, x2 = crop_box
                cropped_bgr = masked_image[y1:y2, x1:x2]
            else:
                # Fallback to the full masked image if no valid crop box
                cropped_bgr = masked_image
        
        # Check for excessive black pixels in the final cropped image
        # Again, ensure the result is not mostly black and we have a sweater to see
        FINAL_BLACK_THRESHOLD = 0.95 

        # Only run this check if not already using the original image
        if cropped_bgr is not image: 
            try:
                gray_cropped = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2GRAY)
                total_pixels = gray_cropped.size
                
                if total_pixels == 0:
                    print(f"  -> Result was an empty (0 pixel) image. Falling back to original.")
                    cropped_bgr = image
                else:
                    # Count pixels that are not black (value > 0)
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

        return cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
    
    except Exception as yolo_e:
        print(f"  -> YOLO/Masking failed for {image_path}: {yolo_e}. Falling back to full image.") 
        cropped_bgr = image
        return cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)