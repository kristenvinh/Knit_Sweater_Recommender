# --- Setup ---
import os
import pickle
import numpy as np
from dino_feature_extraction import extract_features


feature_dim = 768 # Feature dimension for DINO features
data_directory = 'EXAMPLE_PATH' # Example directory path, files should be in pattern subfolders

# --- Define filenames for the final, averaged features ---
master_features_file = 'master_features_DINO_yolo_pose.npy'
pattern_ids_file = 'pattern_ids_DINO_yolo_pose.pkl'
# ---

def build_master_vectors():
    master_vectors = {}
    # Ensure the root data directory exists
    if not os.path.isdir(data_directory):
        return None, None

    pattern_folders = [d for d in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory, d))]

    for pattern_id in enumerate(pattern_folders):
        pattern_folder_path = os.path.join(data_directory, pattern_id)
        pattern_feature_list = []
        
        image_files = [f for f in os.listdir(pattern_folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            continue

        for img_name in image_files:
            img_path = os.path.join(pattern_folder_path, img_name)
            _, feature_vector = extract_features(img_path)

            if feature_vector is not None and isinstance(feature_vector, np.ndarray):
                pattern_feature_list.append(feature_vector)

        # Average features for the pattern if there are any valid features
        if pattern_feature_list:
            master_vectors[pattern_id] = np.mean(pattern_feature_list, axis=0)

    if not master_vectors:
        return None, None

    #Convert to numpy array and save    
    pattern_ids = list(master_vectors.keys())
    feature_list = np.array([master_vectors[pid] for pid in pattern_ids]).astype('float32')

    np.save(master_features_file, feature_list)
    # Save pattern IDs to a pickle file
    with open(pattern_ids_file, 'wb') as f:
        pickle.dump(pattern_ids, f)

    return feature_list, pattern_ids