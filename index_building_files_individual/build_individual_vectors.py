# --- Setup ---
# %%
import os
import pickle
import numpy as np
from annoy import AnnoyIndex
import time
from dotenv import load_dotenv
from slack_sdk import WebClient
from feature_extraction import extract_features

# --- Config ---
# You might want to rename these output files to reflect that they contain ALL images
master_features_file = 'all_image_features.npy' 
pattern_ids_file = 'all_image_pattern_ids.pkl'
image_paths_file = 'all_image_paths.pkl' # New: helpful to know WHICH image matched!

data_directory = '/Volumes/Extreme Pro/ANN_photos' 

def build_individual_vectors():
    """
    Walks through the directory and extracts a feature vector for EVERY image found.
    Does NOT average them.
    """
    print("Starting individual image vector extraction...")
    
    # Lists to hold data (lists remain ordered, so index 0 in features matches index 0 in IDs)
    all_features = []
    all_pattern_ids = []
    all_image_paths = [] # Optional: tracks the specific filename

    if not os.path.isdir(data_directory):
        print(f"Error: Data directory not found at '{data_directory}'")
        return None, None

    pattern_folders = [d for d in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory, d))]
    print(f"Found {len(pattern_folders)} pattern folders to process.")

    total_images_processed = 0

    for i, pattern_id in enumerate(pattern_folders):
        pattern_folder_path = os.path.join(data_directory, pattern_id)
        
        image_files = [f for f in os.listdir(pattern_folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Optional: Print progress every 100 patterns to avoid clutter
        if i % 50 == 0:
            print(f"Processing pattern {i}/{len(pattern_folders)}...")

        for img_name in image_files:
            img_path = os.path.join(pattern_folder_path, img_name)
            
            # Extract feature for this specific image
            _, feature_vector = extract_features(img_path)

            if feature_vector is not None and isinstance(feature_vector, np.ndarray):
                all_features.append(feature_vector)
                all_pattern_ids.append(pattern_id)
                all_image_paths.append(img_path) # Useful for debugging or displaying the result
                total_images_processed += 1

    if not all_features:
        print("No vectors were extracted.")
        return None, None
        
    # Convert feature list to numpy array for Annoy
    feature_matrix = np.array(all_features).astype('float32')

    print(f"\nExtraction complete.")
    print(f"Total Vectors: {len(feature_matrix)}")
    print(f"Shape: {feature_matrix.shape}")

    # Save everything
    print(f"Saving to {master_features_file}...")
    np.save(master_features_file, feature_matrix)
    
    with open(pattern_ids_file, 'wb') as f:
        pickle.dump(all_pattern_ids, f)
        
    with open(image_paths_file, 'wb') as f:
        pickle.dump(all_image_paths, f)

    return feature_matrix, all_pattern_ids