# --- Setup ---
# %%
import os
import pickle
import numpy as np
from annoy import AnnoyIndex
import time
from dotenv import load_dotenv
from slack_sdk import WebClient
from dino_feature_extraction import extract_features

load_dotenv()
# Initialize Slack client if you have a token set up
slack_token = os.environ.get("SLACK_BOT_TOKEN")
client = WebClient(token=slack_token) if slack_token else None

feature_dim = 768
data_directory = '/Volumes/Extreme Pro/ANN_photos' # IMPORTANT: Update this path to your data directory

# --- Define filenames for the final, averaged features ---
# master_features_file = 'master_features.npy'
# pattern_ids_file = 'pattern_ids.pkl'

master_features_file = 'master_features_DINO_yolo_pose.npy'
pattern_ids_file = 'pattern_ids_DINO_yolo_pose.pkl'
# ---

def build_master_vectors():
    """
    Walks through the organized image directory, extracts features for all images
    of a pattern, and computes an average "master" vector for each pattern.
    """
    print("No existing master features found. Starting vector averaging process...")
    master_vectors = {}

    # Ensure the root data directory exists
    if not os.path.isdir(data_directory):
        print(f"Error: Data directory not found at '{data_directory}'")
        return None, None

    pattern_folders = [d for d in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory, d))]
    print(f"Found {len(pattern_folders)} pattern folders to process.")

    for i, pattern_id in enumerate(pattern_folders):
        pattern_folder_path = os.path.join(data_directory, pattern_id)
        pattern_feature_list = []
        
        image_files = [f for f in os.listdir(pattern_folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            continue

        print(f"Processing pattern {i+1}/{len(pattern_folders)}: {pattern_id}")
        for img_name in image_files:
            img_path = os.path.join(pattern_folder_path, img_name)
            _, feature_vector = extract_features(img_path)

            if feature_vector is not None and isinstance(feature_vector, np.ndarray):
                pattern_feature_list.append(feature_vector)

        if pattern_feature_list:
            master_vectors[pattern_id] = np.mean(pattern_feature_list, axis=0)

    if not master_vectors:
        print("No vectors were extracted. Please check the image paths and content.")
        return None, None
        
    pattern_ids = list(master_vectors.keys())
    feature_list = np.array([master_vectors[pid] for pid in pattern_ids]).astype('float32')

    print(f"\nSaving {len(pattern_ids)} master vectors to {master_features_file}")
    np.save(master_features_file, feature_list)
    with open(pattern_ids_file, 'wb') as f:
        pickle.dump(pattern_ids, f)

    return feature_list, pattern_ids