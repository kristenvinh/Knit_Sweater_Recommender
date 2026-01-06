# build_index_hnsw.py
import os
import pickle
import numpy as np
import hnswlib
import time
from dotenv import load_dotenv
from slack_sdk import WebClient
from build_individual_vectors import build_individual_vectors

# --- Setup ---
load_dotenv()
# Initialize Slack client if you have a token set up
slack_token = os.environ.get("SLACK_BOT_TOKEN")
client = WebClient(token=slack_token) if slack_token else None

feature_dim = 1024
data_directory = '/Volumes/Extreme Pro/ANN_photos'  # IMPORTANT: Update this path to your data directory

# --- Define filenames for the final, averaged features ---
# master_features_file = 'master_features.npy'
# pattern_ids_file = 'pattern_ids.pkl'

master_features_file = 'master_features_yolo_seg.npy'
pattern_ids_file = 'pattern_ids_yolo_seg.pkl'
# ---


if __name__ == "__main__":
    script_start_time = time.perf_counter()
    try:
        # --- Step 1: Load or Build Feature Vectors ---
        # We check if the files exist; if not, we build them.
        if os.path.exists(master_features_file) and os.path.exists(pattern_ids_file):
            print(f"Loading existing features from {master_features_file}...")
            feature_list = np.load(master_features_file)
            with open(pattern_ids_file, 'rb') as f:
                pattern_ids = pickle.load(f)
        else:
            print("Feature files not found. building from scratch...")
            feature_list, pattern_ids = build_individual_vectors()

        if feature_list is None or not pattern_ids:
             raise ValueError("Failed to load or build feature vectors. Exiting.")

        # --- CRITICAL FIX: Dynamic Dimension ---
        # Instead of hardcoding 1024, we read the shape from the data itself.
        # ResNet50 is usually 2048.
        num_elements = feature_list.shape[0]
        feature_dim = feature_list.shape[1] 
        
        print(f"\nLoaded {num_elements} feature vectors.")
        print(f"Detected Feature Dimension: {feature_dim}")

        # --- Step 2: Build the HNSWlib Index ---
        print("\n--- Building HNSWlib Index ---")
        
        index = hnswlib.Index(space='cosine', dim=feature_dim)
        
        # Init Index
        # Note: 'max_elements' must be >= num_elements. 
        # Since we are indexing individual images, this number will be much higher now.
        index.init_index(max_elements=num_elements, ef_construction=200, M=16)
        
        print(f"Adding {num_elements} vectors to the index...")
        build_start_time = time.perf_counter()
        
        # Add the vectors
        index.add_items(feature_list, np.arange(num_elements))
        
        build_duration = time.perf_counter() - build_start_time
        print(f"HNSWlib index built in {build_duration:.2f} seconds.")

        # --- Step 3: Save the Index ---
        index_name = 'sweater_hnsw_ResNet50_Individual.bin'
        index.save_index(index_name)
        print(f"Index saved to {index_name}")

        message = f"✅ HNSWlib Index (Individual Images) built. {num_elements} items. Dim: {feature_dim}."
        if client:
             client.chat_postMessage(channel="python_updates", text=message, username="Bot User")

    except Exception as e:
        print(f"A critical error occurred: {e}")
        if client:
            client.chat_postMessage(channel="python_updates", text=f"🔥 Failure: {e}", username="Bot User")
    finally:
        script_duration = time.perf_counter() - script_start_time
        print(f"\nScript finished in {script_duration:.2f} seconds.")