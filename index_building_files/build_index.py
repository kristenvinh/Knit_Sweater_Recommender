# build_index_hnsw.py
import os
import pickle
import numpy as np
import hnswlib
import time
from dotenv import load_dotenv
from slack_sdk import WebClient
from build_master_vectors import build_master_vectors

# --- Setup ---
load_dotenv()
# Initialize Slack client if you have a token set up
slack_token = os.environ.get("SLACK_BOT_TOKEN")
client = WebClient(token=slack_token) if slack_token else None

feature_dim = 768
data_directory = '/Volumes/Extreme Pro/ANN_photos' 

# --- File Names ---
master_features_file = 'master_features_DINO_yolo_pose.npy'
pattern_ids_file = 'pattern_ids_DINO_yolo_pose.pkl'
# ---


if __name__ == "__main__":
    script_start_time = time.perf_counter()
    try:
        # Build Vectors
        if os.path.exists(master_features_file) and os.path.exists(pattern_ids_file):
            print("Loading existing master features and pattern IDs...")
            feature_list = np.load(master_features_file)
            with open(pattern_ids_file, 'rb') as f:
                pattern_ids = pickle.load(f)
        else:
            feature_list, pattern_ids = build_master_vectors()

        if feature_list is None or not pattern_ids:
             raise ValueError("Failed to load or build feature vectors. Exiting.")

        num_elements = len(feature_list)
        print(f"\nLoaded {num_elements} feature vectors.")
        print(f"Feature vector dimension: {feature_list.shape[1]}")

        # Build Index
        print("\n--- Building HNSWlib Index ---")
        
        index = hnswlib.Index(space='cosine', dim=feature_dim)
        
        # Initialize the index
        index.init_index(max_elements=num_elements, ef_construction=200, M=16)
        
        print(f"Adding {num_elements} vectors to the index...")
        build_start_time = time.perf_counter()
        
        # Add the vectors and their corresponding integer IDs
        index.add_items(feature_list, np.arange(num_elements))
        
        build_duration = time.perf_counter() - build_start_time
        
        print(f"HNSWlib index built in {build_duration:.2f} seconds.")

        # --- Step 3: Save the Index ---
        index_name = 'sweater_hnsw_DINO_yolo_pose.bin'
        index.save_index(index_name)
        print(f"Index saved to {index_name}")

        message = f"✅ HNSWlib DINO YOLO pose built in {build_duration:.2f} seconds and saved as {index_name}."
        if client:
             client.chat_postMessage(channel="python_updates", text=message, username="Bot User")

    except Exception as e:
        print(f"A critical error occurred: {e}")
        error_summary = f"🔥 Critical Failure in HNSWlib Script: `{e}`"
        if client:
            client.chat_postMessage(channel="python_updates", text=error_summary, username="Bot User")
    finally:
        script_duration = time.perf_counter() - script_start_time
        print(f"\nScript finished in {script_duration:.2f} seconds.")
        message = f"\nScript DINO finished in {script_duration:.2f} seconds."
        if client:
             client.chat_postMessage(channel="python_updates", text=message, username="Bot User")
