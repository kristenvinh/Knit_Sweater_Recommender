# build_index_hnsw.py
import os
import pickle
import numpy as np
import hnswlib
import time
from build_master_vectors import build_master_vectors

# --- Setup ---
feature_dim = 768 #Feature dimension for DINO features
data_directory = 'data/directory/here'  # Example directory path, images should be in pattern subfolders

# --- File Names ---
master_features_file = 'master_features_DINO_yolo_pose.npy'
pattern_ids_file = 'pattern_ids_DINO_yolo_pose.pkl'
# ---


if __name__ == "__main__":
    script_start_time = time.perf_counter()
    try:
        # Build Vectors
        if os.path.exists(master_features_file) and os.path.exists(pattern_ids_file):
            feature_list = np.load(master_features_file)
            with open(pattern_ids_file, 'rb') as f:
                pattern_ids = pickle.load(f)
        else:
            feature_list, pattern_ids = build_master_vectors()

        if feature_list is None or not pattern_ids:
             raise ValueError("Failed to load or build feature vectors.")

        num_elements = len(feature_list)

        # Build Index

        index = hnswlib.Index(space='cosine', dim=feature_dim)
        
        # Initialize the index
        index.init_index(max_elements=num_elements, ef_construction=200, M=16)
        
        build_start_time = time.perf_counter()
        
        # Add the vectors and their corresponding integer IDs
        index.add_items(feature_list, np.arange(num_elements))
        
        build_duration = time.perf_counter() - build_start_time
        
        print(f"HNSWlib index built in {build_duration:.2f} seconds.")

        # Save the Index
        index_name = 'sweater_hnsw_DINO_yolo_pose.bin'
        index.save_index(index_name)

    except Exception as e:
        print(f"A critical error occurred: {e}")
        error_summary = f"Failure in HNSWlib Script: `{e}`"
    finally:
        script_duration = time.perf_counter() - script_start_time
        print(f"\nScript finished in {script_duration:.2f} seconds.")
        