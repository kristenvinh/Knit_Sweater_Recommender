# build_index_hnsw.py
import os
import pickle
import numpy as np
import hnswlib
import time
from build_master_vectors import build_master_vectors

# --- Setup ---
feature_dim = 2048 #Feature dimension for ResNet features
data_directory = '/Volumes/Extreme Pro/ANN_photos'  # Example directory path, images should be in pattern subfolders

# --- File Names (Multi-centroid format) ---
master_features_file = 'master_features_resnet_yolo_pose_multicentroid.npy'
pattern_ids_file = 'pattern_ids_resnet_yolo_pose_multicentroid.pkl'
pattern_mapping_file = 'pattern_to_centroids_resnet_yolo_pose.pkl'  # Maps centroid_idx -> pattern_id
index_name = 'sweater_hnsw_resnet_yolo_pose_multicentroid.bin'
mapping_file = 'centroid_to_pattern_resnet_yolo_pose.pkl'  # For query-time lookup
# ---


if __name__ == "__main__":
    script_start_time = time.perf_counter()
    try:
        # Load vectors if already built, otherwise build vectors
        # Multi-centroid: returns (all_centroids, pattern_ids, pattern_to_centroid_indices)
        if (os.path.exists(master_features_file) and 
            os.path.exists(pattern_ids_file) and 
            os.path.exists(pattern_mapping_file)):
            feature_list = np.load(master_features_file)
            with open(pattern_ids_file, 'rb') as f:
                pattern_ids = pickle.load(f)
            with open(pattern_mapping_file, 'rb') as f:
                pattern_to_centroid_indices = pickle.load(f)
        else:
            feature_list, pattern_ids, pattern_to_centroid_indices = build_master_vectors()

        if feature_list is None or not pattern_ids:
             raise ValueError("Failed to load or build feature vectors.")

        num_centroids = len(feature_list)
        print(f"\nBuilding HNSW index with {num_centroids} centroids for {len(pattern_ids)} patterns...")

        # Create centroid-to-pattern mapping for query-time lookup
        centroid_to_pattern = {}
        for pattern_id, centroid_indices in pattern_to_centroid_indices.items():
            for centroid_idx in centroid_indices:
                centroid_to_pattern[centroid_idx] = pattern_id

        # Build Index
        index = hnswlib.Index(space='cosine', dim=feature_dim)
        
        # Initialize the index with multi-centroid data
        index.init_index(max_elements=num_centroids, ef_construction=200, M=16)
        
        # Start timer
        build_start_time = time.perf_counter()
        
        # Add all centroids with their indices (0, 1, 2, ...)
        index.add_items(feature_list, np.arange(num_centroids))
        
        build_duration = time.perf_counter() - build_start_time
        
        print(f"HNSWlib index built in {build_duration:.2f} seconds.")

        # Save the Index
        index.save_index(index_name)
        
        # Save the centroid-to-pattern mapping for use in inference
        with open(mapping_file, 'wb') as f:
            pickle.dump(centroid_to_pattern, f)
        
        print(f"Index saved to: {index_name}")
        print(f"Centroid-to-pattern mapping saved to: {mapping_file}")

    except Exception as e:
        print(f"A critical error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        script_duration = time.perf_counter() - script_start_time
        print(f"\nScript finished in {script_duration:.2f} seconds.")