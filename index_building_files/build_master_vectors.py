# --- Setup ---
import os
import sys
import pickle
import numpy as np
from sklearn.cluster import KMeans

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from dino_feature_extraction import extract_features


feature_dim = 768 # Feature dimension for DINO features
data_directory = '/Volumes/Extreme Pro/ANN_photos'  # Example directory path, files should be in pattern subfolders
n_clusters = 3   # K-means clusters per pattern (3-5 recommended, 4 is good balance)

# --- Define filenames for the multi-centroid features ---
# New format: stores all centroids for all patterns in a flat array
# Mapping: pattern_to_centroid_indices maps each pattern to its centroids in the flat array
master_features_file = 'master_features_DINO_yolo_pose_multicentroid3.npy'
pattern_ids_file = 'pattern_ids_DINO_yolo_pose_multicentroid3.pkl'
pattern_mapping_file = 'pattern_to_centroids_DINO_yolo_pose3.pkl'  # Maps pattern_id -> [centroid_indices]
# For backward compatibility, keep the old filename pattern available
legacy_features_file = 'master_features_DINO_yolo_pose.npy'
legacy_ids_file = 'pattern_ids_DINO_yolo_pose.pkl'
# ---

def build_master_vectors():
    """
    Build multi-centroid feature vectors for each pattern using K-means clustering.
    
    Returns:
        - all_centroids: (N*n_clusters, 768) array of all centroids flattened
        - pattern_ids: List of unique pattern IDs in order
        - pattern_to_centroid_indices: Dict mapping pattern_id -> [centroid_indices]
    """
    pattern_to_centroid_indices = {}
    all_centroids = []
    pattern_ids = []
    
    if not os.path.isdir(data_directory):
        print(f"Error: Data directory '{data_directory}' not found")
        return None, None, None

    pattern_folders = sorted([d for d in os.listdir(data_directory) 
                             if os.path.isdir(os.path.join(data_directory, d))])

    for pattern_id in pattern_folders:
        pattern_folder_path = os.path.join(data_directory, pattern_id)
        pattern_feature_list = []
        
        image_files = [f for f in os.listdir(pattern_folder_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            print(f"  Skipping {pattern_id}: no images found")
            continue

        # Extract features for all images in this pattern
        for img_name in image_files:
            img_path = os.path.join(pattern_folder_path, img_name)
            try:
                _, feature_vector = extract_features(img_path)

                if feature_vector is not None and isinstance(feature_vector, np.ndarray):
                    pattern_feature_list.append(feature_vector)
            except Exception as e:
                print(f"  Warning: Failed to extract features from {img_name}: {e}")
                continue

        if not pattern_feature_list:
            print(f"  Skipping {pattern_id}: no valid features extracted")
            continue

        pattern_feature_array = np.array(pattern_feature_list)
        
        # Apply K-means clustering on this pattern's features
        n_samples = len(pattern_feature_array)
        n_centroids = min(n_clusters, n_samples)  # Can't have more clusters than samples
        
        if n_centroids < 2:
            # If only 1 sample, use it directly (single centroid)
            kmeans = KMeans(n_clusters=1, random_state=42, n_init=10)
            kmeans.fit(pattern_feature_array)
            centroids = kmeans.cluster_centers_
        else:
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=n_centroids, random_state=42, n_init=10)
            kmeans.fit(pattern_feature_array)
            centroids = kmeans.cluster_centers_
        
        # Store this pattern's centroids
        start_idx = len(all_centroids)
        for centroid in centroids:
            all_centroids.append(centroid)
        
        # Map pattern to its centroid indices
        centroid_indices = list(range(start_idx, len(all_centroids)))
        pattern_to_centroid_indices[pattern_id] = centroid_indices
        pattern_ids.append(pattern_id)
        
        print(f"  ✓ {pattern_id}: {n_samples} images -> {len(centroids)} centroids")

    if not all_centroids:
        print("Error: No valid patterns with features")
        return None, None, None

    # Convert to numpy array
    all_centroids_array = np.array(all_centroids, dtype='float32')
    
    # Save all centroids
    np.save(master_features_file, all_centroids_array)
    
    # Save pattern IDs
    with open(pattern_ids_file, 'wb') as f:
        pickle.dump(pattern_ids, f)
    
    # Save pattern-to-centroid mapping
    with open(pattern_mapping_file, 'wb') as f:
        pickle.dump(pattern_to_centroid_indices, f)
    
    print(f"\nSaved multi-centroid index:")
    print(f"  Total patterns: {len(pattern_ids)}")
    print(f"  Total centroids: {len(all_centroids)}")
    print(f"  Average centroids per pattern: {len(all_centroids) / len(pattern_ids):.2f}")

    return all_centroids_array, pattern_ids, pattern_to_centroid_indices