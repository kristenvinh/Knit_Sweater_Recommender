#!/usr/bin/env python3
"""
Test script for Phase 2, Step 5: Crop Robustness Improvements
Measures the effectiveness of relaxed keypoint requirements and improved segmentation fallback
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np

# Add root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from YOLO_pose_crop import extract_and_crop_image, _ensure_yolo_models

def test_crop_logic(image_dir="example_photos"):
    """Test crop robustness on all images in directory."""
    
    print("=" * 70)
    print("CROP ROBUSTNESS TEST - Phase 2, Step 5")
    print("=" * 70)
    print(f"Test directory: {image_dir}")
    print(f"Test started: {datetime.now().isoformat()}")
    print()
    
    # Ensure YOLO models are loaded
    print("[1/3] Loading YOLO models...")
    if not _ensure_yolo_models():
        print("ERROR: Failed to initialize YOLO models")
        return None
    print("✓ YOLO models loaded")
    print()
    
    # Collect test images
    print("[2/3] Scanning test images...")
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    image_files = [
        f for f in os.listdir(image_dir)
        if os.path.isfile(os.path.join(image_dir, f)) and 
           Path(f).suffix.lower() in image_extensions
    ]
    
    if not image_files:
        print(f"ERROR: No images found in {image_dir}")
        return None
    
    print(f"✓ Found {len(image_files)} images to test")
    print(f"  Images: {', '.join(image_files)}")
    print()
    
    # Test each image
    print("[3/3] Testing crop logic...")
    print("-" * 70)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "test_directory": image_dir,
        "total_images": len(image_files),
        "images": {},
        "summary": {}
    }
    
    crop_methods = {
        "keypoint_4": 0,
        "keypoint_3": 0,
        "keypoint_2": 0,
        "segmentation": 0,
        "error": 0
    }
    
    for idx, image_file in enumerate(image_files, 1):
        image_path = os.path.join(image_dir, image_file)
        print(f"\n[{idx}/{len(image_files)}] {image_file}")
        
        try:
            # Read original image to get dimensions
            original = cv2.imread(image_path)
            if original is None:
                print(f"  ✗ ERROR: Could not load image")
                crop_methods["error"] += 1
                results["images"][image_file] = {
                    "status": "error",
                    "error": "Could not load image"
                }
                continue
            
            orig_h, orig_w = original.shape[:2]
            print(f"  Original dimensions: {orig_w}x{orig_h}")
            
            # Apply crop logic
            cropped_rgb = extract_and_crop_image(image_path)
            
            if cropped_rgb is None:
                print(f"  ✗ ERROR: Crop returned None")
                crop_methods["error"] += 1
                results["images"][image_file] = {
                    "status": "error",
                    "error": "Crop returned None"
                }
                continue
            
            cropped_h, cropped_w = cropped_rgb.shape[:2]
            crop_ratio = (cropped_w * cropped_h) / (orig_w * orig_h)
            
            print(f"  Cropped dimensions: {cropped_w}x{cropped_h}")
            print(f"  Crop ratio: {crop_ratio:.2%}")
            
            # Analyze pixel distribution to infer crop method
            # Segmentation crops tend to be larger with more varied pixel distribution
            # Keypoint crops tend to be tighter
            
            # Count non-black pixels to assess quality
            if len(cropped_rgb.shape) == 3:
                gray = cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2GRAY)
            else:
                gray = cropped_rgb
            
            total_pixels = gray.size
            non_black = np.count_nonzero(gray)
            non_black_ratio = non_black / total_pixels if total_pixels > 0 else 0
            
            print(f"  Non-black pixels: {non_black_ratio:.1%}")
            
            # Determine crop method based on output
            # This is heuristic since we capture keypoint count in logs
            if crop_ratio < 0.30:
                method = "keypoint_tight"
                keypoint_estimate = 4
            elif crop_ratio < 0.50:
                method = "keypoint_moderate"
                keypoint_estimate = 3
            elif crop_ratio < 0.65:
                method = "keypoint_loose"
                keypoint_estimate = 2
            else:
                method = "segmentation"
                keypoint_estimate = 0
            
            crop_methods[f"keypoint_{keypoint_estimate}" if keypoint_estimate > 0 else "segmentation"] += 1
            
            print(f"  ✓ Status: SUCCESS ({method})")
            
            results["images"][image_file] = {
                "status": "success",
                "original_size": {"width": orig_w, "height": orig_h},
                "cropped_size": {"width": cropped_w, "height": cropped_h},
                "crop_ratio": crop_ratio,
                "non_black_ratio": non_black_ratio,
                "inferred_method": method,
                "estimated_keypoints": keypoint_estimate
            }
            
        except Exception as e:
            print(f"  ✗ EXCEPTION: {str(e)}")
            crop_methods["error"] += 1
            results["images"][image_file] = {
                "status": "error",
                "error": str(e)
            }
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    successful = len([img for img in results["images"].values() if img["status"] == "success"])
    failed = len(results["images"]) - successful
    
    print(f"\nSuccess Rate: {successful}/{len(image_files)} ({successful/len(image_files)*100:.1f}%)")
    print(f"Failed: {failed}")
    print()
    
    print("Crop Method Distribution:")
    for method in ["keypoint_4", "keypoint_3", "keypoint_2", "segmentation", "error"]:
        count = crop_methods[method]
        pct = (count / len(image_files) * 100) if len(image_files) > 0 else 0
        print(f"  {method:20s}: {count:2d} ({pct:5.1f}%)")
    
    # Calculate keypoint coverage
    keypoint_total = crop_methods["keypoint_4"] + crop_methods["keypoint_3"] + crop_methods["keypoint_2"]
    print(f"\nKeypoint-based crops: {keypoint_total}/{successful} ({keypoint_total/successful*100:.1f}% of successes)" if successful > 0 else "Keypoint-based crops: 0/0")
    print(f"Segmentation fallback: {crop_methods['segmentation']}/{successful} ({crop_methods['segmentation']/successful*100:.1f}% of successes)" if successful > 0 else "Segmentation fallback: 0/0")
    
    # Size statistics
    if successful > 0:
        crop_ratios = [img["crop_ratio"] for img in results["images"].values() if img["status"] == "success"]
        print(f"\nCrop Size Statistics:")
        print(f"  Min ratio:  {min(crop_ratios):.1%}")
        print(f"  Max ratio:  {max(crop_ratios):.1%}")
        print(f"  Avg ratio:  {np.mean(crop_ratios):.1%}")
        print(f"  Median ratio: {np.median(crop_ratios):.1%}")
    
    results["summary"] = {
        "success_count": successful,
        "total_count": len(image_files),
        "success_rate": successful / len(image_files),
        "crop_methods": crop_methods,
        "keypoint_crops_pct": keypoint_total / successful if successful > 0 else 0,
        "segmentation_crops_pct": crop_methods["segmentation"] / successful if successful > 0 else 0
    }
    
    print("\n" + "=" * 70)
    print(f"Test completed: {datetime.now().isoformat()}")
    print("=" * 70)
    
    # Save results to JSON
    results_file = "crop_test_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_file}")
    
    return results


if __name__ == "__main__":
    results = test_crop_logic()
    sys.exit(0 if results and results["summary"]["success_count"] > 0 else 1)
