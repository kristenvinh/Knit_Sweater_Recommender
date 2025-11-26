# Knit_Sweater_Recommender

A knitwear recommendation system that suggests Ravelry knitting patterns based on user-uploaded images. This project utilizes machine learning, Approximate Nearest Neighbor (ANN) algorithms, and Explainable AI (XAI) to match real-world sweaters to knitting patterns.

## Overview
This application bridges the gap between seeing a sweater you like and finding a pattern to knit it. By analyzing nearly 10,000 popular patterns from Ravelry, the system builds a searchable vector index. When a user uploads a photo, the system processes it using object detection and feature extraction to find the most visually similar patterns available in the Ravelry database.

 ## Key Features
 
 ### Smart Pre-processing: 
 Uses YOLOv8 for person identification and background removal to isolate the sweater and improve accuracy.
 
 ### Vector Search: 
 Implements HNSWlib (Hierarchical Navigable Small World) for ultra-fast approximate nearest neighbor similarity search.
 
 ### Ravelry Integration: 
 
 Fetches real-time pattern details (photos, links, names) via the Ravelry API.
 
 ### Explainability: 
 
 Includes Grad-CAM (Gradient-weighted Class Activation Mapping) visualizations to show users exactly which features (texture, neckline, color) the model focused on.
 
 ### Interactive UI: A lightweight front-end built with FastAPI.
 
## Built With:

- Web Framework: FastAPI
- Machine Learning: PyTorch 
- Object Detection: YOLOv8 (Ultralytics)
- Feature Extraction: DinoV2 from MetaAI
- Vector Search: HNSWlib
- Explainability: GradCAM

## Installation

Clone the repository: bash git clone https://github.com/KristenVinh/knit_sweater_recommender.git

cd sweater-recommender

Install dependencies: Bash pip install -r requirements.txt

Environment Setup: Create a .env file in the root directory to store your Ravelry API credentials

RAVELRY_USERNAME=your_username
RAVELRY_PASSWORD=your_password

## Usage
1. Start the FastAPI server: Bashuvicorn main:app --reload
2. Open your browser and navigate to http://127.0.0.1:8000 to use the interface.

## Acknowledgments & References

- The indexing strategies were modeled after code used in [(https://github.com/sonu275981/Fashion-Recommender-system/)](https://github.com/sonu275981/Fashion-Recommender-system/).
- Development: Front-end architecture and API integration were developed with the assistance of Google's Gemini AI.
-Data Source: Pattern data and images provided via the Ravelry API.