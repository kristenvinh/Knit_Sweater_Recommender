# Knit_Sweater_Recommender

This project was created by Kristen Vinh in 2025 as part of her Rochester Institute of Technology MS in Data Science capstone course to develop an algorithm that could recommend knitting sweater patterns from a user-uploaded photo.

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
- Explainable AI: GradCAM

## Installation

Clone the repository:
```
bash git clone https://github.com/KristenVinh/knit_sweater_recommender.git
cd sweater-recommender
```
Install dependencies: 
```
bash pip install -r requirements.txt
```
Environment Setup: Create a .env file in the root directory to store your Ravelry API credentials

```
RAVELRY_USERNAME=your_username
RAVELRY_PASSWORD=your_password
```

## Usage
1. Run the App:

```
python main.py
```
2. Open the index.html file in the "frontend" folder in your browser.

3. Upload a photo (sample photos avaiable in the 'example_photos' folder).

## Acknowledgments & References

- The indexing strategies were modeled after code used in [Fashion Recommender system](https://github.com/sonu275981/Fashion-Recommender-system/).
- Development: Front-end architecture and API integration were developed with the assistance of Google's Gemini AI.
- Data Source: Pattern data and images provided via the Ravelry API.s