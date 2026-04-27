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
 
 ### Interactive UI: 
 A lightweight front-end built with FastAPI.
 
## Built With:

- Web Framework: FastAPI
- Machine Learning: PyTorch 
- Object Detection: YOLOv8 (Ultralytics)
- Feature Extraction: DinoV2 (MetaAI)
- Vector Search: HNSWlib
- Explainable AI: GradCAM

## Installation

Clone the repository:
```
bash git clone https://github.com/KristenVinh/knit_sweater_recommender.git
cd knit_sweater_recommender
```
Install dependencies: 
```
bash pip install -r requirements.txt
```
Environment Setup: Create a .env file in the root directory to store your Ravelry API credentials (easily obtained in the "PRO" section of a Ravelry user's account). Further details about the Ravelry API can be found at [https://www.ravelry.com/api](https://www.ravelry.com/api). Only "read-only" access is needed for this app.

```
RAVELRY_ACCESS_KEY=your_ravelry_access_key
RAVELRY_PERSONAL_KEY=your_ravelry_personal_key

# Optional: API hardening controls
REQUIRE_API_KEY=false
RATE_LIMIT_WINDOW_SEC=60
RATE_LIMIT_MAX_REQUESTS=120

# Optional: CORS controls (defaults are localhost-friendly)
CORS_ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000,http://localhost:8000,http://127.0.0.1:8000
CORS_ALLOWED_ORIGIN_REGEX=^https?://(localhost|127\\.0\\.0\\.1)(:\\d+)?$|^null$
```

You can copy `.env.example` to `.env` and fill in your values.

## Usage
1. Run the App:

```
python main.py
```
2. Open the index.html file in your browser.

3. Upload a photo (sample photos avaiable in the 'example_photos' folder).

## Benchmarking

Use the reusable benchmark script to run a timed batch over all images in `example_photos` and print median/P95 metrics from API timing fields.

```bash
bash bench/run_benchmark.sh
```

Optional environment variables:

```bash
API_URL=http://127.0.0.1:8000/recommend
IMAGE_DIR=example_photos
OUTPUT_FILE=bench/phase1_run.jsonl
WARMUP_COUNT=3
SLEEP_BETWEEN_SEC=0.2
BENCH_API_KEY=your_key_if_required
```

You can also summarize any prior run:

```bash
python3 bench/summarize.py bench/phase1_run.jsonl
```

### Phase 1 Performance Snapshot

Measured from benchmark run `bench/benchmark_20260427_113006.jsonl` on 7 sample images.

- Total requests: 7
- Successful (200): 7
- Errors: 0
- total_request_sec: median=0.9394s, p95=1.4186s
- feature_extraction_sec: median=0.2758s, p95=0.4441s
- index_query_sec: median=0.0009s, p95=0.0022s
- ravelry_fetch_sec: median=0.2834s, p95=0.5016s
- xai_sec: median=0.3378s, p95=0.6388s

### Suggested Environment Profiles

- Local development profile:
REQUIRE_API_KEY=false, RATE_LIMIT_WINDOW_SEC=60, RATE_LIMIT_MAX_REQUESTS=120
- Production profile:
REQUIRE_API_KEY=true, RATE_LIMIT_WINDOW_SEC=60, RATE_LIMIT_MAX_REQUESTS=30

## Acknowledgments & References

- The indexing strategies were modeled after code used in [Fashion Recommender system](https://github.com/sonu275981/Fashion-Recommender-system/), the code from the paper [Personalized fashion recommender system with image based neural networks](https://iopscience.iop.org/article/10.1088/1757-899X/981/2/022073), although the code is not currently accessible on Github as of 12.8.25.
- Development: Front-end architecture, XAI scripts, and the final API integration were developed with Google's Gemini AI. Gemini AI also helped in developing and debugging other sections of code, including feature extraction, cropping and index building scripts. 
- Data Source: Pattern data and images provided via the Ravelry API.