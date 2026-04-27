#This script runs the main FastAPI server for sweater recommendations, written with 
# Gemini AI assistance.
import os
import pickle
import shutil
import asyncio
import time
import uuid
import hnswlib
import httpx
import uvicorn
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import base64
from dino_feature_extraction import extract_features, FEATURE_DIM, get_dino_components
from xaiutil import generate_xai_heatmap_bytes


# --- Configuration & Constants ---
INDEX_FILE = 'sweater_hnsw_DINO_yolo_pose.bin'
PATTERN_IDS_FILE = 'pattern_ids_DINO_yolo_pose.pkl'

# --- Global State (Loaded on Startup) ---
app_state = {
    "hnsw_index": None,
    "pattern_id_list": None,
    "ravelry_client": None,
    "ravelry_cache": {},
    "rate_limit_buckets": {},
}

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MIN_IMAGE_DIM = 50
RAVELRY_CACHE_TTL_SEC = 3600
RAVELRY_MAX_RETRIES = 3
DEFAULT_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]


def _parse_csv_env(var_name: str, default_values: list[str]) -> list[str]:
    raw_value = os.environ.get(var_name, "")
    values = [value.strip() for value in raw_value.split(",") if value.strip()]
    return values or default_values


ALLOWED_ORIGINS = _parse_csv_env("CORS_ALLOWED_ORIGINS", DEFAULT_ALLOWED_ORIGINS)
DEFAULT_CORS_ORIGIN_REGEX = r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$|^null$"
CORS_ORIGIN_REGEX = os.environ.get("CORS_ALLOWED_ORIGIN_REGEX", DEFAULT_CORS_ORIGIN_REGEX)
RATE_LIMIT_WINDOW_SEC = int(os.environ.get("RATE_LIMIT_WINDOW_SEC", "60"))
RATE_LIMIT_MAX_REQUESTS = int(os.environ.get("RATE_LIMIT_MAX_REQUESTS", "30"))


def _get_api_key() -> str | None:
    """Resolve API key from Ravelry credentials only."""
    return os.environ.get("RAVELRY_PERSONAL_KEY") or os.environ.get("RAVELRY_ACCESS_KEY")


def _is_api_key_enforcement_enabled() -> bool:
    raw_value = (os.environ.get("REQUIRE_API_KEY") or "false").strip().lower()
    return raw_value in {"1", "true", "yes", "on"}

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Sweater Recommender API",
    description="Upload a sweater image to find similar patterns.",
)

# --- Add CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=CORS_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key"],
)


def _require_api_key(x_api_key: str | None) -> None:
    """Reject requests unless the configured API key is supplied."""
    if not _is_api_key_enforcement_enabled():
        return

    api_key = _get_api_key()
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="API key enforcement is enabled but no key is configured.",
        )

    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing API key.")

    if x_api_key != api_key:
        raise HTTPException(status_code=403, detail="Invalid API key.")


def _get_client_ip(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        # Use the left-most address as the original client.
        return forwarded_for.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _enforce_rate_limit(request: Request) -> None:
    client_ip = _get_client_ip(request)
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW_SEC

    bucket = app_state["rate_limit_buckets"].get(client_ip, [])
    bucket = [timestamp for timestamp in bucket if timestamp >= window_start]

    if len(bucket) >= RATE_LIMIT_MAX_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail=(
                f"Rate limit exceeded. Max {RATE_LIMIT_MAX_REQUESTS} requests per "
                f"{RATE_LIMIT_WINDOW_SEC} seconds."
            ),
        )

    bucket.append(now)
    app_state["rate_limit_buckets"][client_ip] = bucket

# --- Startup Event Handler ---
@app.on_event("startup")
async def load_models_on_startup():
    """
    This function runs ONCE when the server starts.
    It loads all our heavy assets into memory.
    """
    print("--- Server starting up... ---")
    print(f"CORS allowlist: {ALLOWED_ORIGINS}")
    print(f"CORS allow-origin regex: {CORS_ORIGIN_REGEX}")

    # 1. DINOv2/YOLO are now loaded lazily on first request.
    print(f"✅ Model loading configured as lazy. Expected feature dim: {FEATURE_DIM}")

    # 2. Load HNSWlib Index
    if not os.path.exists(INDEX_FILE):
        raise FileNotFoundError(f"Index file not found: {INDEX_FILE}. Did you run build_index.py?")
    
    print(f"Loading HNSW index from {INDEX_FILE}...")
    index = hnswlib.Index(space='cosine', dim=FEATURE_DIM)
    index.load_index(INDEX_FILE)
    index.set_ef(50)  # Set search-time efficiency (higher is more accurate but slower)
    app_state["hnsw_index"] = index
    print("✅ HNSW index loaded.")

    # 3. Load Pattern ID Map
    if not os.path.exists(PATTERN_IDS_FILE):
        raise FileNotFoundError(f"Pattern ID file not found: {PATTERN_IDS_FILE}.")
    
    print(f"Loading Pattern ID map from {PATTERN_IDS_FILE}...")
    with open(PATTERN_IDS_FILE, 'rb') as f:
        app_state["pattern_id_list"] = pickle.load(f)
    print("✅ Pattern ID map loaded.")


    # 4. Initialize Ravelry Client (replaces PRAW)
    print("Loading .env file and initializing Ravelry client...")
    load_dotenv()
    try:
        # --- CHANGED: Swapped PRAW for httpx with Basic Auth ---
        ravelry_user = os.environ.get("RAVELRY_ACCESS_KEY")
        ravelry_pass = os.environ.get("RAVELRY_PERSONAL_KEY")
        RAVELRY_API_URL = "https://api.ravelry.com"
        
        if not ravelry_user or not ravelry_pass:
            raise ValueError("RAVELRY_ACCESS_KEY or RAVELRY_PERSONAL_KEY not found in .env file.")
        # Create a persistent, async HTTP client with Basic Auth
        auth = httpx.BasicAuth(ravelry_user, ravelry_pass)
        client = httpx.AsyncClient(auth=auth, base_url=RAVELRY_API_URL)
        
        app_state["ravelry_client"] = client
        print("✅ Ravelry client initialized.")
    except Exception as e:
        print(f"🔥 FAILED to initialize Ravelry client: {e}")
        print("Please check your .env file for RAVELRY_USERNAME and RAVELRY_PASSWORD.")

    if _is_api_key_enforcement_enabled():
        if not _get_api_key():
            print(
                "🔥 REQUIRE_API_KEY is enabled but no key found in RAVELRY_PERSONAL_KEY or RAVELRY_ACCESS_KEY. "
                "Authenticated requests will fail until keys are configured."
            )
        else:
            print("✅ API key enforcement enabled for /recommend.")
    else:
        print("ℹ️  API key enforcement is disabled (set REQUIRE_API_KEY=true to enable).")
        
    print("--- Server startup complete. Ready for requests. ---")


@app.on_event("shutdown")
async def close_clients_on_shutdown():
    client = app_state.get("ravelry_client")
    if client is not None:
        await client.aclose()
        app_state["ravelry_client"] = None
        print("✅ Ravelry client closed.")

# --- Helper functions ---
def _validate_upload(file: UploadFile):
    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(400, "Only image uploads are supported.")

    current_pos = file.file.tell()
    file.file.seek(0, os.SEEK_END)
    file_size = file.file.tell()
    file.file.seek(current_pos, os.SEEK_SET)
    if file_size > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"Upload exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)}MB limit.")


async def fetch_ravelry_data(pattern_id: str):
    """
    Async function to fetch data for a single Ravelry pattern.
    
    NOTE: This assumes your 'pattern_id' is a Ravelry pattern ID (e.g., '12345').
    """
    client = app_state.get("ravelry_client")
    if not client:
        return {"error": "Ravelry client not initialized"}

    # Fast in-memory cache to reduce repeated API calls.
    cached = app_state["ravelry_cache"].get(pattern_id)
    now = time.time()
    if cached and (now - cached["timestamp"] < RAVELRY_CACHE_TTL_SEC):
        return cached["payload"]
        
    last_error = None
    for attempt in range(RAVELRY_MAX_RETRIES):
        try:
            response = await client.get(f"/patterns/{pattern_id}.json")
            response.raise_for_status()

            data = response.json()
            pattern_data = data.get("pattern")

            if not pattern_data:
                return {"error": f"No 'pattern' key in Ravelry response for ID: {pattern_id}"}

            thumbnail = None
            if pattern_data.get("photos"):
                thumbnail = pattern_data["photos"][0].get("medium2_url")

            payload = {
                "name": pattern_data.get("name"),
                "url": f"https://www.ravelry.com/patterns/library/{pattern_data.get('permalink')}",
                "id": pattern_data.get("id"),
                "thumbnail": thumbnail,
            }
            app_state["ravelry_cache"][pattern_id] = {
                "timestamp": time.time(),
                "payload": payload,
            }
            return payload
        except httpx.HTTPStatusError as e:
            last_error = e
            status_code = e.response.status_code if e.response is not None else None
            # Retry only transient failures.
            if status_code not in {429, 500, 502, 503, 504}:
                break
        except Exception as e:
            last_error = e

        if attempt < RAVELRY_MAX_RETRIES - 1:
            await asyncio.sleep(0.25 * (2 ** attempt))

    if isinstance(last_error, httpx.HTTPStatusError):
        print(f"Ravelry API Error for ID {pattern_id}: {last_error}")
        return {
            "error": f"Failed to fetch Ravelry data for ID: {pattern_id}, Status: {last_error.response.status_code}"
        }
    print(f"Ravelry Error for ID {pattern_id}: {last_error}")
    return {"error": f"Failed to fetch Ravelry data for ID: {pattern_id}"}


# --- API Endpoint ---
@app.post("/recommend")
async def recommend_sweaters(
    request: Request,
    file: UploadFile = File(...),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    """
    The main API endpoint.
    1. Receives an uploaded image.
    2. Saves it temporarily.
    3. Runs the DINOv2/YOLO feature extraction.
    4. Queries the HNSW index.
    5. Maps the results to pattern IDs.
    6. Fetches Reddit data concurrently.
    7. Returns the final JSON.
    """
    
    _require_api_key(x_api_key)
    _enforce_rate_limit(request)

    request_start = time.perf_counter()
    phase_times = {}
    _validate_upload(file)

    # Save the uploaded file to a temporary path.
    safe_name = os.path.basename(file.filename or "upload.jpg")
    temp_file_path = f"temp_{uuid.uuid4().hex}_{safe_name}"
    try:
        save_start = time.perf_counter()
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        phase_times["save_upload_sec"] = time.perf_counter() - save_start

        # Basic image validation after save.
        img = cv2.imread(temp_file_path)
        if img is None:
            raise HTTPException(400, "Uploaded file is not a valid image.")
        h, w = img.shape[:2]
        if h < MIN_IMAGE_DIM or w < MIN_IMAGE_DIM:
            raise HTTPException(400, f"Image too small. Minimum size is {MIN_IMAGE_DIM}x{MIN_IMAGE_DIM}px.")

        # --- 1. Run Your ML Pipeline ---
        print(f"Processing image: {file.filename}...")
        infer_start = time.perf_counter()
        _, feature_vector, cropped_rgb = extract_features(temp_file_path, return_cropped_image=True)
        phase_times["feature_extraction_sec"] = time.perf_counter() - infer_start
        
        if isinstance(feature_vector, Exception):
            raise HTTPException(500, f"Failed to extract features: {feature_vector}")

        if not isinstance(feature_vector, np.ndarray):
            raise HTTPException(500, "Feature extraction did not return a valid vector.")

        # --- 2. Query HNSW Index ---
        print("Querying HNSW index...")
        query_start = time.perf_counter()
        index = app_state["hnsw_index"]
        # k=10 to get 10 recommendations
        # knn_query returns 2D arrays (for batch queries), so we take the first item [0]
        labels, distances = index.knn_query(feature_vector, k=10)
        phase_times["index_query_sec"] = time.perf_counter() - query_start
        
        query_labels = labels[0]
        query_distances = distances[0]

        # --- 3. Map IDs & Prep for Ravelry ---
        print("Mapping results to pattern IDs...")
        pattern_list = app_state["pattern_id_list"]
        
        tasks = []
        base_results = []
        
        for i, index_label in enumerate(query_labels):
            pattern_id = pattern_list[index_label]
            
            base_results.append({
                "pattern_id": pattern_id,
                "distance": float(query_distances[i]),
            })
            
            # Create an async task to fetch Reddit data in a separate thread
            # This lets us fetch all 10 posts in parallel instead of one by one
            pattern_id = pattern_id.split("_")[1]  # Capture variable for closure 
            tasks.append(fetch_ravelry_data(pattern_id))

        # --- 4. Fetch Reddit Data (Concurrently) ---
        print("Fetching Reddit data for 10 items...")
        ravelry_start = time.perf_counter()
        reddit_details_list = await asyncio.gather(*tasks)
        phase_times["ravelry_fetch_sec"] = time.perf_counter() - ravelry_start
        print("Reddit data fetched.")

        # --- 5. Combine and Return Results ---
        final_recommendations = []
        for i, base_res in enumerate(base_results):
            # Combine the base result (ID, distance) with the Reddit data
            base_res.update(reddit_details_list[i])
            final_recommendations.append(base_res)

    
        print("Generating XAI heatmap...")
        # Reuse pre-cropped RGB image from feature extraction to avoid a second YOLO pass.
        xai_start = time.perf_counter()
        model, processor = get_dino_components()
        heatmap_bytes = await asyncio.to_thread(
                generate_xai_heatmap_bytes, 
                temp_file_path, 
                model, 
                processor,
                cropped_rgb,
            )
        phase_times["xai_sec"] = time.perf_counter() - xai_start
            
        heatmap_base64 = None
        if heatmap_bytes:
            heatmap_base64 = base64.b64encode(heatmap_bytes).decode('utf-8')
            print("Heatmap generated and encoded.")

        phase_times["total_request_sec"] = time.perf_counter() - request_start
        print("Request timing:", {k: round(v, 4) for k, v in phase_times.items()})

        return {
            "recommendations": final_recommendations,
            "xai_heatmap_base64": heatmap_base64,
            "timings": phase_times,
        }
    except Exception as e:
        # Catch any other errors
        raise HTTPException(500, str(e))
    finally:
        # --- 6. Clean Up ---
        # ALWAYS remove the temp file, even if an error occurs
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


# --- Run the Server ---
if __name__ == "__main__":
    print("Starting Uvicorn server... Go to http://127.0.0.1:8000/docs")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)