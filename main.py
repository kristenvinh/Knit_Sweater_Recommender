import os
import pickle
import shutil
import asyncio
import hnswlib
import httpx
import uvicorn
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import base64
from dino_feature_extraction import extract_features, FEATURE_DIM
from xaiutil import generate_xai_heatmap_bytes


# --- Configuration & Constants ---
INDEX_FILE = 'sweater_hnsw_DINO_yolo_pose.bin'
PATTERN_IDS_FILE = 'pattern_ids_DINO_yolo_pose.pkl'

# --- Global State (Loaded on Startup) ---
app_state = {
    "hnsw_index": None,
    "pattern_id_list": None,
    "reddit_client": None,
}

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Sweater Recommender API",
    description="Upload a sweater image to find similar patterns.",
)

# --- Add CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (for development)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- Startup Event Handler ---
@app.on_event("startup")
async def load_models_on_startup():
    """
    This function runs ONCE when the server starts.
    It loads all our heavy assets into memory.
    """
    print("--- Server starting up... ---")

    # 1. Load DINOv2 & YOLO models
    # This was already triggered by the import at the top of the file.
    # We just print a confirmation.
    print(f"✅ DINOv2/YOLO models loaded (from import). Feature dim: {FEATURE_DIM}")

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
        
    print("--- Server startup complete. Ready for requests. ---")

# --- CHANGED: Helper function rewritten for Ravelry ---
async def fetch_ravelry_data(pattern_id: str):
    """
    Async function to fetch data for a single Ravelry pattern.
    
    NOTE: This assumes your 'pattern_id' is a Ravelry pattern ID (e.g., '12345').
    """
    client = app_state.get("ravelry_client")
    if not client:
        return {"error": "Ravelry client not initialized"}
        
    try:
        # Make the async API call
        response = await client.get(f"/patterns/{pattern_id}.json")
        
        # Raise an exception for bad responses (4xx, 5xx)
        response.raise_for_status() 
        
        data = response.json()
        pattern_data = data.get("pattern")
        
        if not pattern_data:
            return {"error": f"No 'pattern' key in Ravelry response for ID: {pattern_id}"}

        # Extract the data you want
        thumbnail = None
        if pattern_data.get("photos"):
            thumbnail = pattern_data["photos"][0].get("medium2_url")
            
        return {
            "name": pattern_data.get("name"),
            "url": f"https://www.ravelry.com/patterns/library/{pattern_data.get('permalink')}",
            "id": pattern_data.get("id"),
            "thumbnail": thumbnail,
        }
    except httpx.HTTPStatusError as e:
        print(f"Ravelry API Error for ID {pattern_id}: {e}")
        return {"error": f"Failed to fetch Ravelry data for ID: {pattern_id}, Status: {e.response.status_code}"}
    except Exception as e:
        print(f"Ravelry Error for ID {pattern_id}: {e}")
        return {"error": f"Failed to fetch Ravelry data for ID: {pattern_id}"}


# --- API Endpoint ---
@app.post("/recommend")
async def recommend_sweaters(file: UploadFile = File(...)):
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
    
    # Save the uploaded file to a temporary path
    # Your `extract_features` function needs a file path to read from.
    temp_file_path = f"temp_{file.filename}"
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # --- 1. Run Your ML Pipeline ---
        print(f"Processing image: {file.filename}...")
        # We call your function directly from the imported script
        _, feature_vector = extract_features(temp_file_path)
        
        if isinstance(feature_vector, Exception):
            raise HTTPException(500, f"Failed to extract features: {feature_vector}")

        if not isinstance(feature_vector, np.ndarray):
            raise HTTPException(500, "Feature extraction did not return a valid vector.")

        # --- 2. Query HNSW Index ---
        print("Querying HNSW index...")
        index = app_state["hnsw_index"]
        # k=10 to get 10 recommendations
        # knn_query returns 2D arrays (for batch queries), so we take the first item [0]
        labels, distances = index.knn_query(feature_vector, k=10)
        
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
                "pattern_id": pattern_id
            })
            
            # Create an async task to fetch Reddit data in a separate thread
            # This lets us fetch all 10 posts in parallel instead of one by one
            pattern_id = pattern_id.split("_")[1]  # Capture variable for closure 
            tasks.append(fetch_ravelry_data(pattern_id))

        # --- 4. Fetch Reddit Data (Concurrently) ---
        print("Fetching Reddit data for 10 items...")
        reddit_details_list = await asyncio.gather(*tasks)
        print("Reddit data fetched.")

        # --- 5. Combine and Return Results ---
        final_recommendations = []
        for i, base_res in enumerate(base_results):
            # Combine the base result (ID, distance) with the Reddit data
            base_res.update(reddit_details_list[i])
            final_recommendations.append(base_res)

    
        print("Generating XAI heatmap...")
            # Run the CPU-heavy XAI task in a separate thread
        heatmap_bytes = await asyncio.to_thread(
                generate_xai_heatmap_bytes, 
                temp_file_path, 
                model, 
                processor
            )
            
        heatmap_base64 = None
        if heatmap_bytes:
            heatmap_base64 = base64.b64encode(heatmap_bytes).decode('utf-8')
            print("Heatmap generated and encoded.")

        return {
            "recommendations": final_recommendations,
            "xai_heatmap_base64": heatmap_base64,
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