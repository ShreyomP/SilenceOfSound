import os
import time
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from core.config import AIS_API_KEY, DETECTION_THRESHOLD
from core.models import AcousticEngine
from core.fetcher import QuiltS3Fetcher
from core.maritime import MaritimeEngine
from core.analytics import AnalyticsEngine

app = FastAPI(title="OrcaSafe API")

static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
@app.get("/app")
async def root():
    return FileResponse(os.path.join(static_dir, "index.html"))

# Supported hydrophone nodes
HYDROPHONE_NODES = {
    "rpi_port_townsend": {"id": "rpi_port_townsend", "name": "Port Townsend", "lat": 48.1357, "lng": -122.7597},
    "rpi_bush_point": {"id": "rpi_bush_point", "name": "Bush Point", "lat": 48.0340, "lng": -122.6040},
    "rpi_orcasound_lab": {"id": "rpi_orcasound_lab", "name": "Orcasound Lab", "lat": 48.5580, "lng": -123.1810},
    "rpi_point_robinson": {"id": "rpi_point_robinson", "name": "Point Robinson", "lat": 47.3880, "lng": -122.3740},
    "rpi_sunset_bay": {"id": "rpi_sunset_bay", "name": "Sunset Bay", "lat": 47.8000, "lng": -122.3800},
    "rpi_andrews_bay": {"id": "rpi_andrews_bay", "name": "Andrews Bay", "lat": 48.5400, "lng": -123.1600},
    "rpi_mast_center": {"id": "rpi_mast_center", "name": "MAST Center", "lat": 47.3500, "lng": -122.3300},
    "rpi_north_sjc": {"id": "rpi_north_sjc", "name": "North San Juan Channel", "lat": 48.6000, "lng": -123.0000}
}

model_path = PROJECT_ROOT / "models" / "orca_safe_brain.pth"
acoustic = AcousticEngine(str(model_path) if model_path.exists() else "orca_safe_brain.pth")
maritime = MaritimeEngine(AIS_API_KEY)
analytics = AnalyticsEngine()

# In-memory cache per node to conserve CPU and avoid redundant S3/ML runs
node_cache = {}
CACHE_TTL_SECONDS = 30.0

def sync_process_node(node_id: str):
    """Executes S3 audio download, decoding, and ML inference in a threadpool."""
    fetcher = QuiltS3Fetcher(node=node_id)
    img, img_b64, timestamp = fetcher.get_latest_spectrogram()
    if not img:
        return None, None, None, getattr(fetcher, 'playlist_url', None)
    prob = acoustic.run_inference(img)
    return prob, img_b64, timestamp, getattr(fetcher, 'playlist_url', None)

@app.get("/api/nodes")
async def get_nodes():
    return {"nodes": list(HYDROPHONE_NODES.values())}

@app.get("/api/status")
async def get_status(node: str = "rpi_port_townsend"):
    now = time.time()
    if node in node_cache and (now - node_cache[node]["cached_at"]) < CACHE_TTL_SECONDS:
        return node_cache[node]["data"]

    # Run blocking S3/ffmpeg/model inference off the event loop
    prob, img_b64, timestamp, playlist_url = await run_in_threadpool(sync_process_node, node)

    if prob is None:
        response_data = {
            "error": "Failed to fetch audio from S3 node",
            "offline": True,
            "node": node
        }
        node_cache[node] = {"data": response_data, "cached_at": now}
        return response_data

    orca_detected = prob > DETECTION_THRESHOLD
    vessels = []

    if orca_detected:
        node_info = HYDROPHONE_NODES.get(node, {"lat": 48.1357, "lng": -122.7597})
        try:
            # Query AIS around the detecting node's coordinates
            raw_vessels = await maritime.fetch_live_ais(
                node_lat=node_info["lat"],
                node_lng=node_info["lng"]
            )
            for v in raw_vessels:
                # Measure distance from this detecting hydrophone
                dist = maritime.get_distance(
                    node_info["lat"], node_info["lng"],
                    v["latitude"], v["longitude"]
                )
                sl, reclaimed = analytics.process_recovery(v["sog"])
                v["distance_km"] = dist
                v["source_level_db"] = sl
                v["reclaimed_area_km2"] = reclaimed
                vessels.append(v)
        except Exception as e:
            print(f"Error fetching AIS for {node}: {e}")

    response_data = {
        "node": node,
        "timestamp": timestamp.isoformat() if timestamp else None,
        "orca_detected": orca_detected,
        "detection_confidence": prob,
        "spectrogram_b64": img_b64,
        "vessels": vessels,
        "audio_url": playlist_url
    }

    node_cache[node] = {"data": response_data, "cached_at": now}
    return response_data
