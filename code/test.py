import requests
import pandas as pd
from datetime import datetime

# --- CONFIGURATION ---
TARGET_NODE = 'rpi_port_townsend'  # or 'rpi_orcasound_lab' (Haro Strait)
API_URL = "https://live.orcasound.net/api/json/detections"

def get_whale_sightings():
    print(f"Fetching sighting reports from {API_URL}...")
    
    try:
        response = requests.get(API_URL)
        response.raise_for_status()
        json_resp = response.json()
    except Exception as e:
        print(f"API Error: {e}")
        return []

    # 1. Extract the list of detections
    # JSON:API usually puts the list in a top-level 'data' key
    detections_list = json_resp.get('data', []) if isinstance(json_resp, dict) else json_resp
    
    print(f"Processing {len(detections_list)} reports...")
    
    results = []
    for item in detections_list:
        # 2. Extract Data safely
        # Note: Some APIs put data directly in item, others in item['attributes']
        attrs = item.get('attributes', item) 
        
        # 3. Find the Node/Feed ID
        # Strategy: Check multiple possible locations for the ID
        node_id = None
        
        # Location A: Top level (Simple JSON)
        if 'nodeId' in item: node_id = item['nodeId']
        
        # Location B: Nested Relationship (JSON:API Standard)
        # Structure: item -> relationships -> feed -> data -> id
        elif 'relationships' in item:
            try:
                node_id = item['relationships']['feed']['data']['id']
            except (KeyError, TypeError):
                pass
        
        # Location C: Inside attributes
        elif 'node_id' in attrs: node_id = attrs['node_id']

        # 4. Filter for Port Townsend
        # If we found a node_id, check if it matches
        if node_id and 'port_townsend' in node_id.lower():
            
            # Check for "Whale" label
            # Note: The label might be in 'label', 'category', or 'commonName'
            label = str(attrs.get('category', '')).lower() + str(attrs.get('name', '')).lower()
            
            if 'whale' in label or 'orca' in label:
                # Extract Timestamp
                # API likely returns ISO string (e.g., "2025-11-11T15:00:00Z") or Epoch
                raw_time = attrs.get('startTime') or attrs.get('timestamp')
                
                # Convert to Epoch for S3
                try:
                    if isinstance(raw_time, str):
                        # Parse ISO format
                        dt = datetime.fromisoformat(raw_time.replace('Z', '+00:00'))
                        ts = int(dt.timestamp())
                    else:
                        # Assume it's already a number (ms or s)
                        ts = int(raw_time / 1000) if raw_time > 10000000000 else int(raw_time)
                    
                    human_date = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
                    
                    results.append({
                        'label': f"Whale_{human_date.replace(' ', '_')}", 
                        'epoch': ts
                    })
                except Exception as e:
                    continue # Skip if timestamp format is weird

    print(f"\nFound {len(results)} confirmed Port Townsend sightings.")
    return results

# --- EXECUTION ---
if __name__ == "__main__":
    sightings = get_whale_sightings()
    
    # Output for your Downloader Script
    print("\n--- COPY THIS LIST INTO YOUR DOWNLOADER ---")
    print("sightings = [")
    for s in sightings[:10]: # Print top 10 as example
        print(f"    {{'label': 'Whale_{s['date']}', 'epoch': {s['timestamp']}}},")
    print("]")