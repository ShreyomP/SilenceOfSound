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
        data = response.json()
    except Exception as e:
        print(f"API Error: {e}")
        return []

    # --- FIX STARTS HERE ---
    # 1. Inspect the structure if it's a Dictionary
    if isinstance(data, dict):
        print(f"API returned a Dictionary with keys: {list(data.keys())}")
        
        # 2. Find the key that holds the list (Common API patterns)
        found_list = False
        for key in ['data', 'detections', 'results', 'reports']:
            if key in data and isinstance(data[key], list):
                print(f"Extracting data from key: '{key}'")
                data = data[key]
                found_list = True
                break
        
        # 3. If no list found, warn the user
        if not found_list:
            print("CRITICAL WARNING: Could not find a list inside the API response.")
            print("Please check the 'Keys' printed above and update the script manually.")
            return []

    # 4. Now 'data' is a list, so DataFrame creation will work
    df = pd.DataFrame(data)
    # --- FIX ENDS HERE ---

    print(f"Total reports found: {len(df)}")
    
    # (The rest of your filtering logic remains the same)
    if 'nodeId' not in df.columns:
        print("Error: 'nodeId' column not found. Available columns:", df.columns)
        return []

    print(f"Available Nodes: {df['nodeId'].unique()}")
    
    whale_sightings = df[
        (df['nodeId'].str.contains('port_townsend', case=False, na=False)) & 
        (df['category'] == 'whale') 
    ].copy()

    print(f"\nFound {len(whale_sightings)} confirmed whale sightings for {TARGET_NODE}")
    
    results = []
    for index, row in whale_sightings.iterrows():
        ts = row['startTime']
        # Fix millisecond timestamps if necessary
        if ts > 1000000000000: 
            ts = int(ts / 1000)
        
        human_date = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        
        results.append({
            'timestamp': ts,
            'date': human_date,
            'confidence': row.get('confidence', 'N/A')
        })
        
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