import requests
import pandas as pd
from datetime import datetime

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

    # The API can return a dict wrapping the list
    if isinstance(data, dict):
        print(f"API returned a dict with keys: {list(data.keys())}")

        found_list = False
        for key in ['data', 'detections', 'results', 'reports']:
            if key in data and isinstance(data[key], list):
                print(f"Extracting data from key: '{key}'")
                data = data[key]
                found_list = True
                break

        if not found_list:
            print("Could not find a list in the API response")
            print("Check the keys printed above")
            return []

    df = pd.DataFrame(data)

    print(f"Total reports found: {len(df)}")

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
        # Convert millisecond timestamps to seconds
        if ts > 1000000000000: 
            ts = int(ts / 1000)
        
        human_date = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        
        results.append({
            'timestamp': ts,
            'date': human_date,
            'confidence': row.get('confidence', 'N/A')
        })
        
    return results

if __name__ == "__main__":
    sightings = get_whale_sightings()

    # Paste this list into the downloader
    print("sightings = [")
    for s in sightings[:10]:
        print(f"    {{'label': 'Whale_{s['date']}', 'epoch': {s['timestamp']}}},")
    print("]")