import asyncio
import websockets
import json
import time
from math import radians, sin, cos, asin, sqrt
from .config import NAV_STATUS_MAP

class MaritimeEngine:
    def __init__(self, api_key):
        self.api_key = api_key

    def build_bounding_box(self, node_lat, node_lng, radius_km=30.0):
        # 1 deg lat ~ 111 km, 1 deg lon ~ 111 * cos(lat) km
        delta_lat = radius_km / 111.0
        delta_lng = radius_km / (111.0 * max(0.1, cos(radians(node_lat))))
        return [
            [round(node_lat - delta_lat, 4), round(node_lng - delta_lng, 4)],
            [round(node_lat + delta_lat, 4), round(node_lng + delta_lng, 4)]
        ]

    async def fetch_live_ais(self, node_lat=48.1357, node_lng=-122.7597, radius_km=30.0):
        if not self.api_key:
            return []

        bbox = self.build_bounding_box(node_lat, node_lng, radius_km)
        ships = {}

        try:
            async with websockets.connect("wss://stream.aisstream.io/v0/stream", open_timeout=10) as ws:
                await ws.send(json.dumps({"APIKey": self.api_key, "BoundingBoxes": [bbox]}))
                start_time = time.time()

                # Listen for 3 seconds
                while time.time() - start_time < 3:
                    try:
                        data = json.loads(await asyncio.wait_for(ws.recv(), timeout=1.0))
                        if "MetaData" in data and "Message" in data and "PositionReport" in data["Message"]:
                            mmsi = data["MetaData"]["MMSI"]
                            nav_code = data["Message"]["PositionReport"].get("NavigationalStatus", 15)
                            status_text = NAV_STATUS_MAP.get(nav_code, f"Status {nav_code}")
                            
                            # Deduplicate: keep only the latest position report per MMSI
                            ships[mmsi] = {
                                "mmsi": mmsi,
                                "name": data["MetaData"].get("ShipName", "Unknown").strip(),
                                "latitude": data["MetaData"]["latitude"],
                                "longitude": data["MetaData"]["longitude"],
                                "sog": data["Message"]["PositionReport"].get("Sog", 0),
                                "status": status_text
                            }
                    except asyncio.TimeoutError:
                        continue
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"AISStream error: {e}")

        return list(ships.values())

    @staticmethod
    def get_distance(lat1, lon1, lat2, lon2):
        r_lon1, r_lat1, r_lon2, r_lat2 = map(radians, [lon1, lat1, lon2, lat2])
        dlon, dlat = r_lon2 - r_lon1, r_lat2 - r_lat1
        a = sin(dlat / 2)**2 + cos(r_lat1) * cos(r_lat2) * sin(dlon / 2)**2
        return round(2 * asin(sqrt(a)) * 6371, 2)
