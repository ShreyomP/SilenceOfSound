import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

script_dir = Path(__file__).resolve().parent
PROJECT_ROOT = script_dir.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from core.analytics import BASELINE_SL_DB, ROSS_SPEED_COEFF, MASKING_THRESHOLD_DB
from core.config import SLOWDOWN_SPEED_KNOTS, BASELINE_SPEED_KNOTS

def model_acoustic_footprint(source_level_db, threshold_db=MASKING_THRESHOLD_DB, max_km=50.0):
    distances = np.linspace(0.1, max_km, 2000)
    transmission_loss = 20 * np.log10(distances * 1000)
    received_levels = source_level_db - transmission_loss

    masking_points = np.where(received_levels <= threshold_db)[0]
    if len(masking_points) > 0:
        masking_radius = float(distances[masking_points[0]])
    else:
        masking_radius = float(distances[-1])

    return distances, received_levels, masking_radius

# Sourced from Ross (1976) sixth-power law / ECHO Program vessel slowdown model
v_baseline = 18.0
v_slowdown = SLOWDOWN_SPEED_KNOTS  # 7.0 knots

sl_18kts = BASELINE_SL_DB + ROSS_SPEED_COEFF * np.log10(v_baseline / BASELINE_SPEED_KNOTS)
sl_7kts = BASELINE_SL_DB + ROSS_SPEED_COEFF * np.log10(v_slowdown / BASELINE_SPEED_KNOTS)
threshold = MASKING_THRESHOLD_DB

d, levels_18, r_18 = model_acoustic_footprint(sl_18kts, threshold_db=threshold, max_km=50.0)
_, levels_7, r_7 = model_acoustic_footprint(sl_7kts, threshold_db=threshold, max_km=50.0)

plt.figure(figsize=(10, 6))
plt.plot(d, levels_18, label=f'{int(v_baseline)} Knots (Baseline)', color='red', linewidth=2)
plt.plot(d, levels_7, label=f'{int(v_slowdown)} Knots (Slowdown)', color='green', linewidth=2)
plt.axhline(y=threshold, color='black', linestyle='--', label=f'Masking Threshold ({int(threshold)} dB)')
plt.fill_between(d, levels_7, levels_18, color='orange', alpha=0.2, label='Reclaimed Habitat Area')

plt.title('Acoustic Footprint Reduction: 18 kts vs 7 kts', fontsize=14)
plt.xlabel('Distance from Vessel (km)', fontsize=12)
plt.ylabel('Received Noise Level (dB)', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

out_path = PROJECT_ROOT / 'results_and_graphs' / 'acoustic_footprint_graph.png'
out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path, dpi=200, bbox_inches='tight')
plt.close()

reclaimed_area = np.pi * (r_18**2 - r_7**2)
print(f"18-Knot Masking Radius: {r_18:.2f} km")
print(f"7-Knot Masking Radius:  {r_7:.2f} km")
print(f"Reclaimed Area:        {reclaimed_area:.2f} km²")