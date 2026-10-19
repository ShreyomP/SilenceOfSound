import numpy as np
from .config import SLOWDOWN_SPEED_KNOTS, BASELINE_SPEED_KNOTS

# Sourced speed-to-noise relationship: Ross (1976) sixth-power law / ECHO Program vessel slowdown model
# Commercial vessel cavitation noise scales as: SL(v) = SL_ref + 60 * log10(v / v_ref)
BASELINE_SL_DB = 180.0
ROSS_SPEED_COEFF = 60.0
MASKING_THRESHOLD_DB = 100.0

class AnalyticsEngine:
    @staticmethod
    def model_footprint(sl, threshold=MASKING_THRESHOLD_DB, max_dist_km=100.0):
        if sl <= threshold:
            return 0.0
        d = np.linspace(0.01, max_dist_km, 5000)
        tl = 20 * np.log10(d * 1000)
        masking = np.where((sl - tl) <= threshold)[0]
        return float(d[masking[0]]) if len(masking) > 0 else float(max_dist_km)

    def process_recovery(self, sog):
        sog = float(sog)
        if sog <= 0.1:
            return 0.0, 0.0

        # Ross (1976): SL(v) = 180 + 60 * log10(v / 15)
        sl_now = BASELINE_SL_DB + (ROSS_SPEED_COEFF * np.log10(sog / BASELINE_SPEED_KNOTS))
        target_speed = SLOWDOWN_SPEED_KNOTS
        sl_slow = BASELINE_SL_DB + (ROSS_SPEED_COEFF * np.log10(min(sog, target_speed) / BASELINE_SPEED_KNOTS))

        r1 = self.model_footprint(sl_now)
        r2 = self.model_footprint(sl_slow)
        reclaimed_area = max(0.0, np.pi * (r1**2 - r2**2))
        return round(sl_now, 1), round(reclaimed_area, 2)
