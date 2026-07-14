import pandas as pd
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Tuple
import gc
import os
import sys

if __package__ in {None, ""}:
    BACKEND_ROOT = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
    )

    if BACKEND_ROOT not in sys.path:
        sys.path.insert(0, BACKEND_ROOT)
 

from app.config.Anomaly_Health_Monitering.config import Config


df = pd.read_csv(Config.RESIDUAL_ANOMALY_CSV, usecols=["split", "gmm_context_id", "residual_alert_level"])

print("\nAlert counts by split:")
print(pd.crosstab(df["split"], df["residual_alert_level"]))

print("\nAlert percentages by split:")
print(pd.crosstab(df["split"], df["residual_alert_level"], normalize="index") * 100)

print("\nAlert counts by context:")
print(pd.crosstab(df["gmm_context_id"], df["residual_alert_level"]))

print("\nAlert percentages by context:")
print(pd.crosstab(df["gmm_context_id"], df["residual_alert_level"], normalize="index") * 100)
