import pandas as pd
import numpy as np
import os
import sys


if __package__ in {None, ""}:
    BACKEND_ROOT = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
    )

    if BACKEND_ROOT not in sys.path:
        sys.path.append(BACKEND_ROOT)

from app.config.Anomaly_Health_Monitering.Config import Config
from app.utils.Anomaly_Health_Monitering.model_utils import (
    get_raw_xs_columns,
    get_w_columns,
    get_xv_columns,
)

df = pd.read_csv(Config.SCALED_CSV, nrows=200_000)

xs_cols = get_raw_xs_columns(df)
w_cols = get_w_columns(df)
xv_cols = get_xv_columns(df)

print("Raw X_s columns:", xs_cols)
print("W columns:", w_cols)
print("X_v columns:", xv_cols)

print("\nChecking exact duplicate X_s columns:")
for i, c1 in enumerate(xs_cols):
    for c2 in xs_cols[i + 1:]:
        same = np.allclose(df[c1].values, df[c2].values, atol=1e-12)
        if same:
            print("DUPLICATE OR NEAR DUPLICATE:", c1, c2)

print("\nTop feature-target correlations:")
feature_cols = w_cols + xv_cols

for target in xs_cols:
    corrs = {}
    for feature in feature_cols:
        corr = df[[feature, target]].corr().iloc[0, 1]
        if not np.isnan(corr):
            corrs[feature] = abs(corr)

    top = sorted(corrs.items(), key=lambda x: x[1], reverse=True)[:5]
    print("\nTarget:", target)
    for feature, corr in top:
        print(f"  {feature}: {corr:.6f}")
