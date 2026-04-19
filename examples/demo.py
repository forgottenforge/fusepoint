"""
fusepoint — Hero Examples

Scenarios every engineer understands:
1. ML Learning Rate  — where does training explode?
2. Server Load       — where does latency spike?
3. Pure Noise        — negative control (should score LOW)
4. Before/After      — the viral comparison card
5. DataFrame mode    — the natural API for CSV data
"""

import numpy as np
from fusepoint import analyze, compare

# ============================================================
# 1. ML LEARNING RATE
# ============================================================
print("=" * 60)
print("EXAMPLE 1: ML Learning Rate — Where does training explode?")
print("=" * 60)

lr = np.linspace(1e-4, 0.10, 120)
rng1 = np.random.default_rng(42)
loss = 0.15 + 4.5 / (1 + np.exp(-200 * (lr - 0.04)))
loss += rng1.normal(0, 0.03, 120)

card = analyze(lr, loss, current_x=0.01,
               x_name="Learning Rate", y_name="Loss",
               label="ML Learning Rate")
print(card.summary())
card.save("example_ml_learning_rate.png")
print("→ Saved: example_ml_learning_rate.png\n")

# ============================================================
# 2. SERVER LOAD
# ============================================================
print("=" * 60)
print("EXAMPLE 2: Server Load — Where does latency spike?")
print("=" * 60)

requests = np.linspace(100, 10000, 100)
capacity = 7500
latency = 10 + 50 / (1 - np.clip(requests / capacity, 0, 0.98))
noise2 = np.random.default_rng(7).normal(0, 5, size=len(requests))
latency += noise2

card2 = analyze(requests, latency, current_x=5000,
                x_name="Concurrent Requests", y_name="Response Time (ms)",
                label="Server Response Time")
print(card2.summary())
card2.save("example_server_load.png")
print("→ Saved: example_server_load.png\n")

# ============================================================
# 3. PURE NOISE (Negative Control)
# ============================================================
print("=" * 60)
print("EXAMPLE 3: Pure Noise — Should score LOW (no real tipping point)")
print("=" * 60)

x_noise = np.linspace(0, 1, 80)
y_noise = np.random.default_rng(99).normal(5.0, 1.0, size=80)

card3 = analyze(x_noise, y_noise, label="Random Noise (Control)")
print(card3.summary())
card3.save("example_noise_control.png")
print("→ Saved: example_noise_control.png\n")

# ============================================================
# 4. BEFORE / AFTER COMPARISON
# ============================================================
print("=" * 60)
print("EXAMPLE 4: Before/After — The improvement story")
print("=" * 60)

x_param = np.linspace(0, 1, 100)
y_before = 1.0 / (1 + np.exp(-40 * (x_param - 0.30)))
y_before += np.random.default_rng(10).normal(0, 0.02, size=100)

y_after = 1.0 / (1 + np.exp(-40 * (x_param - 0.80)))
y_after += np.random.default_rng(11).normal(0, 0.02, size=100)

delta = compare(x_param, y_before, x_param, y_after,
                current_x=0.15, label_before="Before Fix", label_after="After Fix")
print(delta.summary())
delta.save("example_comparison.png")
print("→ Saved: example_comparison.png\n")

# ============================================================
# 5. DATAFRAME MODE — The natural API
# ============================================================
print("=" * 60)
print("EXAMPLE 5: DataFrame mode — analyze(df, x='col', y='col')")
print("=" * 60)

try:
    import pandas as pd

    # Simulate a CSV with named columns
    temp = np.linspace(20, 120, 80)
    df = pd.DataFrame({
        "temperature_c": temp,
        "yield_percent": np.clip(
            95 - 0.02 * (temp - 20)
            - 80 / (1 + np.exp(-0.3 * (temp - 85)))
            + np.random.default_rng(55).normal(0, 1.5, 80),
            0, 100
        ),
    })

    card5 = analyze(df, x="temperature_c", y="yield_percent",
                    current_x=60, label="Chemical Reactor Yield")
    print(card5.summary())
    print(f"  x_name = '{card5.x_name}', y_name = '{card5.y_name}'")
    card5.save("example_dataframe.png")
    print("→ Saved: example_dataframe.png\n")
except ImportError:
    print("  (pandas not installed, skipping DataFrame example)")

print("\nDone! Check the PNG files.")
