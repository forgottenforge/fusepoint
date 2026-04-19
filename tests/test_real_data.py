"""
Test fusepoint on REAL experimental data from quantum hardware.

Uses ZZ correlator data from Rigetti Ankaa-3 (published in AVS Quantum Science).
These are real measurements — not synthetic.

The data shows ZZ correlators as a function of circuit depth.
For circuits with real entanglement structure (TFIM, Heisenberg, Kitaev),
correlations should decay with depth — the tipping point is where
the signal is lost to noise.

For "zero_evolution" circuits (no real evolution), correlations should
stay constant — no tipping point expected.
"""

import numpy as np
import csv
import os
import sys

from fusepoint import analyze

# Load the real data
DATA_PATH = r"d:\code\onto\zz_correlators_by_depth.csv"

# Parse CSV
depths = None
circuits = {}

with open(DATA_PATH, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        circuit = row["circuit"]
        label = row["label"]
        observable = row["observable"]

        if depths is None:
            # Extract depth columns
            depth_cols = [c for c in row.keys() if c.startswith("depth_")]
            depths = np.array([int(c.replace("depth_", "")) for c in depth_cols], dtype=float)

        values = np.array([float(row[c]) for c in depth_cols])

        key = f"{circuit}_{label}"
        if key not in circuits:
            circuits[key] = {"label": label, "observables": {}}
        circuits[key]["observables"][observable] = values


print(f"Loaded {len(circuits)} circuits with {len(depths)} depth points each")
print(f"Depths: {depths}")
print()

# Categorize circuits
STRUCTURED = []  # Should have tipping points (signal decays)
TRIVIAL = []     # Should NOT have tipping points (constant)

for key, info in circuits.items():
    if "zero_evolution" in info["label"]:
        TRIVIAL.append(key)
    else:
        STRUCTURED.append(key)

print(f"Structured circuits (expect tipping points): {len(STRUCTURED)}")
print(f"Trivial circuits (expect NO tipping points): {len(TRIVIAL)}")
print()

# ================================================================
# Test 1: Structured circuits should detect signal decay
# ================================================================
print("=" * 60)
print("TEST 1: Structured circuits — signal decay detection")
print("=" * 60)

structured_results = []
os.makedirs("real_data_cards", exist_ok=True)

# Pick the nearest-neighbor ZZ correlators (strongest signal)
for key in sorted(STRUCTURED):
    info = circuits[key]
    label = info["label"]

    # Use Z_0_1 (nearest-neighbor) as the primary observable
    if "Z_0_1" not in info["observables"]:
        continue

    y = info["observables"]["Z_0_1"]
    r = analyze(depths, y, current_x=0, label=f"{key}: {label} (Z_0_1)")

    structured_results.append({
        "key": key,
        "label": label,
        "score": r.score,
        "grade": r.grade,
        "critical_depth": r.critical_x,
        "kappa": r.kappa,
        "p_value": r.p_value,
        "significant": r.p_value < 0.05,
    })

    sig = "***" if r.p_value < 0.01 else ("*" if r.p_value < 0.05 else "")
    print(f"  {key:30s}  score={r.score:3d}  grade={r.grade:8s}  "
          f"critical_depth={r.critical_x:5.1f}  kappa={r.kappa:5.2f}  "
          f"p={r.p_value:.4f} {sig}")

# Save a few example cards
highlight_circuits = [k for k in sorted(STRUCTURED) if any(
    t in circuits[k]["label"] for t in ["tfim_critical", "heisenberg", "kitaev_detuned"]
)][:3]

for key in highlight_circuits:
    info = circuits[key]
    y = info["observables"]["Z_0_1"]
    r = analyze(depths, y, current_x=0, label=f"{info['label']} (Z_0_1)")
    r.save(f"real_data_cards/{key}_Z01.png")
    print(f"  -> Saved: real_data_cards/{key}_Z01.png")

print()

# ================================================================
# Test 2: Trivial circuits should NOT detect tipping points
# ================================================================
print("=" * 60)
print("TEST 2: Trivial circuits (zero_evolution) — should score LOW")
print("=" * 60)

trivial_results = []
for key in sorted(TRIVIAL):
    info = circuits[key]
    label = info["label"]

    if "Z_0_1" not in info["observables"]:
        continue

    y = info["observables"]["Z_0_1"]
    r = analyze(depths, y, label=f"{key}: {label} (Z_0_1)")

    trivial_results.append({
        "key": key,
        "label": label,
        "score": r.score,
        "p_value": r.p_value,
        "significant": r.p_value < 0.05,
    })

    print(f"  {key:30s}  score={r.score:3d}  grade={r.grade:8s}  "
          f"p={r.p_value:.4f}  {'WRONG!' if r.p_value < 0.05 else 'correct'}")

print()

# ================================================================
# Test 3: TFIM circuits — average over all correlator pairs
# ================================================================
print("=" * 60)
print("TEST 3: TFIM critical — averaged ZZ correlator")
print("=" * 60)

tfim_keys = [k for k in STRUCTURED if "tfim_critical" in circuits[k]["label"]]
for key in tfim_keys:
    info = circuits[key]
    # Average over ALL nearest-neighbor correlators
    nn_obs = [k for k in info["observables"] if k in
              ["Z_0_1", "Z_1_2", "Z_2_3", "Z_3_4", "Z_4_5"]]
    if not nn_obs:
        continue

    y_avg = np.mean([info["observables"][obs] for obs in nn_obs], axis=0)
    r = analyze(depths, y_avg, current_x=0,
                label=f"{key}: TFIM critical (NN-averaged)")

    print(f"  {key}: score={r.score}, critical_depth={r.critical_x:.1f}, "
          f"kappa={r.kappa:.2f}, p={r.p_value:.4f}")
    r.save(f"real_data_cards/{key}_nn_averaged.png")
    print(f"  -> Saved: real_data_cards/{key}_nn_averaged.png")

print()

# ================================================================
# Summary
# ================================================================
print("=" * 60)
print("SUMMARY")
print("=" * 60)

# Structured circuits
n_struct_sig = sum(1 for r in structured_results if r["significant"])
n_struct = len(structured_results)
struct_scores = [r["score"] for r in structured_results]

print(f"\nStructured circuits:")
print(f"  Detected tipping point: {n_struct_sig}/{n_struct} ({100*n_struct_sig/n_struct:.0f}%)")
print(f"  Average score: {np.mean(struct_scores):.1f}")
print(f"  Score range: {min(struct_scores)} - {max(struct_scores)}")

# Trivial circuits
n_triv_fp = sum(1 for r in trivial_results if r["significant"])
n_triv = len(trivial_results)
triv_scores = [r["score"] for r in trivial_results]

print(f"\nTrivial circuits (zero_evolution):")
print(f"  False positives: {n_triv_fp}/{n_triv} ({100*n_triv_fp/n_triv:.0f}%)")
print(f"  Average score: {np.mean(triv_scores):.1f}")
print(f"  Score range: {min(triv_scores)} - {max(triv_scores)}")

# Verdict
print(f"\nVERDICT:")
if n_struct_sig / n_struct > 0.5 and n_triv_fp / n_triv < 0.2:
    print("  fusepoint correctly distinguishes real quantum signals from noise!")
elif n_struct_sig / n_struct > 0.3:
    print("  fusepoint detects some signals but misses others (limited data points)")
else:
    print("  fusepoint struggles with this data format — needs investigation")

print()
