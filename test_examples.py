"""
Test all domain-specific examples through FUSE to verify claims.
Each dataset should produce a clear tipping point with the values we document.
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
from fusekit import analyze

rng = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# 1. Physics — Quantum Entanglement vs Magnetic Field
# ---------------------------------------------------------------------------
print("=" * 70)
print("1. PHYSICS — Entanglement Entropy vs Magnetic Field Strength")
print("=" * 70)

B = np.linspace(0.5, 4.0, 80)
# Entanglement entropy: low in ordered phase, spikes at phase transition ~2.3T
entanglement = 0.2 + 3.0 / (1 + np.exp(-8 * (B - 2.3))) + rng.normal(0, 0.1, 80)

r1 = analyze(B, entanglement, current_x=1.9,
             x_name="Magnetic Field (T)",
             y_name="Entanglement Entropy",
             label="Quantum Phase Transition")
print(f"  Score:         {r1.score}")
print(f"  Grade:         {r1.grade}")
print(f"  Tipping Point: {r1.critical_x:.3f} T")
print(f"  CI:            [{r1.ci[0]:.3f}, {r1.ci[1]:.3f}]")
print(f"  Kappa:         {r1.kappa:.2f}")
print(f"  p-value:       {r1.p_value:.4f}")
print(f"  Safety Margin: {r1.safety_margin}")
print(f"  Diagnosis:     {r1.diagnosis}")
print()

# Save test CSV
pd.DataFrame({"magnetic_field_T": B, "entanglement_entropy": entanglement}).to_csv(
    "d:/code/sweetspot/examples/physics_entanglement.csv", index=False)

# ---------------------------------------------------------------------------
# 2. Fusion Reactor — Surface Erosion vs Plasma Temperature
# ---------------------------------------------------------------------------
print("=" * 70)
print("2. FUSION REACTOR — Surface Erosion Rate vs Plasma Temperature")
print("=" * 70)

temp_fusion = np.linspace(400, 1600, 100)
# Erosion rate: low and linear below ~1180°C, then exponential
erosion = 0.5 + 20.0 / (1 + np.exp(-0.03 * (temp_fusion - 1180))) + rng.normal(0, 0.3, 100)

r2 = analyze(temp_fusion, erosion, current_x=1100,
             x_name="Plasma Temperature (°C)",
             y_name="Erosion Rate (μm/pulse)",
             label="Divertor Tile Erosion")
print(f"  Score:         {r2.score}")
print(f"  Grade:         {r2.grade}")
print(f"  Tipping Point: {r2.critical_x:.1f} °C")
print(f"  CI:            [{r2.ci[0]:.1f}, {r2.ci[1]:.1f}]")
print(f"  Kappa:         {r2.kappa:.2f}")
print(f"  p-value:       {r2.p_value:.4f}")
print(f"  Safety Margin: {r2.safety_margin}")
print(f"  Diagnosis:     {r2.diagnosis}")
print()

pd.DataFrame({"plasma_temperature_C": temp_fusion, "erosion_rate_um_per_pulse": erosion}).to_csv(
    "d:/code/sweetspot/examples/fusion_erosion.csv", index=False)

# ---------------------------------------------------------------------------
# 3. Chemistry — Reaction Rate vs Temperature
# ---------------------------------------------------------------------------
print("=" * 70)
print("3. CHEMISTRY — Reaction Rate vs Temperature")
print("=" * 70)

temp_chem = np.linspace(20, 200, 100)
reaction_rate = 0.01 + 5.0 / (1 + np.exp(-0.15 * (temp_chem - 150))) + rng.normal(0, 0.06, 100)

r3 = analyze(temp_chem, reaction_rate, current_x=120,
             x_name="Temperature (°C)",
             y_name="Reaction Rate (mol/s)",
             label="Exothermic Reaction Kinetics")
print(f"  Score:         {r3.score}")
print(f"  Grade:         {r3.grade}")
print(f"  Tipping Point: {r3.critical_x:.1f} °C")
print(f"  CI:            [{r3.ci[0]:.1f}, {r3.ci[1]:.1f}]")
print(f"  Kappa:         {r3.kappa:.2f}")
print(f"  p-value:       {r3.p_value:.4f}")
print(f"  Safety Margin: {r3.safety_margin}")
print(f"  Diagnosis:     {r3.diagnosis}")
print()

pd.DataFrame({"temperature_C": temp_chem, "reaction_rate_mol_s": reaction_rate}).to_csv(
    "d:/code/sweetspot/examples/chemistry_reaction.csv", index=False)

# ---------------------------------------------------------------------------
# 4. Biology — Cell Survival vs Drug Concentration
# ---------------------------------------------------------------------------
print("=" * 70)
print("4. BIOLOGY — Cell Survival vs Drug Concentration")
print("=" * 70)

dose = np.linspace(0, 100, 80)
survival = np.clip(100 - 90 / (1 + np.exp(-0.2 * (dose - 42))) + rng.normal(0, 2, 80), 0, 100)

r4 = analyze(dose, survival, current_x=30,
             x_name="Dose (mg/L)",
             y_name="Cell Survival (%)",
             label="Cytotoxicity Assay")
print(f"  Score:         {r4.score}")
print(f"  Grade:         {r4.grade}")
print(f"  Tipping Point: {r4.critical_x:.1f} mg/L")
print(f"  CI:            [{r4.ci[0]:.1f}, {r4.ci[1]:.1f}]")
print(f"  Kappa:         {r4.kappa:.2f}")
print(f"  p-value:       {r4.p_value:.4f}")
print(f"  Safety Margin: {r4.safety_margin}")
print(f"  Diagnosis:     {r4.diagnosis}")
print()

pd.DataFrame({"dose_mg_L": dose, "cell_survival_pct": survival}).to_csv(
    "d:/code/sweetspot/examples/biology_cytotoxicity.csv", index=False)

# ---------------------------------------------------------------------------
# 5. ML — Gradient Norm vs Learning Rate
# ---------------------------------------------------------------------------
print("=" * 70)
print("5. ML — Gradient Norm vs Learning Rate")
print("=" * 70)

lr = np.linspace(0.001, 0.1, 100)
grad_norm = 1.0 + 50 / (1 + np.exp(-200 * (lr - 0.052))) + rng.normal(0, 0.3, 100)

r5 = analyze(lr, grad_norm, current_x=0.04,
             x_name="Learning Rate",
             y_name="Gradient Norm",
             label="Training Stability")
print(f"  Score:         {r5.score}")
print(f"  Grade:         {r5.grade}")
print(f"  Tipping Point: {r5.critical_x:.4f}")
print(f"  CI:            [{r5.ci[0]:.4f}, {r5.ci[1]:.4f}]")
print(f"  Kappa:         {r5.kappa:.2f}")
print(f"  p-value:       {r5.p_value:.4f}")
print(f"  Safety Margin: {r5.safety_margin}")
print(f"  Diagnosis:     {r5.diagnosis}")
print()

pd.DataFrame({"learning_rate": lr, "gradient_norm": grad_norm}).to_csv(
    "d:/code/sweetspot/examples/ml_gradient.csv", index=False)

# ---------------------------------------------------------------------------
# 6. Infrastructure — Response Latency vs Concurrent Requests
# ---------------------------------------------------------------------------
print("=" * 70)
print("6. INFRASTRUCTURE — Response Latency vs Concurrent Requests")
print("=" * 70)

rq = np.linspace(100, 10000, 100)
latency = 10 + 50 / (1 - np.clip(rq / 7500, 0, 0.98)) + rng.normal(0, 5, 100)

r6 = analyze(rq, latency, current_x=5000,
             x_name="Concurrent Requests",
             y_name="Response Time (ms)",
             label="Server Load Test")
print(f"  Score:         {r6.score}")
print(f"  Grade:         {r6.grade}")
print(f"  Tipping Point: {r6.critical_x:.0f} req/s")
print(f"  CI:            [{r6.ci[0]:.0f}, {r6.ci[1]:.0f}]")
print(f"  Kappa:         {r6.kappa:.2f}")
print(f"  p-value:       {r6.p_value:.4f}")
print(f"  Safety Margin: {r6.safety_margin}")
print(f"  Diagnosis:     {r6.diagnosis}")
print()

pd.DataFrame({"concurrent_requests": rq, "response_time_ms": latency}).to_csv(
    "d:/code/sweetspot/examples/infra_server_load.csv", index=False)

print("=" * 70)
print("ALL TESTS COMPLETE — CSV files saved to examples/")
print("=" * 70)
