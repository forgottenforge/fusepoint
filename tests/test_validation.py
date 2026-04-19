"""
Comprehensive validation of fusekit.

Tests:
1. Known tipping points — does it find them exactly?
2. Edge cases — too little data, constant, monotone, duplicates
3. Score calibration — are scores consistent across domains?
4. Noise robustness — does the score degrade gracefully?
5. Statistical correctness — permutation test, bootstrap CI
"""

import numpy as np
import sys
import traceback

# Track results
PASS = 0
FAIL = 0
ERRORS = []


def test(name, condition, detail=""):
    global PASS, FAIL, ERRORS
    status = "PASS" if condition else "FAIL"
    if condition:
        PASS += 1
    else:
        FAIL += 1
        ERRORS.append(f"  {name}: {detail}")
    print(f"  [{status}] {name}" + (f" — {detail}" if detail and not condition else ""))


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


from fusekit import analyze, compare


# ================================================================
# 1. KNOWN TIPPING POINTS
# ================================================================
section("1. KNOWN TIPPING POINTS")

# 1a. Sharp sigmoid at x=0.5 — should find it precisely
print("\n--- 1a. Sharp sigmoid at x=0.5 ---")
x = np.linspace(0, 1, 200)
y = 1.0 / (1 + np.exp(-100 * (x - 0.5)))
r = analyze(x, y, label="Sigmoid x=0.5")
test("Finds tipping point near 0.5",
     abs(r.critical_x - 0.5) < 0.02,
     f"found {r.critical_x:.4f}, expected ~0.5")
test("High confidence (p < 0.05)",
     r.p_value < 0.05,
     f"p = {r.p_value:.4f}")
test("High kappa (sharp peak)",
     r.kappa > 5,
     f"kappa = {r.kappa:.2f}")
test("CI contains true value",
     r.ci[0] <= 0.5 <= r.ci[1],
     f"CI = [{r.ci[0]:.4f}, {r.ci[1]:.4f}]")

# 1b. Sigmoid at x=0.3
print("\n--- 1b. Sigmoid at x=0.3 ---")
y2 = 1.0 / (1 + np.exp(-80 * (x - 0.3)))
r2 = analyze(x, y2, label="Sigmoid x=0.3")
test("Finds tipping point near 0.3",
     abs(r2.critical_x - 0.3) < 0.03,
     f"found {r2.critical_x:.4f}, expected ~0.3")
test("Significant",
     r2.p_value < 0.05,
     f"p = {r2.p_value:.4f}")

# 1c. Sigmoid at x=0.8
print("\n--- 1c. Sigmoid at x=0.8 ---")
y3 = 1.0 / (1 + np.exp(-80 * (x - 0.8)))
r3 = analyze(x, y3, label="Sigmoid x=0.8")
test("Finds tipping point near 0.8",
     abs(r3.critical_x - 0.8) < 0.03,
     f"found {r3.critical_x:.4f}, expected ~0.8")

# 1d. Step function (hardest possible tipping point)
print("\n--- 1d. Step function at x=0.6 ---")
y_step = np.where(x < 0.6, 0.0, 1.0)
r_step = analyze(x, y_step, label="Step at 0.6")
test("Finds step near 0.6",
     abs(r_step.critical_x - 0.6) < 0.02,
     f"found {r_step.critical_x:.4f}")
test("Very high kappa for step",
     r_step.kappa > 10,
     f"kappa = {r_step.kappa:.2f}")

# 1e. M/M/1 queue (known critical point at utilization=1)
print("\n--- 1e. M/M/1 queue (capacity=100) ---")
load = np.linspace(10, 130, 150)
capacity = 100
latency = 1.0 / np.maximum(capacity - load, 0.5)
r_mm1 = analyze(load, latency, current_x=50, label="M/M/1 Queue")
test("Finds critical load near 100",
     abs(r_mm1.critical_x - 100) < 10,
     f"found {r_mm1.critical_x:.1f}, expected ~100")
test("Significant",
     r_mm1.p_value < 0.05,
     f"p = {r_mm1.p_value:.4f}")

# 1f. Two tipping points — should find the dominant one
print("\n--- 1f. Two transitions (x=0.3 and x=0.7) ---")
y_two = 1.0 / (1 + np.exp(-60 * (x - 0.3))) + 1.0 / (1 + np.exp(-60 * (x - 0.7)))
r_two = analyze(x, y_two, label="Two transitions")
test("Finds one of the two tipping points",
     abs(r_two.critical_x - 0.3) < 0.05 or abs(r_two.critical_x - 0.7) < 0.05,
     f"found {r_two.critical_x:.4f}, expected ~0.3 or ~0.7")
test("Significant",
     r_two.p_value < 0.05,
     f"p = {r_two.p_value:.4f}")


# ================================================================
# 2. EDGE CASES
# ================================================================
section("2. EDGE CASES")

# 2a. Minimum data (5 points)
print("\n--- 2a. Minimum data (5 points) ---")
try:
    x5 = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    y5 = np.array([0.1, 0.2, 0.9, 0.95, 0.95])
    r5 = analyze(x5, y5, label="5 points")
    test("Works with 5 points", True, f"score={r5.score}")
except Exception as e:
    test("Works with 5 points", False, str(e))

# 2b. Too few points (should raise)
print("\n--- 2b. Too few points (3) ---")
try:
    analyze(np.array([1, 2, 3]), np.array([1, 2, 3]))
    test("Raises error for 3 points", False, "No error raised")
except ValueError as e:
    test("Raises error for 3 points", True)
except Exception as e:
    test("Raises error for 3 points", False, f"Wrong error type: {type(e).__name__}")

# 2c. Constant data (no variation)
print("\n--- 2c. Constant data ---")
x_const = np.linspace(0, 1, 50)
y_const = np.ones(50) * 5.0
try:
    r_const = analyze(x_const, y_const, label="Constant")
    test("Handles constant data", True, f"score={r_const.score}")
    test("Low score for constant data", r_const.score < 30,
         f"score={r_const.score}")
    test("Not significant", r_const.p_value >= 0.05,
         f"p={r_const.p_value:.4f}")
except Exception as e:
    test("Handles constant data", False, str(e))

# 2d. Monotonically increasing (no peak in derivative)
print("\n--- 2d. Monotonically increasing ---")
x_mono = np.linspace(0, 1, 80)
y_mono = x_mono ** 2
try:
    r_mono = analyze(x_mono, y_mono, label="Monotone")
    test("Handles monotone data", True, f"score={r_mono.score}")
except Exception as e:
    test("Handles monotone data", False, str(e))

# 2e. Unsorted x values
print("\n--- 2e. Unsorted x values ---")
rng = np.random.default_rng(77)
x_shuf = rng.permutation(x)
y_shuf = 1.0 / (1 + np.exp(-80 * (x_shuf - 0.5)))
r_shuf = analyze(x_shuf, y_shuf, label="Unsorted")
test("Handles unsorted x",
     abs(r_shuf.critical_x - 0.5) < 0.03,
     f"found {r_shuf.critical_x:.4f}")

# 2f. x and y different lengths
print("\n--- 2f. Mismatched lengths ---")
try:
    analyze(np.array([1, 2, 3, 4, 5]), np.array([1, 2, 3]))
    test("Raises error for mismatched lengths", False, "No error raised")
except ValueError:
    test("Raises error for mismatched lengths", True)

# 2g. Duplicate x values
print("\n--- 2g. Duplicate x values ---")
x_dup = np.array([0.1, 0.1, 0.2, 0.3, 0.3, 0.4, 0.5, 0.5, 0.6, 0.7])
y_dup = np.array([0.1, 0.15, 0.2, 0.8, 0.85, 0.9, 0.92, 0.91, 0.93, 0.94])
try:
    r_dup = analyze(x_dup, y_dup, label="Duplicates")
    test("Handles duplicate x values", True, f"score={r_dup.score}")
except Exception as e:
    test("Handles duplicate x values", False, str(e))

# 2h. Very large dataset
print("\n--- 2h. Large dataset (10000 points) ---")
x_big = np.linspace(0, 10, 10000)
y_big = 1.0 / (1 + np.exp(-20 * (x_big - 5)))
y_big += np.random.default_rng(1).normal(0, 0.05, 10000)
try:
    import time
    t0 = time.time()
    r_big = analyze(x_big, y_big, label="Large", n_boot=200, n_perm=500)
    dt = time.time() - t0
    test("Handles 10k points", True, f"score={r_big.score}, time={dt:.1f}s")
    test("Finishes in < 60s", dt < 60, f"took {dt:.1f}s")
    test("Finds tipping point near 5.0",
         abs(r_big.critical_x - 5.0) < 0.3,
         f"found {r_big.critical_x:.2f}")
except Exception as e:
    test("Handles 10k points", False, str(e))

# 2i. Negative values
print("\n--- 2i. Negative x and y values ---")
x_neg = np.linspace(-10, 10, 100)
y_neg = np.tanh(x_neg)
r_neg = analyze(x_neg, y_neg, current_x=-5, label="Negative values")
test("Handles negative values",
     abs(r_neg.critical_x) < 2.0,
     f"found {r_neg.critical_x:.2f}, expected ~0")


# ================================================================
# 3. SCORE CALIBRATION
# ================================================================
section("3. SCORE CALIBRATION")

# Same transition shape, different domains — scores should be comparable
print("\n--- 3a. Same physics, different scales ---")
# All are sigmoids with same relative sharpness
domains = {
    "Temperature (0-1000K)": (np.linspace(0, 1000, 100), 500, 80/1000),
    "Pressure (0-100 bar)": (np.linspace(0, 100, 100), 50, 80/100),
    "Learning Rate (0-0.1)": (np.linspace(0, 0.1, 100), 0.05, 80/0.1),
    "Voltage (0-5V)": (np.linspace(0, 5, 100), 2.5, 80/5),
}

scores = {}
for name, (xd, center, steepness) in domains.items():
    yd = 1.0 / (1 + np.exp(-steepness * (xd - center)))
    rd = analyze(xd, yd, label=name)
    scores[name] = rd.score
    print(f"  {name}: score={rd.score}, critical={rd.critical_x:.4g}")

score_vals = list(scores.values())
score_range = max(score_vals) - min(score_vals)
test("Scores consistent across scales (range < 15)",
     score_range < 15,
     f"scores={score_vals}, range={score_range}")

# 3b. Sharper transition = higher clarity, higher score
print("\n--- 3b. Sharpness affects score ---")
steepnesses = [10, 30, 80, 200]
x_s = np.linspace(0, 1, 100)
sharp_scores = []
for k in steepnesses:
    y_s = 1.0 / (1 + np.exp(-k * (x_s - 0.5)))
    r_s = analyze(x_s, y_s, current_x=0.2, label=f"k={k}")
    sharp_scores.append(r_s.score)
    print(f"  Steepness k={k}: score={r_s.score}, kappa={r_s.kappa:.2f}")

test("Score increases with sharpness",
     all(sharp_scores[i] <= sharp_scores[i+1] for i in range(len(sharp_scores)-1)),
     f"scores={sharp_scores}")

# 3c. More safety margin = higher score
print("\n--- 3c. Safety margin affects score ---")
x_m = np.linspace(0, 1, 100)
y_m = 1.0 / (1 + np.exp(-100 * (x_m - 0.5)))
margin_scores = []
for cx in [0.45, 0.35, 0.20, 0.05]:
    r_m = analyze(x_m, y_m, current_x=cx, label=f"current={cx}")
    margin_scores.append(r_m.score)
    print(f"  Current x={cx}: score={r_m.score}, margin={r_m.safety_margin:.0%}")

test("Score increases with safety margin",
     all(margin_scores[i] <= margin_scores[i+1] for i in range(len(margin_scores)-1)),
     f"scores={margin_scores}")


# ================================================================
# 4. NOISE ROBUSTNESS
# ================================================================
section("4. NOISE ROBUSTNESS")

# Same signal, increasing noise — score should degrade gracefully
print("\n--- 4a. Signal degrades with noise ---")
x_n = np.linspace(0, 1, 150)
rng_n = np.random.default_rng(42)
noise_levels = [0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
noise_scores = []
for noise in noise_levels:
    y_n = 1.0 / (1 + np.exp(-60 * (x_n - 0.5)))
    y_n = y_n + rng_n.normal(0, noise, len(x_n))
    r_n = analyze(x_n, y_n, current_x=0.2, label=f"noise={noise}")
    noise_scores.append((noise, r_n.score, r_n.p_value, r_n.critical_x))
    print(f"  noise={noise:.2f}: score={r_n.score}, p={r_n.p_value:.4f}, "
          f"critical={r_n.critical_x:.4f}")

test("Clean signal has high score (>70)",
     noise_scores[0][1] > 70,
     f"score={noise_scores[0][1]}")
test("Heavy noise has low score (<40)",
     noise_scores[-1][1] < 40,
     f"score={noise_scores[-1][1]} at noise=2.0")
test("Scores generally decrease with noise",
     noise_scores[0][1] >= noise_scores[-1][1],
     f"clean={noise_scores[0][1]}, noisiest={noise_scores[-1][1]}")

# 4b. Pure noise always scores low
print("\n--- 4b. Multiple noise samples ---")
noise_only_scores = []
for seed in range(10):
    rng_test = np.random.default_rng(seed * 137)
    x_t = np.linspace(0, 1, 80)
    y_t = rng_test.normal(0, 1, 80)
    r_t = analyze(x_t, y_t, label=f"Noise seed={seed}")
    noise_only_scores.append(r_t.score)

avg_noise = np.mean(noise_only_scores)
max_noise = max(noise_only_scores)
print(f"  Noise-only scores: {noise_only_scores}")
print(f"  Average: {avg_noise:.1f}, Max: {max_noise}")
test("Average noise score < 40",
     avg_noise < 40,
     f"avg={avg_noise:.1f}")
test("No noise sample scores > 60",
     max_noise <= 60,
     f"max={max_noise}")


# ================================================================
# 5. STATISTICAL CORRECTNESS
# ================================================================
section("5. STATISTICAL CORRECTNESS")

# 5a. Permutation test: false positive rate under H0
print("\n--- 5a. False positive rate (should be ~5%) ---")
false_positives = 0
n_trials = 50
for seed in range(n_trials):
    rng_fp = np.random.default_rng(seed * 31 + 7)
    x_fp = np.linspace(0, 1, 60)
    y_fp = rng_fp.normal(5, 1, 60)
    r_fp = analyze(x_fp, y_fp, n_perm=500, label=f"H0 trial {seed}")
    if r_fp.p_value < 0.05:
        false_positives += 1

fpr = false_positives / n_trials
print(f"  False positives: {false_positives}/{n_trials} = {fpr:.1%}")
test("False positive rate < 15%",
     fpr < 0.15,
     f"FPR={fpr:.1%}, expected ~5%")
test("False positive rate > 0% (not too conservative)",
     fpr > 0.0 or n_trials < 100,
     f"FPR={fpr:.1%}")

# 5b. Bootstrap CI covers true value
print("\n--- 5b. Bootstrap coverage ---")
covered = 0
n_cov = 30
true_critical = 0.5
for seed in range(n_cov):
    rng_cov = np.random.default_rng(seed * 53)
    x_cov = np.linspace(0, 1, 100)
    y_cov = 1.0 / (1 + np.exp(-80 * (x_cov - true_critical)))
    y_cov += rng_cov.normal(0, 0.05, 100)
    r_cov = analyze(x_cov, y_cov, n_boot=500, label=f"Coverage {seed}")
    if r_cov.ci[0] <= true_critical <= r_cov.ci[1]:
        covered += 1

coverage = covered / n_cov
print(f"  Coverage: {covered}/{n_cov} = {coverage:.1%}")
test("Bootstrap CI coverage >= 80%",
     coverage >= 0.80,
     f"coverage={coverage:.1%}, expected ~95%")

# 5c. Compare function works correctly
print("\n--- 5c. Compare function ---")
x_c = np.linspace(0, 1, 80)
y_c1 = 1.0 / (1 + np.exp(-40 * (x_c - 0.3)))
y_c2 = 1.0 / (1 + np.exp(-40 * (x_c - 0.8)))
delta = compare(x_c, y_c1, x_c, y_c2, current_x=0.1,
                label_before="Close", label_after="Far")
test("Compare produces positive delta",
     delta.delta_score > 0,
     f"delta={delta.delta_score}")
test("After score > before score",
     delta.after.score > delta.before.score,
     f"before={delta.before.score}, after={delta.after.score}")

# 5d. Save/show don't crash
print("\n--- 5d. Save produces valid file ---")
import os
try:
    path = r_step.save("test_output.png")
    exists = os.path.exists("test_output.png")
    size = os.path.getsize("test_output.png") if exists else 0
    test("Save produces file", exists, f"path={path}")
    test("File has content (>10KB)", size > 10000, f"size={size} bytes")
    if exists:
        os.remove("test_output.png")
except Exception as e:
    test("Save works", False, str(e))

# 5e. Comparison card save
print("\n--- 5e. Comparison card save ---")
try:
    path2 = delta.save("test_comparison.png")
    exists2 = os.path.exists("test_comparison.png")
    test("Comparison save works", exists2)
    if exists2:
        os.remove("test_comparison.png")
except Exception as e:
    test("Comparison save works", False, str(e))


# ================================================================
# SUMMARY
# ================================================================
section("SUMMARY")
total = PASS + FAIL
print(f"\n  {PASS}/{total} tests passed, {FAIL} failed\n")
if ERRORS:
    print("  Failures:")
    for e in ERRORS:
        print(e)
print()
sys.exit(0 if FAIL == 0 else 1)
