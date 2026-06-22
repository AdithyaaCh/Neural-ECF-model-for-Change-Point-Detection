# ECF vs MIDAST: Change Point Detection Benchmark

Comparison of the Empirical Characteristic Function (ECF) method against MIDAST on sub-Gaussian multivariate data.

**MIDAST paper:** "Identifying the Temporal Distribution Structure in Multivariate Data for Time-Series Segmentation Based on Two-Sample Test"

---

## Folder structure

```
.
├── codes/
│   ├── ecf_vs_midast_subgaussian.py      # Sub-Gaussian d=10, MIDAST s=10, ECF stride=5
│   ├── ecf_vs_midast_subgaussian_s1.py   # Sub-Gaussian d=10, MIDAST s=1,  ECF stride=5
│   └── trajectory_utils.py               # stblrnd: alpha-stable sampler (calls R via rpy2)
│
├── common/
│   ├── algorithm12.py                    # Algorithm 1 (window calibration) + k rule-of-thumb
│   └── data_simulators.py                # generate_subgaussian_segment, generate_student_t_segment
│
├── vendored_midast/                      # MIDAST source copied verbatim from MIDAST-1.0.0
│   ├── multivariate_statistical_test_method.py   # ChangeDetector (Algorithm 1/2 + fit/analyze)
│   ├── multivariate_tests_from_R.py              # MMDTest via rpy2
│   ├── ks_2samp.py / ndtest.py                   # KSTest internals
│   └── ...
│
├── midast_runner.py                      # Thin wrapper: run MIDAST on a single series from CLI
│
├── results_subg_d10/                     # Sub-Gaussian d=10, s=10  (35 cells, 500 trials each)
├── results_s1_subg_d10/                  # Sub-Gaussian d=10, s=1   (35 cells, 500 trials each)
└── results_subg_d2/                      # Sub-Gaussian d=2          (35 cells, 500 trials each)
```


## MIDAST implementation details

The vendored code is **unmodified** from MIDAST-1.0.0. 

| Parameter | Value | Source |
|-----------|-------|--------|
| Window `w` | Algorithm 1 (MC power calibration, target=0.9) | Paper §3.1 |
| Segments `k` | `w / (100 × s)` rule of thumb | Paper Appendix A |
| Shift `s` | 1, 5, or 10 depending on script | Paper §4 |
| Decision rule | `based_on="statistic"`, `max_no_changes=1` | Paper §3.2 |
| Test | KSTest (Fasano-Franceschini for d>1) | Paper §2 |

Algorithm 1 calibrates on ρ_pre=0.5 → ρ_post=0.0 (correlation-only shift, no tail change). Results: **w=50** for sub-Gaussian d=10, **w=400** for student-t d=10.

---

## Experimental parameters

### Pre-change (fixed across all trials)

| Distribution | Parameter | Value |
|---|---|---|
| Sub-Gaussian | tail index α₁ | 1.9 |
| Sub-Gaussian | correlation ρ₁ | 0.5 |
| Student-t | degrees of freedom ν₁ | 5.0 |
| Student-t | correlation ρ₁ | 0.5 |

Change point fixed at n* = N/2 = 500 (mid-series). Series length N = 1000.

### Post-change grid (Experiment A)

| Distribution | Axis | Values |
|---|---|---|
| Sub-Gaussian | α₂ | 1.5, 1.7, 1.85, 1.95, 1.98 |
| Sub-Gaussian | ρ₂ | -0.9, -0.6, -0.3, 0.0, 0.3, 0.6, 0.9 |
| Student-t | ν₂ | 2.0, 3.0, 5.0, 8.0, 12.0 |
| Student-t | ρ₂ | -0.9, -0.6, -0.3, 0.0, 0.3, 0.6, 0.9 |

35 cells total (5 × 7), 500 trials per cell (sub-Gaussian) / 30 trials per cell (student-t).

---

## ECF implementation details

The ECF method detects change points by comparing the empirical characteristic functions of two adjacent windows. At each candidate position t, a past window [t−L, t) and a future window [t+gap, t+L+gap) are each summarised into a fixed-length fingerprint vector. The fingerprint is built by projecting the (MAD-standardised) window onto M=256 random frequency directions U ~ N(0,I)/√d at three scales {0.5, 1.0, 2.0}, computing the mean cosine and sine responses at each frequency and scale, and L2-normalising the concatenated result. This gives a 1536-dimensional unit vector that approximates the characteristic function φ(u) = E[e^{iu·X}] evaluated at 768 frequency points. The change-point score at position t is the cosine dissimilarity `1 − z_pre · z_post`: when the two windows come from the same distribution their fingerprints are nearly identical (score ≈ 0), and when the distribution shifts the characteristic functions diverge (score → 1). The score series is smoothed with a length-5 moving average and the change point is extracted from the smoothed series.

**Technical details:**
- **Frequencies:** M=256 directions U ~ N(0,I)/√d, fixed seed=0, drawn once at startup and shared across all trials
- **Scales:** {0.5, 1.0, 2.0} probe low/mid/high frequency bands — fingerprint dimension = 3 × 2 × 256 = 1536
- **Standardisation:** per-dimension MAD standardisation (median + 1.4826×MAD) applied before fingerprinting, making the score robust to heavy tails
- **Window:** L=150, gap=10 between past and future windows
- **Scan stride:** SCAN_STEP=5 (sub-Gaussian) or 10 (student-t); smooth=5

**Change point extraction — argmax vs peaks:**

- `Neural-ECF[argmax]`: returns the position with the single highest score in the smoothed series. Simple and always returns an answer. Works well when there is exactly one change point and the score curve has one dominant peak.

- `Neural-ECF[peaks]`: first finds all local peaks in the score series using prominence-based peak detection (`scipy.signal.find_peaks` with `distance=L//SCAN_STEP`). If no peaks are found it falls back to argmax. If peaks are found it returns the one with the highest prominence (sharpest relative rise above its surroundings), which is more robust to a noisy baseline that might shift the global argmax away from the true change point. For a single change point setting argmax and peaks give identical results in most trials — any difference appears near the boundaries of the parameter grid where the score curve is flatter.

---


## Running an experiment

```bash

WORKERS=8 python3 codes/ecf_vs_midast_subgaussian.py

SMOKE=1 python3 codes/ecf_vs_midast_studentt_s5.py

NO_MMD=1 python3 codes/ecf_vs_midast_subgaussian.py
```

Results are written to the corresponding `results_*/` folder with per-cell checkpointing — safe to kill and resume.
