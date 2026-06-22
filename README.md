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

Algorithm 1 calibrates on ρ_pre=0.5 → ρ_post=0.0 (correlation-only shift, no tail change). Results: **w=50** for sub-Gaussian d=10.

---

## ECF implementation details

- **Frequencies:** M=256 directions U ~ N(0,I)/√d, fixed seed, drawn once
- **Scales:** multi-scale {0.5, 1.0, 2.0} → fingerprint dimension = 3 × 2 × 256 = 1536
- **Fingerprint:** [cos(X·(sU)^T).mean, sin(X·(sU)^T).mean] per scale, L2-normalised
- **Score:** cosine dissimilarity `1 − z_pre · z_post` between two windows of length L=150
- **Scan:** stride SCAN_STEP=5 (sub-Gaussian) or 10 (student-t), smooth=5

---


## Running an experiment

```bash

WORKERS=8 python3 codes/ecf_vs_midast_subgaussian.py

SMOKE=1 python3 codes/ecf_vs_midast_studentt_s5.py

NO_MMD=1 python3 codes/ecf_vs_midast_subgaussian.py
```

Results are written to the corresponding `results_*/` folder with per-cell checkpointing — safe to kill and resume.
