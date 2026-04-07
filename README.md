# Neural-ECF for Change Point Detection

## Overview

This project explores a representation learning approach for multivariate change point detection using a Neural Empirical Characteristic Function (Neural-ECF) model.
Parent paper - "https://www.sciencedirect.com/science/article/pii/S1566253525005184"

The core idea is to learn a transformation that maps time-series windows into a feature space where distributional differences between segments become easier to detect. This learned representation is then used within a sliding-window framework to identify structural changes in the data.

The work focuses on understanding when such learned representations actually help — and where they fail.

---

## Problem Setting

We consider a single change point setting:

- A multivariate time series $X_1, \dots, X_N$
- A true change point at $n^*$
- Distribution shift between:
  - Pre-change segment
  - Post-change segment

The goal is to:

- Detect the change point location  
- Minimize localization error (MAE)  
- Maintain reliable detection (high recall, low false positives)

---

## Base Paper

This project is based on the following paper:

**Identifying the Temporal Distribution Structure in Multivariate Data for Time-Series Segmentation Based on Two-Sample Test**

This work (MIDAST) proposes a two-sample testing framework for detecting change points in multivariate time series by comparing distributions across segments.

It provides:
- A principled statistical framework for change-point detection  
- Simulation setups (Student-t and sub-Gaussian data)  
- Benchmark comparisons against methods such as e-Divisive and KCPA  

In this project, MIDAST serves as the primary reference for:
- Experimental design  
- Evaluation methodology  
- Baseline comparisons  

The Neural-ECF approach is developed and evaluated within this same framework to understand whether learned representations can improve detection performance.

---

## Method: Neural-ECF

### Representation Learning

A neural network is trained using triplet loss:

- Anchor: window from regime A  
- Positive: nearby window from same regime  
- Negative: window from different regime  

- The neural network only learns a frequency matrix and does not correspond to a deep architecture. 

The objective is to enforce:

$$
\text{sim}(A, P) - \text{sim}(A, N) \geq \text{margin}
$$



In practice:

- Margin = 0.4  
- Cosine similarity is used in the learned embedding space  

---

### Detection Pipeline

After training:

1. Slide a window across the time series  

2. Compute embeddings:
   - $z_p(t)$: past window  
   - $z_f(t)$: future window  

3. Compute detection score:

$$
D(t) = 1 - \langle z_p(t), z_f(t) \rangle
$$

4. Peaks in $D(t)$ correspond to candidate change points  

5. A statistical verification step (threshold $\alpha$) is used to control false positives  

---

## Baselines

Following the MIDAST framework, the following methods are implemented for comparison:

- MIDAST (KS-based and MMD-based)  
- e-Divisive (energy distance)  
- KCPA (kernel-based segmentation)  

**Note:**  
e-Divisive and KCPA are implemented as greedy segmentation baselines and are not exact reproductions of the original formulations.

---

## Experimental Setup

### Data

Two distributions are used:

- **Student-t**
  - Heavy-tailed  
  - Controlled via degrees of freedom $\nu$

- **Sub-Gaussian (Alpha-stable)**
  - Controlled via tail parameter $\alpha$

### Key parameters:

- Sample size: $N = 1000$  
- Change point: $n^* \in [0.1N, 0.9N]$  
- Dimensions tested: $d = 2, 10$  
- Correlation shift:
  - $\rho_1 = 0.9$ (pre-change)  
  - $\rho_2 \in [-0.9, 0.9]$  

---

## Evaluation Metrics

- **MAE** (Mean Absolute Error)  
- **Recall**  
- **Runtime**  
- **False Positive Rate (FPR)**  

---

## Key Findings

### 1. High-Dimensional Advantage

- Neural-ECF performs significantly better in $d = 10$ compared to $d = 2$  
- Particularly strong performance on sub-Gaussian data  
- Achieves:
  - Lower MAE  
  - Better recall stability near boundaries  

---

### 2. Representation Limitation

- The model fails to achieve the target triplet margin (0.4)  
- Empirical margin $\approx 0.31$–$0.34$  
- This leads to:
  - Weak separation between regimes  
  - Low contrast in detection scores  

---

### 3. False Positive Issue

- FPR $\approx 20\%$ across all settings  
- False detections occur across the entire time axis (not just boundaries)  

**Implication:**  
Raw scores are not reliable without statistical thresholding  

---

### 4. Window Size Trade-off

- Larger window size $L$:
  - Reduces noise  
  - But blurs regime boundaries  

- Leads to decreasing signal-to-noise ratio  

---

### 5. Training–Testing Mismatch

- Training uses clean triplets with clear separation  
- Testing involves subtle, continuous transitions  

**Result:**  
Learned embedding does not generalize strongly to detection  

---

### 6. Comparison with CNNs

- CNN-based architectures were also tested  
- Observed:
  - Worse performance in low dimensions ($d = 2$)  
  - Less stable representations  

---

## Limitations

- Weak margin realization → insufficient separation  
- High false positive rate without calibration  
- Sensitivity to threshold $\alpha$  
- Dependence on window size $L$  
- Training objective not fully aligned with detection objective  

---

## Conclusion

Neural-ECF shows promise in **high-dimensional settings**, where representation learning begins to provide meaningful gains over classical methods.

However, its effectiveness is currently limited by:

- weak discriminative power  
- calibration issues  
- sensitivity to hyperparameters  

Future improvements should focus on:

- stronger representation separation  
- adaptive thresholding  
- better alignment between training and detection objectives  

---

## Status

This is an experimental research project focused on:

- understanding behavior of learned representations  
- identifying failure modes  
- benchmarking against classical methods  
