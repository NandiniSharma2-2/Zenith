# Zenith Sky Condition Monitoring — Model Evaluation Report (V2)

## 1. Executive Summary

This report presents the rigorous evaluation of the **Zenith V2 Sky Condition Monitoring Model** against the original baseline architecture on the held-out test dataset from unseen all-sky camera `C009`.

### Benchmark Comparison (Held-Out Test Set — Unseen Camera C009)

| Evaluation Metric | Baseline Model (`zenith_mobilenetv2.keras`) | Zenith V2 (`zenith_v2.keras`) | Delta / Improvement |
| :--- | :---: | :---: | :---: |
| **Nighttime Accuracy** | 26.67% | **38.06%** | **+11.39%** |
| **Nighttime Okta MAE** | 1.29 oktas | **0.98 oktas** | **-0.31 oktas** |
| **Within ±1 Okta Accuracy** | 65.28% | **76.11%** | **+10.83%** |
| **Macro F1-Score (Night)** | 24.47% | **37.07%** | **+12.60%** |
| **Daytime Accuracy (C009)** | 42.50% | **60.56%** | **+18.06%** |

Nighttime test sample counts: baseline **360**, V2 **360**. Per-class support (V2): `{'0': 40, '1': 40, '2': 40, '3': 40, '4': 40, '5': 40, '6': 40, '7': 40, '8': 40}`.

## 2. Key Architectural & Training Enhancements in V2

1. **Domain-Generalizing Data Augmentation**: Injected rotational symmetry, brightness shifts (±40), contrast jitter, and 30% desaturation/grayscale conversion to prevent the CNN from relying on daytime blue-sky chromaticity.
2. **Nighttime Adaptation Set Integration**: Holds out 20% of C005 nighttime images for validation and oversamples the remaining images for training.
3. **Two-Stage Controlled Transfer Learning**: Phase 1 trained a regularized head (GAP + BatchNorm + Dense(256) + L2 + Dropout(0.4)), followed by Phase 2 controlled fine-tuning of the top 35 layers with gentle learning rate (`7e-5`) and early stopping.
4. **Continuous Ordinal Prediction**: Expected Okta calculation ($\sum c \cdot P(c)$) complements discrete classification for fine-grained astronomical windowing.

## 3. Detailed Class-Wise Performance Breakdown (Nighttime Test Set)

| Okta Class | Cloud Cover Category | Support | V2 Precision | V2 Recall | V2 F1-Score | Baseline F1-Score |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **0** | 0 (Completely Clear) | 40 | 0.921 | 0.875 | 0.897 | 0.130 |
| **1** | 1 (Nearly Clear) | 40 | 0.568 | 0.625 | 0.595 | 0.082 |
| **2** | 2 (Scattered) | 40 | 0.368 | 0.175 | 0.237 | 0.315 |
| **3** | 3 (Scattered) | 40 | 0.245 | 0.600 | 0.348 | 0.296 |
| **4** | 4 (Half Clouded) | 40 | 0.185 | 0.125 | 0.149 | 0.093 |
| **5** | 5 (Broken) | 40 | 0.219 | 0.175 | 0.194 | 0.250 |
| **6** | 6 (Broken) | 40 | 0.191 | 0.325 | 0.241 | 0.258 |
| **7** | 7 (Nearly Overcast) | 40 | 0.200 | 0.050 | 0.080 | 0.269 |
| **8** | 8 (Overcast) | 40 | 0.792 | 0.475 | 0.594 | 0.509 |

## 4. Astronomical Suitability & Measurable Image Indicators

- **Image Indicators**: Zenith calculates brightness, contrast, saturation, and Laplacian sharpness.
- **Physical Disclaimers**: Zenith makes explicit that RGB indicators evaluate optical camera conditions, not atmospheric seeing (arcsecond turbulence) or photometer sky background (mag/arcsec²).

## 5. Top-Class Score Reliability

| Nighttime Metric | Baseline | Zenith V2 |
| :--- | :---: | :---: |
| Exact accuracy | 0.267 | 0.381 |
| Mean top-class softmax score | 0.425 | 0.549 |
| Top-class ECE (10 bins; lower is better) | 0.163 | 0.175 |
| Multiclass Brier score (lower is better) | 0.837 | 0.709 |

The top-class softmax score is not accuracy or a calibrated probability. ECE compares mean top-class scores with observed correctness in equal-width confidence bins; the Brier score evaluates the full probability vector.

On this nighttime test set, the V2 mean top-class score exceeds exact accuracy by **16.9 percentage points**, indicating substantial overconfidence on average.

| V2 top-class score range | Images | Mean score | Observed exact accuracy |
| :--- | :---: | :---: | :---: |
| 0.2–0.3 | 8 | 0.277 | 0.125 |
| 0.3–0.4 | 77 | 0.357 | 0.208 |
| 0.4–0.5 | 97 | 0.450 | 0.216 |
| 0.5–0.6 | 60 | 0.548 | 0.367 |
| 0.6–0.7 | 47 | 0.641 | 0.426 |
| 0.7–0.8 | 26 | 0.749 | 0.654 |
| 0.8–0.9 | 14 | 0.851 | 0.643 |
| 0.9–1.0 | 31 | 0.966 | 1.000 |

## 6. Generalization Limits

The V2 nighttime result on unseen camera C009 is **38.06% exact accuracy**, **76.11% within ±1 okta**, and **0.98 okta MAE**. Exact-class accuracy alone understates the ordinal nature of some errors, but the result is not strong evidence of reliable performance on arbitrary cameras.

Some classes remain especially unreliable: nighttime recall is below 10% for okta class(es) 7.

The training sources include 5,400 daytime images from C013 and only 180 nighttime images from C005. The revised pipeline holds out 36 C005 images for checkpoint validation and oversamples the remaining 144; because both sets come from C005, this is not unseen-camera validation. The former C013 validation score also did not estimate cross-camera performance. C009 remains held out for final evaluation. This covers only a small number of camera domains and does not validate ordinary photos or other sensor types. Predictions should be treated as estimates until evaluated on labeled data from the intended camera.
