# IMU-Aided Multi-Class GNSS Spoofing Detection

**Paper:** "IMU-Aided Multi-Class GNSS Spoofing Detection Using Machine Learning: Onset Detection Approach and Detectability Threshold"  
**Author:** Semih Kahraman — Istanbul University-Cerrahpasa  
**Journal:** Journal of Aeronautics and Space Technologies (JAST), under review

---

## Results at a Glance

| Metric | Held-out | External test |
|---|---|---|
| Accuracy | 96.8% | 95.3% |
| macro-F1 | 88.2% | 88.5% |
| Sudden Jump detection | 100% | 100% |
| Replay detection | 100% | 100% |
| False alarm rate | 1.3% | 2.3% |
| ROC AUC | 0.934 | — |

---

## Repository Structure

```
├── 1b_real_data_pipeline.py   # Sensor data loading and merging
├── 1c_realistic_pipeline.py   # Dataset generation (run this first)
├── 2_prepare_data.py          # Feature extraction
├── final_4class.py            # Main training and evaluation
├── detectability_curve.py     # Slow drift detectability analysis
├── lstm_compare.py            # BiLSTM baseline comparison
├── converter_kitti.py         # KITTI raw data conversion
├── imu_kitti_*.csv            # KITTI driving sequence IMU data
├── gps_kitti_*.csv            # KITTI driving sequence GPS data
├── imu_20260519_*.csv         # Istanbul field IMU recordings
├── gps_20260519_*.csv         # Istanbul field GPS recordings
├── checkpoints/
│   └── onset_4class_final.pkl # Pre-trained HGB model
├── data/                      # Generated datasets (see Setup)
├── results/                   # Pre-computed result CSVs
└── paper/
    ├── generate_jast_word.py  # Generates jast_paper.docx
    ├── regen_figures_en.py    # Regenerates all figures
    ├── compute_real_roc.py    # Computes real ROC curve from model
    ├── Submission_Template.docx
    ├── jast_paper.docx        # Final submission document
    ├── jast_paper.pdf         # Final submission PDF
    └── figures/               # Paper figures (PNG)
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Quickstart

### 1. Generate datasets from raw sensor data

```bash
python 1c_realistic_pipeline.py
```

This reads all `imu_kitti_*.csv` / `gps_kitti_*.csv` and Istanbul field files,
applies software-injected spoofing, and writes:
- `data/dataset_realistic_train.csv` (~83 MB, 60 trajectories)
- `data/dataset_realistic_test.csv` (~11 MB, 8 trajectories)

> Takes ~3–5 minutes on a modern CPU.

### 2. Train and evaluate

```bash
python final_4class.py
```

Trains HGB on 75% of trajectories, evaluates on held-out 25% and external test set.
Saves updated model to `checkpoints/onset_4class_final.pkl`.

### 3. Detectability curve

```bash
python detectability_curve.py
```

### 4. BiLSTM comparison (optional, requires PyTorch)

```bash
python lstm_compare.py
```

### 5. Regenerate paper figures

```bash
python paper/regen_figures_en.py
python paper/compute_real_roc.py
```

### 6. Regenerate paper (Word document)

```bash
python paper/generate_jast_word.py
```

Requires `paper/Submission_Template.docx` (included).

---

## Data Sources

- **KITTI Raw Dataset:** A. Geiger et al., "Vision meets robotics: The KITTI dataset," *Int. J. Robotics Research*, 2013. Available at [cvlibs.net/datasets/kitti](http://www.cvlibs.net/datasets/kitti/).
- **Istanbul field data:** Collected with an STM32-based IMU/GPS logger at Istanbul University-Cerrahpasa.

---

## Method Overview

A two-stage framework:
1. **Stage 1 — Onset filter:** HGB classifies each 2 s sliding window; trajectory flagged as attacked if any window in a 1 s buffer is non-Normal.
2. **Stage 2 — Type assignment:** Majority vote over first flagged windows determines attack type.

Key features: IMU–GPS dead-reckoning residual (innovation), GPS jump magnitude, position difference. 100-dimensional feature vector (20 channels × 5 statistics).

---

## Citation

```
S. Kahraman, "IMU-Aided Multi-Class GNSS Spoofing Detection Using Machine Learning:
Onset Detection Approach and Detectability Threshold,"
Journal of Aeronautics and Space Technologies, under review, 2026.
```

---

## Contact

Semih Kahraman — semihkahraman001@gmail.com  
ORCID: [0009-0001-8642-9005](https://orcid.org/0009-0001-8642-9005)
