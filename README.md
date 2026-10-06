# KMT Behavioral Biometric Few-Shot Verification Pipeline

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1qyf04KYzq5t3qAOWpkSuEKIFYYpFjx0j?usp=sharing)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

> **Final-Year B.Tech Capstone Project**  
> Verifying user identity from **keystroke dynamics and mouse trajectories** using a few-shot multimodal Siamese network trained with metric learning.

---

## 📌 Project Overview

This repository implements an end-to-end open-set few-shot behavioral biometric verification pipeline. Instead of relying solely on passwords, the system verifies users based on **how they type and move their mouse**.

* **Dataset:** CyberSignature KMT Dataset ([Mendeley Data, DOI: 10.17632/fnf8b85kr6.1](https://data.mendeley.com/datasets/fnf8b85kr6/1)) — 88 participants, 1,760 total trials.
* **Strict Evaluation Protocol:** 52 train / 16 validation / 20 test users. **Test identities are strictly unseen during training** (zero leakage).
* **Few-Shot Enrollment:** A new user is enrolled using only **$K = 5$ genuine samples** to form a behavioral profile centroid.

```
Keystrokes (Dwell, Flight RP/PP) ──► 1D-CNN + BiGRU ──┐
                                                      ├──► Modality Gate ──► LayerNorm ──► 128-D Embedding
Mouse Trajectory (Δx, Δy, v, a)  ──► 1D-CNN + BiGRU ──┘
```

---

## 🚀 Benchmark Results

Evaluated on **20 strictly unseen test users** with decision thresholds calibrated on validation identities:

| Experiment | Pipeline Details | Test EER | ROC-AUC | Test FAR | Test FRR | Threshold ($\tau^*$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **EXP-01 (Baseline)** | 15 ep, random triplets, unmasked mean pool | **16.00%** | **0.9243** | 33.50% | 6.00% | 0.641 |
| **EXP-02 (Dynamics)** | 35 ep, best-validation checkpoint | **13.00%** | **0.9533** | 14.00% | 11.00% | 0.675 |
| **EXP-03 (Best)** | **Masked pooling, window 96/256, targeted semi-hard mining** | **7.25%** | **0.9759** | **14.50%** | **3.00%** | **0.617** |

### 📈 Enrollment Size Sensitivity ($K$-Sweep on EXP-03)

As enrollment samples increase, the behavioral profile stabilizes rapidly:

| Enrollment ($K$) | Test EER | ROC-AUC | Operational FAR | Operational FRR | Threshold ($\tau^*_V$) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$K = 1$ (One-shot)** | 21.86% | 0.8884 | 21.50% | 22.22% | 0.452 |
| **$K = 3$** | 12.07% | 0.9634 | 21.00% | 1.43% | 0.489 |
| **$K = 5$ (Primary)** | **7.25%** | **0.9759** | **14.50%** | **3.00%** | **0.617** |
| **$K = 7$** | **7.08%** | **0.9789** | **9.00%** | **5.00%** | **0.696** |
| **$K = 9$** | 11.50% | 0.9735 | 13.00% | 10.00% | 0.655 |

---

## 💡 Key Architectural Improvements (EXP-03)

1. **Masked Temporal Pooling:** Evaluates only valid interaction frames and eliminates zero-padding dilution.
2. **Extended Temporal Windows:** Expanded keystroke window to 96 and mouse window to 256 to cover the complete web transaction.
3. **Semi-Hard Targeted-Impostor Mining:** Enforces a 50% quota of targeted impostor trials (`false_data`), penalizing mimics who type the exact same text and cutting baseline error in half.
4. **LayerNorm & Modality Gate:** Replaced batch-dependent BatchNorm with LayerNorm and dynamic softmax weighting between modalities.

---

## 📂 Repository File Structure

| File | Purpose |
| :--- | :--- |
| `train_exp03.py` | **Best Model:** Masked pooling, semi-hard mining, cosine annealing, and evaluation across $K$. |
| `model_v2.py` | Length-aware `MultimodalBiometricEncoderV2` with masked pooling and modality gate. |
| `dataset_v2.py` | Window-extended parser (`KEY_MAX=96`, `MOUSE_MAX=256`) returning true sequence lengths. |
| `train_exp02.py` | Extended training dynamics baseline with best-validation checkpointing. |
| `train.py` | Original 15-epoch baseline training script. |
| `model.py` / `dataset.py` | Original v1 architecture and preprocessing. |
| `requirements.txt` | Python dependency specifications. |

---

## 🛠️ Quickstart

### Option 1: Run in Google Colab (Recommended)
Click the badge above or open the [Colab Notebook](https://colab.research.google.com/drive/1qyf04KYzq5t3qAOWpkSuEKIFYYpFjx0j?usp=sharing) to train on a free GPU with 1 click.

### Option 2: Run Locally

```bash
# 1. Clone repository
git clone https://github.com/sanskriti49/kmt-training-pipeline.git
cd kmt-training-pipeline

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train best model (EXP-03)
python train_exp03.py --data path/to/raw_kmt_dataset --epochs 35 --steps 32 --K 5
```

---

## 📜 Dataset Reference

* **Dataset:** Behaviour Biometrics Dataset (KMT)
* **Authors:** Nonso Nnamoko, Joe Barrowclough, Mark Liptrott, Ioannis Korkontzelos (Edge Hill University)
* **DOI:** [10.17632/fnf8b85kr6.1](https://doi.org/10.17632/fnf8b85kr6.1)
* **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
