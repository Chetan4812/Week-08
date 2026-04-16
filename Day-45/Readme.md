# Week 08 - Day 45 - Hospital Data Analysis

---

## What This Assignment Covers

30-day hospital readmission prediction using:
- **Data cleaning** (Sub-steps 1–2): audit and fix a messy real-world clinical dataset
- **NumPy neural network** (Sub-steps 3–4): build a 3-layer NN from scratch with forward + backpropagation
- **Clinical cost optimisation** (Sub-step 5): find the optimal decision threshold for the hospital
- **Accuracy trap** (Sub-step 6, Hard): reproduce and fix a misleading 94% accuracy claim
- **Embedding-based approach** (Sub-step 7, Hard): use NN as feature extractor + LR on top

---

## Folder Structure

```
Week-08/Day-45/
├── hospital_analysis.py     ← Main script (all 7 sub-steps)
├── analysis_notes.md        ← Detailed analytical narrative + AI usage log
├── README.md
├── requirements.txt
├── hospital_data_analysis.csv           
```

---

## How to Run

### 1. Prerequisites

**Python version:** 3.9 or higher

**Install packages:**
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

Or using a requirements file:
```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
python hospital_analysis.py
```

The script will:
1. Download the dataset from GitHub automatically
2. Run all 7 sub-steps sequentially
3. Print results to console
4. Save 4 plots to the `outputs/` folder

### 3. Run individual sub-steps (optional)

Each sub-step is a standalone function. You can call them individually in a REPL or notebook:

```python
from hospital_analysis import load_dataset, audit_data_quality, clean_dataset

df_raw   = load_dataset(DATASET_URL)
issues   = audit_data_quality(df_raw)
df_clean = clean_dataset(df_raw)
```

---

## Package Requirements

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | ≥ 1.23 | Neural network implementation (no PyTorch/TF) |
| `pandas` | ≥ 1.5 | Data loading, cleaning, auditing |
| `matplotlib` | ≥ 3.6 | Plotting training loss, cost curves |
| `seaborn` | ≥ 0.12 | Optional enhanced visualisation |
| `scikit-learn` | ≥ 1.1 | Preprocessing, baselines, metrics |

**requirements.txt:**
```
numpy>=1.23
pandas>=1.5
matplotlib>=3.6
seaborn>=0.12
scikit-learn>=1.1
```

---

## Dataset

**URL:** `https://raw.githubusercontent.com/Chetan4812/Week-08/refs/heads/main/Day-45/hospital_data_analysis.csv`

The script fetches this automatically. No manual download required.

**Expected shape:** ~2,000 rows × multiple columns  
**Target column:** contains "readmit" (automatically detected)

---

## Sub-step Summary

| # | Difficulty | What it does |
|---|-----------|-------------|
| 1 | 🟢 Easy | Load dataset, audit data quality across all columns |
| 2 | 🟢 Easy | Clean dataset: handle NaN, outliers, encoding |
| 3 | 🟡 Medium | Build 3-layer NumPy NN (forward + backprop) |
| 4 | 🟡 Medium | Train NN, evaluate with AUC, compare to sklearn LR |
| 5 | 🟡 Medium | Find optimal clinical threshold using asymmetric cost |
| 6 | 🔴 Hard | Reproduce 94% accuracy trap, show why it fails, fix it |
| 7 | 🔴 Hard | Use NN as feature extractor, compare to direct classification |

---

## Key Design Decisions

### Neural Network Architecture
```
Input → Dense(64, ReLU) → Dense(32, ReLU) → Dense(1, Sigmoid)
Loss: Binary Cross-Entropy
Init: He (pairs with ReLU, prevents vanishing gradients)
Optimiser: Mini-batch SGD (batch=64, lr=0.01)
```

### Why ROC-AUC (not Accuracy)
The dataset has class imbalance (~6–10% readmission rate). A model that always predicts "no readmission" achieves ~94% accuracy but catches zero readmissions. AUC measures true discrimination ability — see Sub-step 6 for a full demonstration.

### Clinical Cost Assumptions (Sub-step 5)
- **False Negative cost = 10** (missed readmission → unplanned emergency, patient harm)
- **False Positive cost = 1** (unnecessary follow-up call → small resource cost)

These are explicit assumptions. Change `cost_fn` and `cost_fp` in `find_optimal_clinical_threshold()` to reflect actual hospital cost data.

---

## Output Plots

| File | What it shows |
|------|--------------|
| `outputs/training_loss.png` | BCE loss across 500 training epochs |
| `outputs/cost_curve.png` | Expected clinical cost vs decision threshold |
| `outputs/accuracy_trap.png` | Before/after comparison: dummy vs balanced classifier |
| `outputs/embeddings_separation.png` | PCA of raw features vs NN embeddings, coloured by class |

---


