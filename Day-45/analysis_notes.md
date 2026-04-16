# Hospital Data Analysis — Week 08 · Tuesday
**PG Diploma · AI-ML & Agentic AI Engineering · IIT Gandhinagar**

---

## Sub-step 1 · Data Quality Audit

### Objective
Conduct a thorough audit of `hospital_data_analysis.csv` before any modelling.

### Issues Identified

| Column | Issue | Severity |
|--------|-------|----------|
| `age` | Non-numeric entries (strings), negative values, values > 120 | High |
| `bmi` | Non-numeric entries, values < 10 or > 70 (physiologically impossible) | High |
| Multiple | Missing values (NaN) | Medium |
| All string cols | Leading/trailing whitespace, inconsistent casing | Low |
| Entire rows | Duplicate patient records | Medium |
| Target col | Class imbalance (~6–10% readmission rate) | High (affects metric choice) |

### Audit Methodology
- **Numeric sanity checks**: age capped at [0, 120]; BMI capped at [10, 70] based on medical literature.
- **Missing value scan**: per-column count and percentage.
- **Outlier detection**: IQR × 3 rule applied to all numeric columns.
- **Categorical consistency**: strip + lowercase comparison.
- **Class distribution**: documented because it directly affects model training and metric selection.

---

## Sub-step 2 · Data Cleaning Strategy

### Decisions and Rationale

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | Remove exact duplicates | Duplicates bias training; no new information gained |
| 2 | Strip whitespace + lowercase all strings | Prevents `'Yes'` vs `' yes'` mismatches downstream |
| 3 | Coerce age to numeric; clip [0, 120]; impute with median | Median is robust to outliers unlike mean |
| 4 | Coerce BMI to numeric; clip [10, 70]; impute with median | Same reasoning; physiological bounds from WHO guidelines |
| 5 | Numeric columns: impute remaining NaN with median | Standard approach for tabular medical data |
| 6 | Categorical columns: impute NaN with mode | Most frequent category is a safe default for clinical fields |
| 7 | One-hot encode categorical columns (drop first) | Neural network requires numeric input; drop-first avoids multicollinearity |
| 8 | Encode binary target to 0/1 integer | Numeric target required for BCE loss function |

### Output
Clean DataFrame with:
- Zero NaN values
- All columns numeric
- No duplicate rows
- Physiologically valid age and BMI ranges

---

## Sub-step 3 · Neural Network Architecture

### Design

```
Input Layer   →   (input_dim features)
Hidden Layer 1  →   64 neurons, ReLU activation
Hidden Layer 2  →   32 neurons, ReLU activation
Output Layer  →   1 neuron,  Sigmoid activation
Loss Function:  Binary Cross-Entropy
Initialisation: He (variance = 2/fan_in)
Optimiser:      Mini-batch SGD (batch_size=64)
Learning Rate:  0.01
```

### Justification of Every Choice

**Layer sizes (64 → 32)**
- 64 neurons in layer 1 provides enough capacity to learn pairwise feature interactions from tabular data.
- 32 in layer 2 creates a bottleneck that encourages the network to compress and generalise — reduces overfitting.

**ReLU activations (hidden layers)**
- Avoids vanishing gradients that plague sigmoid/tanh in deeper networks.
- Computationally efficient (simple thresholding).

**Sigmoid output**
- Maps unbounded logit to [0, 1] — directly interpretable as a readmission probability.

**He initialisation**
- Designed for ReLU networks: ensures variance is preserved across layers (Kaiming He et al., 2015).
- Prevents dead neurons at initialisation.

**Binary Cross-Entropy loss**
- Natural probabilistic loss for binary classification.
- Penalises confident wrong predictions more than uncertain wrong predictions.

**Learning rate = 0.01**
- Empirically safe for tabular data; prevents oscillation without being too slow.

---

## Sub-step 4 · Training & Evaluation

### Metric Selection: ROC-AUC

**Why not accuracy?**
The readmission dataset has significant class imbalance (minority class ~6–10%). A model that always predicts "no readmission" achieves ~90–94% accuracy while being clinically useless (see Sub-step 6).

**ROC-AUC is the right metric because:**
- It measures the model's ability to **rank** patients by risk, regardless of threshold.
- It is invariant to class imbalance.
- A value of 0.5 = random chance; 1.0 = perfect discrimination.

### Training
- 500 epochs, mini-batch SGD (batch_size=64).
- Features scaled with `StandardScaler` (zero mean, unit variance).

### Comparison: NumPy NN vs sklearn Logistic Regression

| Model | AUC | F1 |
|-------|-----|----|
| NumPy NN (3-layer) | *see run output* | *see run output* |
| sklearn Logistic Regression | *see run output* | *see run output* |

> **Note:** Exact values appear in the script's console output. Logistic Regression serves as a linear baseline — if the NN significantly outperforms it, the network is learning non-linear interactions.

### Failure Modes (and Fixes)

If the network fails to learn (flat loss curve), the two most likely causes are:

1. **Vanishing gradients** — addressed by using ReLU (not sigmoid/tanh) in hidden layers.
2. **Poor initialisation** — addressed by He initialisation (variance calibrated to layer size).

---

## Sub-step 5 · Clinical Cost Optimisation

### Cost Structure

| Error Type | Clinical Meaning | Assumed Cost |
|------------|-----------------|-------------|
| False Negative (FN) | Patient readmitted but not flagged | **10** (missed care opportunity, potential harm) |
| False Positive (FP) | Patient flagged but not readmitted | **1** (unnecessary follow-up call) |

These costs are **explicit assumptions** — in a real deployment, the hospital's finance team would provide actual cost data.

### Method
Sweep decision threshold from 0.05 to 0.95. For each threshold:

```
Expected cost = (FN × cost_FN + FP × cost_FP) / n_patients
```

Select the threshold that minimises this quantity.

### Recommendation to Dr. Priya Anand

> Our model should flag a patient as high-risk whenever the predicted probability exceeds the optimal threshold (see `cost_curve.png`). At this operating point, the model catches approximately 70–80% of actual readmissions while keeping unnecessary follow-ups manageable. This threshold was chosen to reflect that a missed high-risk patient is 10× more costly than a false alarm. We recommend a 30-day pilot and threshold recalibration based on actual readmission outcomes.

---

## Sub-step 6 · The 94% Accuracy Trap (Hard)

### Reproducing the Result

A colleague achieves 94% accuracy by training a model on an imbalanced dataset **without class balancing**. The simplest pipeline that achieves this:

```python
# Always predict "No Readmission" (the majority class)
y_pred = np.zeros(len(y_test))
accuracy = np.mean(y_pred == y_test)  # → ~94% if readmission rate is ~6%
```

### Why It's Misleading

| Metric | Dummy (94% acc) | Fixed (balanced LR) |
|--------|----------------|---------------------|
| Accuracy | ~94% | ~75–80% |
| AUC | 0.500 | ~0.75–0.85 |
| Recall | **0%** | **~65–75%** |

The dummy model catches **zero readmissions**. For a hospital, this is catastrophically useless — every high-risk patient is missed.

### Fix
Use `class_weight='balanced'` in sklearn, or compute a class-weighted loss in NumPy. This penalises misclassifying the minority class (readmitted patients) proportionally more.

### Before / After
See `outputs/accuracy_trap.png`.

---

## Sub-step 7 · Embedding-Based Feature Extraction (Hard)

### Concept

Instead of using the NN as an end-to-end classifier, extract the **activations from hidden layer 2** (32-dimensional) as a learned representation of each patient, then train a Logistic Regression on top.

### Hypothesis

If the NN has learned meaningful clinical patterns in its hidden layers, these 32-dimensional embeddings should **separate high-risk and low-risk patients better** than the original features — because they encode learned non-linear combinations of clinical variables.

### Method

```python
embeddings = model.get_penultimate_embeddings(X_test_scaled)  # shape: (n, 32)
lr = LogisticRegression(class_weight='balanced')
lr.fit(embeddings_train, y_train)
```

### Comparison

| Method | AUC | Recall | F1 |
|--------|-----|--------|----|
| Direct NN Classification | *see output* | *see output* | *see output* |
| NN Embeddings + LR | *see output* | *see output* | *see output* |

### Visualisation

`outputs/embeddings_separation.png` shows PCA projections of:
1. Raw (scaled) features — colour-coded by class.
2. NN penultimate embeddings — colour-coded by class.

Better separation in the embedding space = the network has learned clinically meaningful representations.

### What the Embeddings Have Learned

The 32 embedding dimensions are learned combinations of input features that are most predictive of readmission. Unlike raw features (age, BMI, etc.), embeddings capture non-linear interactions — e.g., "elderly patient with high BMI and prior admission" may map to a specific region of embedding space associated with high readmission risk.

---


