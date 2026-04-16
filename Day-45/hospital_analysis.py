"""
Week 08 · Tuesday Daily Assignment
Hospital Data Analysis — All 7 Sub-steps
PG Diploma · AI-ML & Agentic AI Engineering · IIT Gandhinagar
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, f1_score, recall_score, precision_score
)
from sklearn.impute import SimpleImputer
import warnings
import os

warnings.filterwarnings('ignore')
np.random.seed(42)

DATASET_URL = "https://raw.githubusercontent.com/Chetan4812/Week-08/refs/heads/main/Day-45/hospital_data_analysis.csv"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# SUB-STEP 1 · Data Quality Audit
# ─────────────────────────────────────────────

def load_dataset(url: str) -> pd.DataFrame:
    """Load the hospital dataset from the given URL."""
    try:
        df = pd.read_csv(url)
        print(f"[✓] Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
        return df
    except Exception as e:
        raise RuntimeError(f"[✗] Failed to load dataset: {e}")


def audit_data_quality(df: pd.DataFrame) -> dict:
    """
    Conduct a thorough data quality audit.
    Returns a dictionary of issues found per column.
    """
    print("\n" + "="*60)
    print("SUB-STEP 1 · DATA QUALITY AUDIT")
    print("="*60)

    issues = {}

    # 1a. Shape & dtypes
    print(f"\nShape        : {df.shape}")
    print(f"Dtypes:\n{df.dtypes}\n")

    # 1b. Missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_report = pd.DataFrame({
        "missing_count": missing,
        "missing_pct": missing_pct.round(2)
    }).query("missing_count > 0")
    print("Missing values:\n", missing_report)
    issues["missing"] = missing_report

    # 1c. Age column
    if "age" in df.columns:
        age_col = pd.to_numeric(df["age"], errors="coerce")
        age_issues = {
            "non_numeric": df["age"].isna().sum() - age_col.isna().sum(),
            "negative": (age_col < 0).sum(),
            "unrealistically_high": (age_col > 120).sum(),
            "zero": (age_col == 0).sum(),
        }
        print(f"\nAge column issues  : {age_issues}")
        issues["age"] = age_issues

    # 1d. BMI column
    if "bmi" in df.columns:
        bmi_col = pd.to_numeric(df["bmi"], errors="coerce")
        bmi_issues = {
            "non_numeric": df["bmi"].isna().sum() - bmi_col.isna().sum(),
            "negative": (bmi_col < 0).sum(),
            "unrealistically_high": (bmi_col > 70).sum(),
            "unrealistically_low": (bmi_col < 10).sum(),
        }
        print(f"BMI column issues  : {bmi_issues}")
        issues["bmi"] = bmi_issues

    # 1e. Duplicate rows
    dup_count = df.duplicated().sum()
    print(f"\nDuplicate rows     : {dup_count}")
    issues["duplicates"] = dup_count

    # 1f. Numeric columns — outlier scan (IQR method)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    outlier_summary = {}
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        outliers = ((df[col] < q1 - 3 * iqr) | (df[col] > q3 + 3 * iqr)).sum()
        if outliers > 0:
            outlier_summary[col] = outliers
    print(f"\nExtreme outliers (>3×IQR): {outlier_summary}")
    issues["outliers"] = outlier_summary

    # 1g. Categorical column consistency
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    cat_issues = {}
    for col in cat_cols:
        stripped = df[col].str.strip().str.lower()
        if (stripped != df[col].str.lower()).any():
            cat_issues[col] = "leading/trailing whitespace detected"
    if cat_issues:
        print(f"\nCategorical whitespace issues: {cat_issues}")
    issues["categorical"] = cat_issues

    # 1h. Target column check
    target_candidates = [c for c in df.columns if "readmit" in c.lower()]
    if target_candidates:
        t_col = target_candidates[0]
        dist = df[t_col].value_counts(normalize=True).round(3)
        print(f"\nTarget column '{t_col}' distribution:\n{dist}")
        issues["class_imbalance"] = dist

    print("\n[✓] Audit complete.")
    return issues


# ─────────────────────────────────────────────
# SUB-STEP 2 · Data Cleaning
# ─────────────────────────────────────────────

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply a principled cleaning strategy based on the audit findings.
    Every decision is documented inline.
    """
    print("\n" + "="*60)
    print("SUB-STEP 2 · DATA CLEANING")
    print("="*60)

    df = df.copy()

    # Decision 1: Remove exact duplicate rows (no information gained).
    before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"[1] Removed {before - len(df)} duplicate rows.")

    # Decision 2: Strip whitespace and lowercase all string columns for
    #             consistency (avoids 'Yes' vs ' yes' mismatches).
    obj_cols = df.select_dtypes(include="object").columns
    for col in obj_cols:
        df[col] = df[col].str.strip().str.lower()
    print(f"[2] Normalised {len(obj_cols)} string columns (strip + lowercase).")

    # Decision 3: Coerce age to numeric; replace non-numeric with NaN.
    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
        # Clip physiologically implausible ages (< 0 or > 120) to NaN.
        df.loc[(df["age"] < 0) | (df["age"] > 120), "age"] = np.nan
        age_median = df["age"].median()
        df["age"].fillna(age_median, inplace=True)
        print(f"[3] Age: coerced, clipped (0–120), imputed missing with median={age_median:.1f}.")

    # Decision 4: Coerce bmi to numeric; replace implausible values with NaN.
    if "bmi" in df.columns:
        df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
        df.loc[(df["bmi"] < 10) | (df["bmi"] > 70), "bmi"] = np.nan
        bmi_median = df["bmi"].median()
        df["bmi"].fillna(bmi_median, inplace=True)
        print(f"[4] BMI: coerced, clipped (10–70), imputed missing with median={bmi_median:.1f}.")

    # Decision 5: For remaining numeric columns, impute missing values with
    #             median (robust to outliers unlike mean).
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            med = df[col].median()
            df[col].fillna(med, inplace=True)
            print(f"[5] Numeric '{col}': imputed {df[col].isnull().sum()} NaN → median={med:.3f}.")

    # Decision 6: For categorical columns, impute missing with mode.
    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            print(f"[6] Categorical '{col}': imputed NaN → mode='{mode_val}'.")

    # Decision 7: One-hot encode remaining object columns (drop first to avoid multicollinearity).
    cat_cols_after = df.select_dtypes(include="object").columns.tolist()
    # Preserve target if string-encoded
    target_col = next((c for c in df.columns if "readmit" in c.lower()), None)
    encode_cols = [c for c in cat_cols_after if c != target_col]
    if encode_cols:
        df = pd.get_dummies(df, columns=encode_cols, drop_first=True)
        print(f"[7] One-hot encoded: {encode_cols}. New shape: {df.shape}.")

    # Decision 8: Encode binary target to int (0/1) if it's still string.
    if target_col and df[target_col].dtype == object:
        positive_label = df[target_col].unique()[0]
        df[target_col] = (df[target_col] == positive_label).astype(int)
        print(f"[8] Target '{target_col}' encoded — positive label: '{positive_label}'.")

    print(f"\n[✓] Clean DataFrame shape: {df.shape}")
    print(f"    Remaining NaNs: {df.isnull().sum().sum()}")
    return df


# ─────────────────────────────────────────────
# Neural Network (NumPy only)
# ─────────────────────────────────────────────

def sigmoid(z: np.ndarray) -> np.ndarray:
    """Sigmoid activation — numerically stable."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def sigmoid_derivative(a: np.ndarray) -> np.ndarray:
    return a * (1.0 - a)


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)


def relu_derivative(z: np.ndarray) -> np.ndarray:
    return (z > 0).astype(float)


def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Binary cross-entropy loss."""
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


class ThreeLayerNN:
    """
    3-layer neural network (2 hidden + 1 output) built entirely in NumPy.

    Architecture
    ─────────────
      Input  → Hidden-1 (ReLU) → Hidden-2 (ReLU) → Output (Sigmoid)

    Rationale for choices
    ─────────────────────
    • ReLU hidden activations: avoids vanishing gradients in deeper networks.
    • Sigmoid output + BCE loss: natural for binary classification.
    • He initialisation: pairs with ReLU to preserve variance across layers.
    • Learning rate 0.01: conservative default; reduces oscillation on tabular data.
    """

    def __init__(
        self,
        input_dim: int,
        hidden1: int = 64,
        hidden2: int = 32,
        learning_rate: float = 0.01,
    ):
        self.lr = learning_rate
        # He initialisation
        self.W1 = np.random.randn(input_dim, hidden1) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, hidden1))
        self.W2 = np.random.randn(hidden1, hidden2) * np.sqrt(2.0 / hidden1)
        self.b2 = np.zeros((1, hidden2))
        self.W3 = np.random.randn(hidden2, 1) * np.sqrt(2.0 / hidden2)
        self.b3 = np.zeros((1, 1))
        self.loss_history: list = []

    # ── Forward pass ──────────────────────────────────────
    def forward(self, X: np.ndarray) -> np.ndarray:
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = relu(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = relu(self.Z2)
        self.Z3 = self.A2 @ self.W3 + self.b3
        self.A3 = sigmoid(self.Z3)
        return self.A3

    # ── Backward pass (backpropagation) ───────────────────
    def backward(self, X: np.ndarray, y: np.ndarray) -> None:
        m = X.shape[0]
        y = y.reshape(-1, 1)

        # Output layer gradient (BCE + Sigmoid combined)
        dZ3 = self.A3 - y                        # shape (m, 1)
        dW3 = (self.A2.T @ dZ3) / m
        db3 = np.sum(dZ3, axis=0, keepdims=True) / m

        # Hidden layer 2
        dA2 = dZ3 @ self.W3.T
        dZ2 = dA2 * relu_derivative(self.Z2)
        dW2 = (self.A1.T @ dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        # Hidden layer 1
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * relu_derivative(self.Z1)
        dW1 = (X.T @ dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # Gradient descent update
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    # ── Training loop ────────────────────────────────────
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 500,
        batch_size: int = 64,
        verbose: bool = True,
    ) -> None:
        m = X.shape[0]
        for epoch in range(1, epochs + 1):
            # Mini-batch SGD
            indices = np.random.permutation(m)
            X_shuf, y_shuf = X[indices], y[indices]
            for start in range(0, m, batch_size):
                Xb = X_shuf[start : start + batch_size]
                yb = y_shuf[start : start + batch_size]
                self.forward(Xb)
                self.backward(Xb, yb)

            # Epoch loss on full training set
            y_pred = self.forward(X).flatten()
            loss = binary_cross_entropy(y, y_pred)
            self.loss_history.append(loss)

            if verbose and epoch % 50 == 0:
                print(f"  Epoch {epoch:>4d} | Loss: {loss:.4f}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X).flatten()

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def get_penultimate_embeddings(self, X: np.ndarray) -> np.ndarray:
        """Return activations from hidden layer 2 (penultimate layer)."""
        self.forward(X)
        return self.A2.copy()


# ─────────────────────────────────────────────
# SUB-STEP 3 · Build the NN
# ─────────────────────────────────────────────

def prepare_features(df: pd.DataFrame):
    """Split cleaned DataFrame into X (features) and y (target)."""
    target_col = next((c for c in df.columns if "readmit" in c.lower()), None)
    if target_col is None:
        raise ValueError("No readmission target column found.")
    y = df[target_col].values.astype(float)
    X = df.drop(columns=[target_col]).values.astype(float)
    return X, y, target_col


def build_and_describe_nn(input_dim: int) -> ThreeLayerNN:
    """
    Instantiate the 3-layer NN and print architecture details.

    Architecture decisions
    ──────────────────────
    • Hidden-1: 64 neurons — enough capacity for feature interactions.
    • Hidden-2: 32 neurons — bottleneck encourages compression (generalisation).
    • ReLU hidden + Sigmoid output — standard for binary tabular classification.
    • Learning rate 0.01 — prevents overshooting on a small dataset.
    """
    print("\n" + "="*60)
    print("SUB-STEP 3 · NEURAL NETWORK ARCHITECTURE")
    print("="*60)
    model = ThreeLayerNN(input_dim=input_dim, hidden1=64, hidden2=32, learning_rate=0.01)
    print(f"  Input dim  : {input_dim}")
    print(f"  Hidden-1   : 64 neurons (ReLU)")
    print(f"  Hidden-2   : 32 neurons (ReLU)")
    print(f"  Output     : 1 neuron  (Sigmoid)")
    print(f"  Loss       : Binary Cross-Entropy")
    print(f"  Init       : He (pairs with ReLU)")
    print(f"  LR         : 0.01")
    return model


# ─────────────────────────────────────────────
# SUB-STEP 4 · Train & Evaluate
# ─────────────────────────────────────────────

def train_and_evaluate(
    model: ThreeLayerNN,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Train the NN, evaluate with ROC-AUC (appropriate for imbalanced data), compare with sklearn LR."""

    print("\n" + "="*60)
    print("SUB-STEP 4 · TRAINING & EVALUATION")
    print("="*60)
    print("\nClass distribution in training set:")
    unique, counts = np.unique(y_train, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  Class {int(u)}: {c} samples ({c/len(y_train)*100:.1f}%)")

    print("\n[Metric choice] ROC-AUC is used because:")
    print("  • The dataset has class imbalance (readmission is a minority event).")
    print("  • Accuracy would be misleading — a model predicting all-negatives")
    print("    could score ~90%+ without ever catching a readmission.")
    print("  • AUC measures ranking quality across all thresholds.\n")

    # ── Scale features ────────────────────────────────────
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ── Train NN ──────────────────────────────────────────
    print("Training NumPy NN (500 epochs, mini-batch SGD)...")
    model.fit(X_train_s, y_train, epochs=500, batch_size=64, verbose=True)

    # ── NN predictions ────────────────────────────────────
    y_prob_nn = model.predict_proba(X_test_s)
    y_pred_nn = model.predict(X_test_s)

    nn_auc = roc_auc_score(y_test, y_prob_nn)
    nn_f1  = f1_score(y_test, y_pred_nn)
    print(f"\nNumPy NN  →  AUC={nn_auc:.4f}  |  F1={nn_f1:.4f}")
    print(classification_report(y_test, y_pred_nn, target_names=["No Readmit", "Readmit"]))

    # ── sklearn Logistic Regression baseline ──────────────
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_s, y_train)
    y_prob_lr = lr.predict_proba(X_test_s)[:, 1]
    y_pred_lr = lr.predict(X_test_s)

    lr_auc = roc_auc_score(y_test, y_prob_lr)
    lr_f1  = f1_score(y_test, y_pred_lr)
    print(f"sklearn LR →  AUC={lr_auc:.4f}  |  F1={lr_f1:.4f}")
    print(classification_report(y_test, y_pred_lr, target_names=["No Readmit", "Readmit"]))

    # ── Plot training loss curve ───────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(model.loss_history, color="#E76F51", linewidth=1.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Binary Cross-Entropy Loss")
    ax.set_title("Training Loss Curve — NumPy NN")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "training_loss.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n[✓] Training loss plot saved → {path}")

    return {
        "scaler": scaler,
        "nn_auc": nn_auc,
        "nn_f1": nn_f1,
        "lr_auc": lr_auc,
        "lr_f1": lr_f1,
        "y_prob_nn": y_prob_nn,
        "y_pred_nn": y_pred_nn,
    }


# ─────────────────────────────────────────────
# SUB-STEP 5 · Clinical Cost Optimisation
# ─────────────────────────────────────────────

def find_optimal_clinical_threshold(
    y_test: np.ndarray,
    y_prob: np.ndarray,
    cost_fn: float = 10.0,   # Cost of a missed high-risk patient (False Negative)
    cost_fp: float = 1.0,    # Cost of a false alarm (False Positive)
) -> float:
    """
    Find the probability threshold that minimises expected clinical cost.

    Cost assumptions (explicit)
    ─────────────────────────────
    • FN cost = 10  — missing a readmission → unplanned emergency, far more expensive
      and harmful (patient harm, penalty from payer).
    • FP cost = 1   — unnecessary follow-up call or monitoring → small resource cost.
    """
    print("\n" + "="*60)
    print("SUB-STEP 5 · CLINICAL COST OPTIMISATION")
    print("="*60)
    print(f"\nCost assumptions: FN={cost_fn}  |  FP={cost_fp}")

    thresholds = np.linspace(0.05, 0.95, 200)
    costs = []
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        cm = confusion_matrix(y_test, preds)
        tn, fp, fn, tp = cm.ravel()
        expected_cost = (fn * cost_fn + fp * cost_fp) / len(y_test)
        costs.append(expected_cost)

    best_idx = np.argmin(costs)
    best_threshold = thresholds[best_idx]
    best_cost = costs[best_idx]

    # Metrics at optimal threshold
    y_opt = (y_prob >= best_threshold).astype(int)
    recall_opt    = recall_score(y_test, y_opt)
    precision_opt = precision_score(y_test, y_opt, zero_division=0)
    f1_opt        = f1_score(y_test, y_opt)

    print(f"\nOptimal threshold : {best_threshold:.2f}")
    print(f"Expected cost/pt  : {best_cost:.4f}")
    print(f"Recall at opt.    : {recall_opt:.3f}")
    print(f"Precision at opt. : {precision_opt:.3f}")
    print(f"F1 at opt.        : {f1_opt:.3f}")

    # ── Plot cost curve ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(thresholds, costs, color="#264653", linewidth=1.8)
    ax.axvline(best_threshold, color="#E9C46A", linestyle="--", label=f"Optimal={best_threshold:.2f}")
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Expected Cost per Patient")
    ax.set_title("Clinical Cost vs Decision Threshold")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "cost_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[✓] Cost curve plot saved → {path}")

    # ── Plain-language recommendation ─────────────────────
    print("\n" + "─"*60)
    print("RECOMMENDATION TO DR. PRIYA ANAND")
    print("─"*60)
    print(
        f"\nOur model should flag a patient as high-risk for 30-day readmission whenever "
        f"its predicted probability exceeds {best_threshold:.0%}.\n"
        f"\nAt this threshold:\n"
        f"  • The model catches approximately {recall_opt*100:.0f}% of patients who "
        f"will actually be readmitted (recall).\n"
        f"  • About {precision_opt*100:.0f}% of flagged patients truly are at high risk "
        f"(precision).\n"
        f"\nThis setting reflects that a missed high-risk patient (false negative) is "
        f"{int(cost_fn)}× more costly than an unnecessary follow-up (false positive). "
        f"Lowering the threshold further would catch even more true positives but would "
        f"significantly increase the care team's workload. We recommend piloting this "
        f"threshold for 30 days and recalibrating once real-world feedback is available."
    )
    return best_threshold


# ─────────────────────────────────────────────
# SUB-STEP 6 · Reproducing the "94% accuracy" trap
# ─────────────────────────────────────────────

def accuracy_trap_demonstration(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    scaler: StandardScaler,
) -> None:
    """
    Reproduce a pipeline that plausibly achieves 94% accuracy on an imbalanced dataset
    and demonstrate why accuracy is a misleading metric here.
    """
    print("\n" + "="*60)
    print("SUB-STEP 6 · THE 94% ACCURACY TRAP (Hard)")
    print("="*60)

    X_train_s = scaler.transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ── Reproduce: all-negative dummy classifier ──────────
    # On highly imbalanced data (e.g. 6% readmission rate),
    # always predicting 0 gives ~94% accuracy trivially.
    y_dummy = np.zeros_like(y_test)
    dummy_acc = np.mean(y_dummy == y_test)
    dummy_auc = 0.5  # no discrimination ability
    dummy_recall = recall_score(y_test, y_dummy, zero_division=0)
    cm_dummy = confusion_matrix(y_test, y_dummy)

    print("\n[Reproduced pipeline] Always predict 'No Readmission':")
    print(f"  Accuracy : {dummy_acc*100:.1f}%   ← looks great!")
    print(f"  AUC      : {dummy_auc:.3f}        ← random chance")
    print(f"  Recall   : {dummy_recall*100:.0f}%  ← catches ZERO readmissions")
    print("\nConfusion matrix (dummy all-zeros):")
    print(pd.DataFrame(cm_dummy,
        index=["Actual 0","Actual 1"],
        columns=["Pred 0","Pred 1"]))

    # ── Fixed pipeline: LR with class_weight='balanced' ───
    lr_fixed = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr_fixed.fit(X_train_s, y_train)
    y_fixed = lr_fixed.predict(X_test_s)
    y_fixed_prob = lr_fixed.predict_proba(X_test_s)[:, 1]

    fixed_acc    = np.mean(y_fixed == y_test)
    fixed_auc    = roc_auc_score(y_test, y_fixed_prob)
    fixed_recall = recall_score(y_test, y_fixed)
    cm_fixed     = confusion_matrix(y_test, y_fixed)

    print(f"\n[Fixed pipeline] LR with class_weight='balanced':")
    print(f"  Accuracy : {fixed_acc*100:.1f}%   ← lower, but honest")
    print(f"  AUC      : {fixed_auc:.3f}        ← real discrimination")
    print(f"  Recall   : {fixed_recall*100:.0f}%  ← catches readmissions")
    print("\nConfusion matrix (fixed):")
    print(pd.DataFrame(cm_fixed,
        index=["Actual 0","Actual 1"],
        columns=["Pred 0","Pred 1"]))

    # ── Before / After visual ──────────────────────────────
    metrics = ["Accuracy", "AUC", "Recall"]
    dummy_vals = [dummy_acc, dummy_auc, dummy_recall]
    fixed_vals = [fixed_acc, fixed_auc, fixed_recall]

    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width/2, dummy_vals, width, label="Dummy (94% acc trap)", color="#E76F51")
    ax.bar(x + width/2, fixed_vals, width, label="Fixed (balanced LR)",   color="#2A9D8F")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Before vs After: Fixing the 94% Accuracy Trap")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "accuracy_trap.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n[✓] Before/after comparison saved → {path}")

    print("\nConclusion: 94% accuracy is achieved by a model that NEVER predicts a readmission.")
    print("ROC-AUC and Recall are the metrics that actually matter for this problem.")


# ─────────────────────────────────────────────
# SUB-STEP 7 · Embedding-based approach
# ─────────────────────────────────────────────

def embedding_approach(
    model: ThreeLayerNN,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    scaler: StandardScaler,
) -> None:
    """
    Use the trained NN as a feature extractor (penultimate layer activations),
    then train a Logistic Regression on top.
    Compare against direct NN classification.
    """
    print("\n" + "="*60)
    print("SUB-STEP 7 · EMBEDDING-BASED APPROACH (Hard)")
    print("="*60)

    X_train_s = scaler.transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # Extract embeddings from hidden layer 2
    emb_train = model.get_penultimate_embeddings(X_train_s)
    emb_test  = model.get_penultimate_embeddings(X_test_s)
    print(f"\nEmbedding shape (train): {emb_train.shape}")

    # Train a simple LR on top of embeddings
    lr_emb = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    lr_emb.fit(emb_train, y_train)
    y_emb_prob = lr_emb.predict_proba(emb_test)[:, 1]
    y_emb_pred = lr_emb.predict(emb_test)

    emb_auc    = roc_auc_score(y_test, y_emb_prob)
    emb_recall = recall_score(y_test, y_emb_pred)
    emb_f1     = f1_score(y_test, y_emb_pred)

    # Direct NN performance for comparison
    y_nn_prob = model.predict_proba(X_test_s)
    y_nn_pred = model.predict(X_test_s)
    nn_auc    = roc_auc_score(y_test, y_nn_prob)
    nn_recall = recall_score(y_test, y_nn_pred)
    nn_f1     = f1_score(y_test, y_nn_pred)

    print("\nPerformance Comparison:")
    print(f"{'Method':<30} {'AUC':>8} {'Recall':>8} {'F1':>8}")
    print("-"*54)
    print(f"{'Direct NN Classification':<30} {nn_auc:>8.4f} {nn_recall:>8.4f} {nn_f1:>8.4f}")
    print(f"{'NN Embeddings + LR':<30} {emb_auc:>8.4f} {emb_recall:>8.4f} {emb_f1:>8.4f}")

    # ── t-SNE-style scatter using first 2 PCs of embeddings ─
    # Using PCA instead of t-SNE for speed (same concept — visualise separation)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=42)
    emb_2d = pca.fit_transform(emb_test)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (data_2d, title) in zip(axes, [
        (emb_2d, "NN Embeddings (PCA 2D)"),
        (pca.transform(PCA(n_components=2, random_state=42).fit(X_test_s).transform(X_test_s)
                       if False else emb_2d), ""),  # placeholder
    ]):
        pass  # handled below

    # Embeddings scatter
    ax = axes[0]
    colors = np.where(y_test == 1, "#E76F51", "#264653")
    ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=colors, alpha=0.5, s=15)
    ax.set_title("NN Penultimate Layer Embeddings (PCA)")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#E76F51", label="Readmitted"),
        Patch(color="#264653", label="Not Readmitted")
    ])

    # Raw features scatter
    ax = axes[1]
    pca_raw = PCA(n_components=2, random_state=42)
    raw_2d = pca_raw.fit_transform(X_test_s)
    ax.scatter(raw_2d[:, 0], raw_2d[:, 1], c=colors, alpha=0.5, s=15)
    ax.set_title("Raw Features (PCA)")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend(handles=[
        Patch(color="#E76F51", label="Readmitted"),
        Patch(color="#264653", label="Not Readmitted")
    ])

    plt.suptitle("Patient Separation: Embeddings vs Raw Features", fontsize=13)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "embeddings_separation.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n[✓] Embedding separation plot saved → {path}")

    print("\nInterpretation:")
    print("  The penultimate layer activations are a learned 32-dimensional embedding.")
    print("  If the NN has learned meaningful clinical patterns, these embeddings should")
    print("  show better class separation than raw (even scaled) features.")
    print("  The PCA scatter above visualises this separation in 2D.")
    if emb_auc > nn_auc:
        print(f"\n  ✅ Embedding + LR (AUC={emb_auc:.4f}) OUTPERFORMS direct NN (AUC={nn_auc:.4f}).")
        print("     The NN's intermediate representations provide richer signal for a linear separator.")
    else:
        print(f"\n  ℹ️  Direct NN (AUC={nn_auc:.4f}) matches or exceeds Embedding+LR (AUC={emb_auc:.4f}).")
        print("     The end-to-end NN already exploits the embedding implicitly.")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("="*60)
    print("HOSPITAL DATA ANALYSIS — Week 08 Tuesday")
    print("="*60)

    # Sub-step 1
    df_raw = load_dataset(DATASET_URL)
    audit_data_quality(df_raw)

    # Sub-step 2
    df_clean = clean_dataset(df_raw)

    # Sub-step 3
    X, y, target_col = prepare_features(df_clean)
    model = build_and_describe_nn(input_dim=X.shape[1])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Sub-step 4
    results = train_and_evaluate(model, X_train, X_test, y_train, y_test)

    # Sub-step 5
    find_optimal_clinical_threshold(y_test, results["y_prob_nn"])

    # Sub-step 6 (Hard)
    accuracy_trap_demonstration(X_train, X_test, y_train, y_test, results["scaler"])

    # Sub-step 7 (Hard)
    embedding_approach(model, X_train, X_test, y_train, y_test, results["scaler"])

    print("\n" + "="*60)
    print("[✓] All 7 sub-steps complete. Check the 'outputs/' folder for plots.")
    print("="*60)


if __name__ == "__main__":
    main()
