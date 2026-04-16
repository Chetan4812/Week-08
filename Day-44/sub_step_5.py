"""
sub_step_5.py
-------------
Sub-step 5: Equipment failure risk model using cleaned sensor data.
Predicts BROKEN/RECOVERING vs NORMAL for next 24-hour window.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_curve, roc_auc_score)

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_DIR       = Path(__file__).parent
SENSOR_CLEAN   = DATA_DIR / 'sensor_data_clean.csv'
WINDOW_STEPS   = 60    # 60-min rolling window for feature aggregation (1 hr)
LOOKAHEAD      = 1440  # predict failure within next 1440 min = 24 hours
FAILURE_THRESH = 0.35  # probability threshold (lower = higher recall)


# ── Feature Engineering ───────────────────────────────────────────────────────

def load_clean_sensor(path):
    """Load cleaned sensor data."""
    try:
        df = pd.read_csv(path, parse_dates=['timestamp'], index_col='timestamp')
        assert 'machine_status' in df.columns, "Missing: machine_status"
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {path}")
    return df


def engineer_features(df, window=WINDOW_STEPS):
    """
    Rolling statistical features over a sliding window:
    mean, std, max, rate-of-change (diff) for each sensor.
    Rationale: failure events are preceded by gradual sensor drift —
    capturing rolling stats detects this drift before threshold breach.
    """
    sensor_cols = [c for c in df.columns if c != 'machine_status']
    feat_frames = []

    for col in sensor_cols:
        s = df[col]
        feat_frames.append(s.rolling(window).mean().rename(f'{col}_mean'))
        feat_frames.append(s.rolling(window).std().rename(f'{col}_std'))
        feat_frames.append(s.rolling(window).max().rename(f'{col}_max'))
        feat_frames.append(s.diff(window).rename(f'{col}_roc'))

    feats = pd.concat(feat_frames, axis=1)
    feats['machine_status'] = df['machine_status']
    feats = feats.dropna()
    return feats


def build_failure_labels(df, lookahead=LOOKAHEAD):
    """
    Target: will the machine enter BROKEN/RECOVERING within `lookahead` steps?
    This transforms the problem into binary classification with positive = 'at-risk'.
    """
    status = (df['machine_status'] != 'NORMAL').astype(int)
    # For each step, check if ANY future step within lookahead is at-risk
    target = pd.Series(0, index=df.index)
    for i in range(len(df) - lookahead):
        if status.iloc[i:i + lookahead].any():
            target.iloc[i] = 1
    return target


def temporal_train_test_split(X, y, test_frac=0.2):
    """Respect temporal ordering — no shuffling."""
    split = int(len(X) * (1 - test_frac))
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]


def evaluate_failure_model(model, X_test, y_test, threshold=FAILURE_THRESH):
    """
    Compute metrics relevant to maintenance planning.
    Primary metric: Recall (minimise missed failures — cost 10× false alarms).
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    return {
        'threshold':  threshold,
        'AUC_ROC':    round(roc_auc_score(y_test, y_prob), 4),
        'Recall':     round(tp / (tp + fn + 1e-9), 4),
        'Precision':  round(tp / (tp + fp + 1e-9), 4),
        'F1':         round(2 * tp / (2 * tp + fp + fn + 1e-9), 4),
        'FN':         int(fn),
        'FP':         int(fp),
        'TP':         int(tp),
        'TN':         int(tn),
    }, y_prob


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    print("=" * 62)
    print("SUB-STEP 5 — Equipment Failure Risk (24h Ahead)")
    print("=" * 62)

    df = load_clean_sensor(SENSOR_CLEAN)
    print(f"\nLoaded: {df.shape}  Status: {df['machine_status'].value_counts().to_dict()}")

    # Engineer features
    print("\nEngineering rolling features (window=60 min)...")
    feats   = engineer_features(df, window=WINDOW_STEPS)
    target  = build_failure_labels(feats, lookahead=LOOKAHEAD)

    X = feats.drop(columns=['machine_status'])
    y = target.loc[X.index]

    print(f"Feature matrix  : {X.shape}")
    print(f"Positive class  : {y.sum()} / {len(y)}  ({y.mean()*100:.1f}% at-risk)")

    X_train, X_test, y_train, y_test = temporal_train_test_split(X, y)
    print(f"Train/test split: {len(X_train)} / {len(X_test)}  (temporal, no shuffle)")

    # Fit Gradient Boosting — chosen for:
    # 1. Captures non-linear sensor interactions (gradual drift patterns)
    # 2. Built-in feature importance for maintenance team interpretation
    # 3. class_weight equivalent via sample_weight for imbalanced classes
    print("\nFitting GradientBoostingClassifier...")
    class_ratio = (y_train == 0).sum() / (y_train == 1).sum()
    sample_weights = np.where(y_train == 1, class_ratio, 1.0)

    model = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
    )
    model.fit(X_train, y_train, sample_weight=sample_weights)

    metrics, y_prob = evaluate_failure_model(model, X_test, y_test, FAILURE_THRESH)

    print(f"\nMetrics at threshold={FAILURE_THRESH}:")
    for k, v in metrics.items():
        print(f"  {k:<12}: {v}")

    print(f"\nWhy Recall is primary metric:")
    print(f"  A missed failure (FN={metrics['FN']}) triggers emergency repair.")
    print(f"  A false alarm (FP={metrics['FP']}) triggers a planned inspection.")
    print(f"  Emergency repair >> planned inspection in cost and downtime.")
    print(f"  Threshold lowered to {FAILURE_THRESH} (vs default 0.5) to maximise Recall.")

    # Feature importance — top 10
    feat_importance = pd.Series(
        model.feature_importances_, index=X.columns
    ).sort_values(ascending=False)

    print(f"\nTop 10 predictive features:")
    print(feat_importance.head(10).to_string())

    # Maintenance team output: risk score for each sensor timestamp
    risk_df = pd.DataFrame({
        'timestamp':    X_test.index,
        'risk_score':   y_prob.round(3),
        'alert':        (y_prob >= FAILURE_THRESH).astype(int),
        'actual_label': y_test.values,
    }).set_index('timestamp')

    out_path = DATA_DIR / 'failure_risk_scores.csv'
    risk_df.to_csv(out_path)
    print(f"\nRisk scores → {out_path}")
    print("Maintenance team interpretation:")
    print("  risk_score ≥ 0.35 → ALERT: schedule inspection within 24h")
    print("  risk_score < 0.35 → NORMAL: next scheduled inspection applies")

    # Plots
    fig, axes = plt.subplots(2, 1, figsize=(13, 8))

    # Risk score over time (test period)
    axes[0].plot(range(len(y_prob)), y_prob, color='steelblue', lw=0.7, label='Risk score')
    axes[0].axhline(FAILURE_THRESH, color='crimson', lw=1.5, linestyle='--',
                    label=f'Alert threshold={FAILURE_THRESH}')
    # Mark actual failures
    fail_idx = np.where(y_test.values == 1)[0]
    axes[0].scatter(fail_idx, y_prob[fail_idx], color='red', s=8, zorder=5, label='Actual at-risk')
    axes[0].set_title('Failure Risk Score — Test Period', fontweight='bold')
    axes[0].set_ylabel('Risk score (P(failure))')
    axes[0].legend(fontsize=8)

    # Precision-Recall curve
    prec, rec, thresholds = precision_recall_curve(y_test, y_prob)
    axes[1].plot(rec, prec, color='teal', lw=2)
    axes[1].scatter(
        metrics['Recall'], metrics['Precision'],
        color='crimson', s=100, zorder=5,
        label=f"Operating point (T={FAILURE_THRESH})\nRecall={metrics['Recall']:.2f}, Prec={metrics['Precision']:.2f}"
    )
    axes[1].set_title('Precision–Recall Curve', fontweight='bold')
    axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')
    axes[1].legend(fontsize=8)

    plt.suptitle('Sub-step 5 — Equipment Failure Risk Model', fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig.savefig(DATA_DIR / 'substep5_failure_risk.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Plot → substep5_failure_risk.png")

    # Save model metrics for sub-steps 6 & 7
    pd.Series(metrics).to_csv(DATA_DIR / 'substep5_metrics.csv')
    pd.Series({'y_prob_mean': float(y_prob.mean()),
               'n_test': len(y_test),
               'n_positive': int(y_test.sum()),
               'threshold': FAILURE_THRESH}).to_csv(DATA_DIR / 'substep5_meta.csv')
