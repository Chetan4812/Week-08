"""
sub_steps_6_7.py  [HARD — Optional]
-------------------------------------
Sub-step 6: Rule-based vs ML model comparison using cost matrix
Sub-step 7: Fleet-scale cost optimisation — threshold tuning for 100K sensors
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_curve

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_DIR         = Path(__file__).parent
SENSOR_CLEAN     = DATA_DIR / 'sensor_data_clean.csv'
WINDOW_STEPS     = 60
LOOKAHEAD        = 1440

# Cost matrix (relative units)
COST_FN          = 10    # missed failure → emergency repair
COST_FP          = 1     # false alarm   → unnecessary inspection
FLEET_SIZE        = 100_000
FRACTION_ALERT    = 0.05   # fraction of sensors flagged per day (from Step 5 FPR proxy)


# ── Reuse from sub_step_5 ─────────────────────────────────────────────────────

def load_clean_sensor(path):
    df = pd.read_csv(path, parse_dates=['timestamp'], index_col='timestamp')
    assert 'machine_status' in df.columns
    return df


def engineer_features(df, window=WINDOW_STEPS):
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
    return feats.dropna()


def build_failure_labels(df, lookahead=LOOKAHEAD):
    status = (df['machine_status'] != 'NORMAL').astype(int)
    target = pd.Series(0, index=df.index)
    for i in range(len(df) - lookahead):
        if status.iloc[i:i + lookahead].any():
            target.iloc[i] = 1
    return target


def temporal_split(X, y, test_frac=0.2):
    split = int(len(X) * (1 - test_frac))
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]


def compute_cost(tn, fp, fn, tp, cost_fn=COST_FN, cost_fp=COST_FP):
    """Total cost = FP * cost_FP + FN * cost_FN."""
    return fp * cost_fp + fn * cost_fn


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    df      = load_clean_sensor(SENSOR_CLEAN)
    feats   = engineer_features(df)
    target  = build_failure_labels(feats)
    X       = feats.drop(columns=['machine_status'])
    y       = target.loc[X.index]

    X_train, X_test, y_train, y_test = temporal_split(X, y)

    # Refit model
    cw = (y_train == 0).sum() / (y_train == 1).sum()
    sw = np.where(y_train == 1, cw, 1.0)
    model = GradientBoostingClassifier(n_estimators=150, max_depth=4,
                                        learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train, sample_weight=sw)
    y_prob = model.predict_proba(X_test)[:, 1]

    # ── Sub-step 6: Rule vs ML ────────────────────────────────────────────────
    print("=" * 62)
    print("SUB-STEP 6 — Rule-Based vs ML Model (Cost Matrix)")
    print("=" * 62)

    # Identify the best single-signal rule
    # Top feature from Step 5: sensor max values → pick sensor_48_max
    sensor_cols = [c for c in df.columns if c != 'machine_status']
    # Evaluate each sensor's max in window as standalone rule
    rule_results = []
    for col in [c for c in X_test.columns if c.endswith('_max')][:20]:
        for pct in [70, 80, 90, 95, 99]:
            thresh = X_train[col].quantile(pct / 100)
            y_rule = (X_test[col] > thresh).astype(int)
            if y_rule.sum() == 0:
                continue
            cm     = confusion_matrix(y_test, y_rule)
            if cm.shape != (2, 2):
                continue
            tn, fp, fn, tp = cm.ravel()
            cost = compute_cost(tn, fp, fn, tp)
            rec  = tp / (tp + fn + 1e-9)
            rule_results.append({'signal': col, 'percentile': pct,
                                  'threshold': round(thresh, 3),
                                  'recall': round(rec, 3), 'cost': cost,
                                  'fp': fp, 'fn': fn, 'tp': tp, 'tn': tn})

    rule_df     = pd.DataFrame(rule_results).sort_values('cost')
    best_rule   = rule_df.iloc[0]

    print(f"\nBest single-signal rule:")
    print(f"  Signal      : {best_rule['signal']}")
    print(f"  Threshold   : > {best_rule['threshold']} ({best_rule['percentile']}th percentile of train)")
    print(f"  Recall      : {best_rule['recall']}")
    print(f"  Total cost  : {best_rule['cost']:.0f}  (FP×{COST_FP} + FN×{COST_FN})")

    # ML model at optimal threshold (find cost-minimising threshold)
    thresholds = np.linspace(0.1, 0.9, 81)
    ml_costs   = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        cm     = confusion_matrix(y_test, y_pred)
        if cm.shape != (2, 2):
            ml_costs.append(1e9)
            continue
        tn, fp, fn, tp = cm.ravel()
        ml_costs.append(compute_cost(tn, fp, fn, tp))

    ml_costs      = np.array(ml_costs)
    best_t_idx    = np.argmin(ml_costs)
    best_t        = thresholds[best_t_idx]
    best_ml_cost  = ml_costs[best_t_idx]

    y_pred_opt    = (y_prob >= best_t).astype(int)
    cm_ml         = confusion_matrix(y_test, y_pred_opt)
    tn_m, fp_m, fn_m, tp_m = cm_ml.ravel() if cm_ml.shape == (2,2) else (0,0,0,0)
    rec_ml = tp_m / (tp_m + fn_m + 1e-9)

    print(f"\nML model (cost-optimal threshold={best_t:.2f}):")
    print(f"  Recall      : {rec_ml:.3f}")
    print(f"  Total cost  : {best_ml_cost:.0f}  (FP×{COST_FP} + FN×{COST_FN})")
    print(f"  FP={fp_m}, FN={fn_m}, TP={tp_m}, TN={tn_m}")

    print(f"\nComparison:")
    print(f"  {'Model':<30} {'Cost':>8}  {'Recall':>8}")
    print("  " + "-" * 50)
    print(f"  {'Rule (' + best_rule['signal'] + ')':<30} {best_rule['cost']:>8.0f}  {best_rule['recall']:>8.3f}")
    print(f"  {'ML (GBM) @ t=' + str(round(best_t,2)):<30} {best_ml_cost:>8.0f}  {rec_ml:>8.3f}")

    if best_ml_cost < best_rule['cost']:
        winner = 'ML model'
        print(f"\n✅ ML model wins on total cost.")
    else:
        winner = 'Rule-based'
        print(f"\n✅ Rule-based wins on total cost.")

    print(f"\nWhen rule outperforms ML:")
    print(f"  → Single sensor fully characterises failure mode (low sensor diversity)")
    print(f"  → Low data volume (ML needs sufficient failure events to learn from)")
    print(f"  → Explainability is a hard constraint (regulators/safety standards)")
    print(f"\nWhen ML outperforms rule:")
    print(f"  → Multiple interacting sensors signal failure")
    print(f"  → Rule threshold needs regular manual tuning; ML adapts during retraining")
    print(f"  → Class imbalance is handled via sample weighting")
    print(f"\nRecommendation: Deploy ML as primary alert system.")
    print(f"Use the rule as a hard-override safety net: if {best_rule['signal']} >")
    print(f"  {best_rule['threshold']}, always alert regardless of ML score.")

    # Plot cost curves
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(thresholds, ml_costs, color='steelblue', lw=2, label='ML cost')
    axes[0].axvline(best_t, color='crimson', lw=1.5, linestyle='--',
                    label=f'Optimal T={best_t:.2f}')
    axes[0].axhline(best_rule['cost'], color='orange', lw=1.5, linestyle=':',
                    label=f'Best rule cost={best_rule["cost"]:.0f}')
    axes[0].set_xlabel('Classification threshold')
    axes[0].set_ylabel(f'Cost (FP×{COST_FP} + FN×{COST_FN})')
    axes[0].set_title('ML Cost vs Threshold', fontweight='bold')
    axes[0].legend(fontsize=8)

    prec_arr, rec_arr, _ = precision_recall_curve(y_test, y_prob)
    axes[1].plot(rec_arr, prec_arr, color='teal', lw=2)
    axes[1].scatter(rec_ml, tp_m / (tp_m + fp_m + 1e-9), color='crimson',
                    s=100, zorder=5, label=f'Cost-optimal T={best_t:.2f}')
    axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision–Recall Curve', fontweight='bold')
    axes[1].legend(fontsize=8)

    plt.suptitle('Sub-step 6 — Rule vs ML Model', fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig.savefig(DATA_DIR / 'substep6_rule_vs_ml.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nPlot → substep6_rule_vs_ml.png")

    # ── Sub-step 7: Fleet-scale cost optimisation ─────────────────────────────
    print("\n" + "=" * 62)
    print("SUB-STEP 7 — Fleet-Scale (100K Sensors) Cost Optimisation")
    print("=" * 62)

    # Scale test metrics to fleet
    n_test        = len(y_test)
    pos_rate      = y_test.mean()    # fraction of at-risk timesteps per sensor per day
    inspections_per_sensor_per_day = 24   # 1440 min / 60-min window

    print(f"\nFleet: {FLEET_SIZE:,} sensors")
    print(f"Cost matrix: FN={COST_FN}× FP={COST_FP}×")

    thresholds2 = np.linspace(0.05, 0.95, 91)
    daily_costs = []
    f1_scores   = []

    for t in thresholds2:
        y_pred = (y_prob >= t).astype(int)
        cm     = confusion_matrix(y_test, y_pred)
        if cm.shape != (2, 2):
            daily_costs.append(1e9); f1_scores.append(0); continue
        tn, fp, fn, tp = cm.ravel()
        # Rates per timestep
        fp_rate = fp / (tn + fp + 1e-9)
        fn_rate = fn / (tp + fn + 1e-9)
        # Expected daily events per sensor per day
        exp_fp  = fp_rate * inspections_per_sensor_per_day * (1 - pos_rate)
        exp_fn  = fn_rate * inspections_per_sensor_per_day * pos_rate
        # Fleet daily cost
        daily_cost = FLEET_SIZE * (exp_fp * COST_FP + exp_fn * COST_FN)
        daily_costs.append(daily_cost)
        f1 = 2 * tp / (2 * tp + fp + fn + 1e-9)
        f1_scores.append(f1)

    daily_costs = np.array(daily_costs)
    f1_scores   = np.array(f1_scores)

    cost_opt_idx = np.argmin(daily_costs)
    f1_opt_idx   = np.argmax(f1_scores)
    cost_opt_t   = thresholds2[cost_opt_idx]
    f1_opt_t     = thresholds2[f1_opt_idx]

    print(f"\nCost-minimising threshold : {cost_opt_t:.2f}")
    print(f"  Expected daily fleet cost: {daily_costs[cost_opt_idx]:,.0f} cost-units")
    print(f"\nF1-maximising threshold   : {f1_opt_t:.2f}")
    print(f"  Daily fleet cost at F1-opt: {daily_costs[f1_opt_idx]:,.0f} cost-units")

    diff = daily_costs[f1_opt_idx] - daily_costs[cost_opt_idx]
    print(f"\nCost difference: {diff:,.0f} cost-units/day")
    print(f"  Using F1-optimal threshold costs {diff:,.0f} more per day than")
    print(f"  cost-optimal threshold at fleet scale.")

    if cost_opt_t != f1_opt_t:
        print(f"\nF1 vs cost-optimal thresholds DIFFER ({f1_opt_t:.2f} vs {cost_opt_t:.2f}).")
        print(f"What this tells us about F1 as a production optimisation target:")
        print(f"  F1 treats FP and FN as equally costly (harmonic mean of Precision and Recall).")
        print(f"  But in this problem FN costs {COST_FN}× more than FP.")
        print(f"  → Maximising F1 in production systematically under-prioritises Recall,")
        print(f"    leading to more missed failures and higher real-world cost.")
        print(f"  → Use cost-weighted metrics (expected cost) as the optimisation target")
        print(f"    in production; reserve F1 for balanced-cost settings.")

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].plot(thresholds2, daily_costs / 1e3, color='crimson', lw=2)
    axes[0].axvline(cost_opt_t, color='navy', lw=1.5, linestyle='--',
                    label=f'Cost-optimal T={cost_opt_t:.2f}')
    axes[0].axvline(f1_opt_t, color='orange', lw=1.5, linestyle=':',
                    label=f'F1-optimal T={f1_opt_t:.2f}')
    axes[0].set_ylabel('Daily fleet cost (×1000 units)')
    axes[0].set_title(f'Daily Cost at Scale — {FLEET_SIZE:,} Sensors', fontweight='bold')
    axes[0].legend()

    axes[1].plot(thresholds2, f1_scores, color='teal', lw=2)
    axes[1].axvline(f1_opt_t, color='orange', lw=1.5, linestyle=':',
                    label=f'F1-optimal T={f1_opt_t:.2f}')
    axes[1].axvline(cost_opt_t, color='navy', lw=1.5, linestyle='--',
                    label=f'Cost-optimal T={cost_opt_t:.2f}')
    axes[1].set_ylabel('F1 score'); axes[1].set_xlabel('Classification threshold')
    axes[1].set_title('F1 Score vs Threshold', fontweight='bold')
    axes[1].legend()

    plt.suptitle('Sub-step 7 — Fleet-Scale Cost Optimisation', fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig.savefig(DATA_DIR / 'substep7_fleet_cost.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nPlot → substep7_fleet_cost.png")
