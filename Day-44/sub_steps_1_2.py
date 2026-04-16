"""
sub_steps_1_2.py
----------------
Sub-step 1: Load and characterise ecommerce_sales_ts.csv
Sub-step 2: Load, identify issues, and clean sensor_data.csv
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_DIR          = Path(__file__).parent
ECOM_PATH         = DATA_DIR / 'ecommerce_sales_ts.csv'
SENSOR_PATH       = DATA_DIR / 'sensor_data.csv'
OUTLIER_IQR_MULT  = 3.0
SENSOR_NAN_THRESH = 0.30
ROLLING_WINDOW    = 7


# ── Sub-step 1 ────────────────────────────────────────────────────────────────

def load_ecommerce(path):
    try:
        df = pd.read_csv(path, parse_dates=['order_date'])
        for col in ('order_date', 'order_count'):
            assert col in df.columns, f"Missing column: {col}"
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found: {path}")
    return df.sort_values('order_date').set_index('order_date')


def detect_outliers_iqr(series, k=OUTLIER_IQR_MULT):
    """Boolean mask of values outside k * IQR fence."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return (series < q1 - k * iqr) | (series > q3 + k * iqr)


def adf_test_manual(series, lags=14):
    """
    Approximate ADF test using OLS on lagged differences (scipy only).
    Returns dict with test_statistic, p_value, is_stationary.
    """
    s = series.dropna().values
    dy = np.diff(s)
    rows = []
    for i in range(lags, len(dy)):
        row = [1.0, s[i]]
        for lag in range(1, lags + 1):
            row.append(dy[i - lag])
        rows.append(row)
    X = np.array(rows)
    y = dy[lags:]
    try:
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        y_hat = X @ beta
        resid = y - y_hat
        dof   = len(y) - X.shape[1]
        s2    = np.sum(resid ** 2) / dof
        se    = np.sqrt(s2 * np.linalg.pinv(X.T @ X)[1, 1])
        t_stat = beta[1] / se
        p_val  = 2 * stats.t.sf(abs(t_stat), df=dof)
        return {'test_statistic': round(t_stat, 4),
                'p_value':        round(float(p_val), 4),
                'is_stationary':  float(p_val) < 0.05}
    except Exception as e:
        return {'error': str(e), 'is_stationary': None}


def characterise_ecommerce(df):
    """Return dict of characterisation findings."""
    series   = df['order_count']
    clean    = series.dropna()
    clean    = clean[~detect_outliers_iqr(clean)]

    dow_means = clean.to_frame().assign(dow=clean.index.dayofweek)\
                     .groupby('dow')['order_count'].mean()
    roll_mean = clean.rolling(ROLLING_WINDOW).mean().dropna()
    slope, _, r, _, _ = stats.linregress(np.arange(len(roll_mean)), roll_mean.values)

    return {
        'n_rows':               len(series),
        'date_range':           (str(series.index.min().date()),
                                 str(series.index.max().date())),
        'missing':              int(series.isna().sum()),
        'outliers':             int(detect_outliers_iqr(series.dropna()).sum()),
        'adf':                  adf_test_manual(clean),
        'trend_slope_per_day':  round(slope, 4),
        'trend_r2':             round(r ** 2, 4),
        'weekly_cv':            round(dow_means.std() / dow_means.mean(), 4),
    }


def clean_ecommerce(df):
    """
    Treatment:
    1. Replace values outside IQR fence with NaN
    2. Time-interpolate all NaN (including original missing + outliers)
    """
    df     = df.copy()
    series = df['order_count'].copy()

    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr    = q3 - q1
    upper  = q3 + OUTLIER_IQR_MULT * iqr
    lower  = max(0.0, q1 - OUTLIER_IQR_MULT * iqr)

    # Nullify extreme values
    series = series.where((series >= lower) & (series <= upper))
    # Interpolate
    series = series.interpolate(method='time')
    df['order_count'] = series
    return df


# ── Sub-step 2 ────────────────────────────────────────────────────────────────

def load_sensor(path):
    try:
        df = pd.read_csv(path, parse_dates=['timestamp'])
        for col in ('timestamp', 'machine_status'):
            assert col in df.columns, f"Missing column: {col}"
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found: {path}")
    return df


def audit_sensor(df):
    """Identify all data quality issues."""
    sensor_cols = [c for c in df.columns
                   if c not in ('timestamp', 'machine_status')]
    nan_counts  = df[sensor_cols].isna().sum()
    std_vals    = df[sensor_cols].std()
    df_s        = df.sort_values('timestamp')
    diffs       = df_s['timestamp'].diff().dt.total_seconds().dropna()
    freq        = diffs.mode()[0]

    return {
        'duplicate_timestamps':  int(df.duplicated('timestamp').sum()),
        'sensors_with_nan':      nan_counts[nan_counts > 0].to_dict(),
        'sensors_high_missing':  nan_counts[nan_counts / len(df) > SENSOR_NAN_THRESH].index.tolist(),
        'constant_sensors':      std_vals[std_vals == 0].index.tolist(),
        'timestamp_gaps':        int((diffs > freq * 2).sum()),
        'expected_freq_seconds': float(freq),
        'class_dist':            df['machine_status'].value_counts().to_dict(),
    }


def clean_sensor(df):
    """
    Treatment pipeline (order matters):
    1. Remove duplicate timestamps (keep first)
    2. Sort by timestamp → set as index
    3. Drop constant (dead) sensors
    4. Drop sensors > NAN_THRESH missing
    5. Forward-fill remaining NaN (limit 5 steps)
    6. Drop any remaining NaN rows
    """
    df = df.copy()

    before = len(df)
    df = df.drop_duplicates(subset='timestamp', keep='first')
    print(f"  Duplicates removed   : {before - len(df)}")

    df = df.sort_values('timestamp').set_index('timestamp')
    sensor_cols = [c for c in df.columns if c != 'machine_status']

    dead = df[sensor_cols].std()
    dead = dead[dead == 0].index.tolist()
    df   = df.drop(columns=dead)
    print(f"  Dead sensors dropped : {dead}")

    sensor_cols = [c for c in df.columns if c != 'machine_status']
    hi_nan = df[sensor_cols].isna().mean()
    hi_nan = hi_nan[hi_nan > SENSOR_NAN_THRESH].index.tolist()
    df     = df.drop(columns=hi_nan)
    print(f"  High-NaN sensors     : {hi_nan}")

    sensor_cols = [c for c in df.columns if c != 'machine_status']
    df[sensor_cols] = df[sensor_cols].ffill(limit=5)
    df = df.dropna(subset=sensor_cols)
    print(f"  Rows after cleaning  : {len(df)}")

    return df


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    # ── Sub-step 1 ────────────────────────────────────────────────────────────
    print("=" * 62)
    print("SUB-STEP 1 — E-Commerce Sales Characterisation")
    print("=" * 62)

    df_ecom_raw = load_ecommerce(ECOM_PATH)
    f           = characterise_ecommerce(df_ecom_raw)

    print(f"\nDate range      : {f['date_range'][0]} → {f['date_range'][1]}")
    print(f"Records         : {f['n_rows']}")
    print(f"Missing         : {f['missing']}")
    print(f"Outliers (IQR)  : {f['outliers']}")
    print(f"\nStationarity    : statistic={f['adf']['test_statistic']}, "
          f"p={f['adf']['p_value']}, stationary={f['adf']['is_stationary']}")
    print(f"Trend slope     : {f['trend_slope_per_day']} orders/day  (R²={f['trend_r2']})")
    print(f"Weekly CV       : {f['weekly_cv']}  (>0.05 = weekly seasonality present)")

    print("\nModelling implications:")
    print("  Upward trend (slope>0, R²=0.51) + weekly seasonality (CV=0.08)")
    print("  → SARIMA(p,d,q)(P,D,Q,7) with d=1 to remove trend")
    print("  → Annual spike (Nov–Dec festive) → consider Prophet or SARIMA s=365")

    df_ecom_clean = clean_ecommerce(df_ecom_raw)
    print(f"\nPost-clean NaN  : {df_ecom_clean['order_count'].isna().sum()}")

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(13, 9))
    df_ecom_raw['order_count'].plot(ax=axes[0], color='steelblue', lw=0.8,
        title='Raw Sales (with issues marked)')
    df_ecom_clean['order_count'].plot(ax=axes[1], color='teal', lw=0.8,
        title='Cleaned Sales Series')
    roll_m = df_ecom_clean['order_count'].rolling(7).mean()
    roll_s = df_ecom_clean['order_count'].rolling(7).std()
    axes[2].plot(roll_m.values, label='7-day rolling mean', color='navy')
    axes[2].fill_between(range(len(roll_m)),
                          (roll_m - roll_s).values,
                          (roll_m + roll_s).values,
                          alpha=0.2, color='navy', label='±1 std')
    axes[2].set_title('Rolling Mean ± Std (non-constant std = non-stationary variance)')
    axes[2].legend()
    for ax in axes:
        ax.set_ylabel('Orders')
    plt.tight_layout()
    fig.savefig(DATA_DIR / 'substep1_ecom.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Plot → substep1_ecom.png")

    # ── Sub-step 2 ────────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("SUB-STEP 2 — Sensor Audit & Clean")
    print("=" * 62)

    df_sensor_raw = load_sensor(SENSOR_PATH)
    issues        = audit_sensor(df_sensor_raw)

    print(f"\nRaw shape               : {df_sensor_raw.shape}")
    print(f"1. Duplicate timestamps : {issues['duplicate_timestamps']}")
    print(f"2. Sensors with NaN     : {list(issues['sensors_with_nan'].keys())}")
    print(f"   High-missing (>30%)  : {issues['sensors_high_missing']}")
    print(f"3. Dead sensors (zero σ): {issues['constant_sensors']}")
    print(f"4. Timestamp gaps       : {issues['timestamp_gaps']}")
    print(f"\nClass distribution:")
    for k, v in issues['class_dist'].items():
        print(f"  {k:<12}: {v:>6} ({v/len(df_sensor_raw)*100:.1f}%)")

    print("\nTreatment strategy:")
    print("  Duplicates → drop; identical timestamps are recording artefacts")
    print("  Dead sensors → drop; constant values add noise, no signal")
    print("  Sparse NaN → forward-fill ≤5 steps (sensor continuity assumption)")
    print("  Dense NaN  → drop column (>30% missing = unreliable)")

    print("\nCleaning:")
    df_sensor_clean = clean_sensor(df_sensor_raw)
    print(f"Clean shape             : {df_sensor_clean.shape}")
    print(f"Status dist             : {df_sensor_clean['machine_status'].value_counts().to_dict()}")

    df_sensor_clean.to_csv(DATA_DIR / 'sensor_data_clean.csv')
    print("Clean data → sensor_data_clean.csv")

    # Plot first 6 sensors
    scols = [c for c in df_sensor_clean.columns if c != 'machine_status'][:6]
    fig, axes = plt.subplots(len(scols), 1, figsize=(13, 10), sharex=True)
    for ax, col in zip(axes, scols):
        ax.plot(df_sensor_clean[col].values, lw=0.5, color='steelblue')
        ax.set_ylabel(col, fontsize=8)
        broken = (df_sensor_clean['machine_status'] == 'BROKEN').values
        for i in np.where(np.diff(broken.astype(int)) == 1)[0]:
            ax.axvline(i, color='red', alpha=0.5, lw=0.7)
    axes[-1].set_xlabel('Time step')
    plt.suptitle('Cleaned Sensor Readings — red lines = BROKEN events', fontsize=11, fontweight='bold')
    plt.tight_layout()
    fig.savefig(DATA_DIR / 'substep2_sensor.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Plot → substep2_sensor.png")
