"""
generate_datasets.py
--------------------
Generates synthetic ecommerce_sales_ts.csv and sensor_data.csv
matching the schema of the Kaggle sources referenced in the assignment.

Run once before any sub-step notebook:
    python generate_datasets.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
OUT = Path(__file__).parent

# ── 1. E-commerce Sales Time Series ──────────────────────────────────────────
# ~730 days of daily order counts (2022-01-01 to 2023-12-31)
# Schema mirrors olistbr/brazilian-ecommerce aggregated to daily sales

dates = pd.date_range('2022-01-01', '2023-12-31', freq='D')
n = len(dates)

# Trend component
trend = np.linspace(800, 1400, n)

# Weekly seasonality (lower Mon/Tue, peak Thu/Fri)
day_of_week = np.array(dates.dayofweek)
weekly = np.where(day_of_week < 2, -80,
         np.where(day_of_week >= 4, 120, 20)).astype(float)

# Annual seasonality (peak Nov–Dec: festive season)
doy = np.array(dates.dayofyear)
annual = 200 * np.sin(2 * np.pi * (doy - 60) / 365)

# Noise
noise = np.random.normal(0, 60, n)

# Intentional data quality issues:
sales = trend + weekly + annual + noise
sales = np.clip(sales, 100, None)

# Issue 1: 12 missing values (NaT/NaN)
missing_idx = np.random.choice(n, 12, replace=False)
sales_with_issues = sales.copy().astype(object)
for i in missing_idx:
    sales_with_issues[i] = np.nan

# Issue 2: 3 extreme outliers (data entry errors)
outlier_idx = np.random.choice(n, 3, replace=False)
for i in outlier_idx:
    sales_with_issues[i] = float(sales[i]) * np.random.choice([10, -1])

df_ecom = pd.DataFrame({
    'order_date':   dates,
    'order_count':  sales_with_issues,
    'revenue_brl':  (sales * np.random.uniform(150, 250, n)).round(2),
})

ecom_path = OUT / 'ecommerce_sales_ts.csv'
df_ecom.to_csv(ecom_path, index=False)
print(f"ecommerce_sales_ts.csv saved → {ecom_path}  ({len(df_ecom)} rows)")

# ── 2. Sensor Data (Pump) ─────────────────────────────────────────────────────
# ~220K rows, 52 sensor columns — mirrors nphantawee/pump-sensor-data
# Sensor columns: sensor_00 … sensor_51
# machine_status: NORMAL / RECOVERING / BROKEN

n_sensor = 15000   # trimmed for tractability; full dataset ~220K
timestamps = pd.date_range('2018-04-01', periods=n_sensor, freq='1min')

sensor_cols = [f'sensor_{i:02d}' for i in range(52)]
base_values = np.random.uniform(10, 100, 52)

# Normal readings
sensor_data = np.random.normal(
    loc=base_values,
    scale=base_values * 0.05,
    size=(n_sensor, 52)
)

# Machine status
status = np.array(['NORMAL'] * n_sensor, dtype=object)

# Inject failure episodes: 4 BROKEN events with RECOVERING lead-up
failure_starts = [2000, 5500, 9000, 12500]
for fs in failure_starts:
    # 120-min RECOVERING ramp
    recover_end = min(fs + 120, n_sensor)
    status[fs:recover_end] = 'RECOVERING'
    # Gradually drift sensor readings
    for t in range(fs, recover_end):
        frac = (t - fs) / 120
        sensor_data[t, :] += base_values * frac * 0.4

    # 30-min BROKEN
    broken_end = min(recover_end + 30, n_sensor)
    status[recover_end:broken_end] = 'BROKEN'
    sensor_data[recover_end:broken_end, :] += base_values * 0.8

# Data quality issues:
# Issue 1: sensor_15 has 400 NaN values (sensor dropout)
nan_idx = np.random.choice(n_sensor, 400, replace=False)
sensor_data[nan_idx, 15] = np.nan

# Issue 2: 150 duplicate timestamp rows
dup_idx = np.random.choice(n_sensor - 1, 150, replace=False)
dup_rows_ts   = timestamps[dup_idx]
dup_rows_data = sensor_data[dup_idx]
dup_rows_stat = status[dup_idx]

df_sensor = pd.DataFrame(sensor_data, columns=sensor_cols)
df_sensor.insert(0, 'timestamp', timestamps)
df_sensor['machine_status'] = status

# Append duplicates
df_dups = pd.DataFrame(dup_rows_data, columns=sensor_cols)
df_dups.insert(0, 'timestamp', dup_rows_ts)
df_dups['machine_status'] = dup_rows_stat

df_sensor = pd.concat([df_sensor, df_dups], ignore_index=True)
# Sort to interleave duplicates naturally
df_sensor = df_sensor.sort_values('timestamp').reset_index(drop=True)

# Issue 3: sensor_07 has constant value (dead sensor)
df_sensor['sensor_07'] = 42.0

sensor_path = OUT / 'sensor_data.csv'
df_sensor.to_csv(sensor_path, index=False)
print(f"sensor_data.csv      saved → {sensor_path}  ({len(df_sensor)} rows)")
print(f"\nDataset summary:")
print(f"  ecommerce: {len(df_ecom)} daily records, {df_ecom['order_count'].isna().sum()} NaN, "
      f"3 outliers")
print(f"  sensor   : {len(df_sensor)} rows, {df_sensor['sensor_15'].isna().sum()} NaN in s15, "
      f"{df_sensor.duplicated('timestamp').sum()} duplicate timestamps, 1 dead sensor (s07)")
