# Week 08 · Monday — Time Series & Sensor Modelling

**Scenario:** Riya Shah, Senior Data Scientist at an Indian e-commerce platform, needs daily sales forecasts and equipment failure alerts before Monday's business review.

---

## Folder Structure

```
week-08/monday/
├── generate_datasets.py        ← Run first — generates both CSVs
├── sub_steps_1_2.py            ← Easy: characterise + clean both datasets
├── sub_steps_3_4.py            ← Medium: sales forecasting models + comparison
├── sub_step_5.py               ← Medium: sensor failure risk model (24h)
├── sub_steps_6_7.py            ← Hard: rule vs ML + fleet-scale cost optimisation
├── prompts.md                  ← AI usage log with prompts and critiques
└── README.md
```

---

## How to Run

### Python version
Python 3.10+

### Install dependencies

```bash
pip install numpy pandas scipy scikit-learn matplotlib
```

> **Note:** `statsmodels` and `prophet` are NOT required — all time-series modelling is implemented using `numpy`/`scipy` OLS + Fourier features, which avoids the dependency and is mathematically equivalent.

### Step 1 — Generate datasets

```bash
python generate_datasets.py
```

This creates `ecommerce_sales_ts.csv` (730 daily rows) and `sensor_data.csv` (15,150 rows with injected data quality issues).

If you have the real Kaggle datasets, place them in this directory with the same filenames and skip this step.

### Step 2 — Run sub-steps in order

```bash
python sub_steps_1_2.py     # Sub-steps 1 & 2 — characterisation + cleaning
python sub_steps_3_4.py     # Sub-steps 3 & 4 — sales forecasting
python sub_step_5.py        # Sub-step 5     — failure risk model
python sub_steps_6_7.py     # Sub-steps 6 & 7 — rule vs ML + fleet cost (Hard)
```

Each script is self-contained and can be run independently after `sub_steps_1_2.py` (which generates `sensor_data_clean.csv`).

---

## Expected Output

### Sub-step 1 — E-Commerce Sales
- 730 daily records, 12 NaN, 3 outliers detected and treated
- Trend: +0.69 orders/day (R²=0.51) | Weekly CV=0.08 → seasonality confirmed
- ADF p=0.008 → stationary after trend removal
- Plots: `substep1_ecom.png`

### Sub-step 2 — Sensor Data
- 15,150 → 15,000 rows (150 duplicates removed)
- Dead sensor (`sensor_07`) dropped | NaN in `sensor_15` forward-filled
- Status: NORMAL 96% / RECOVERING 3.2% / BROKEN 0.8%
- Plots: `substep2_sensor.png`

### Sub-step 3 — Baseline Sales Model
| Metric | Value |
|:---|:---|
| MAE | ~207 orders/day |
| RMSE | ~216 |
| MAPE | ~17% |
| Seasonal-naive MAE | ~81 (baseline to beat) |

### Sub-step 4 — Extended Model (+ Annual Seasonality)
| Metric | Value |
|:---|:---|
| MAE | ~50 orders/day |
| Improvement over Step 3 | ~157 orders/day |
| Verdict | Annual component justified |

### Sub-step 5 — Failure Risk Model
| Metric | Value |
|:---|:---|
| AUC-ROC | 0.67 |
| Recall (T=0.35) | 0.67 |
| Primary metric | Recall — missed failure costs 10× false alarm |

### Sub-step 6 — Rule vs ML
| Model | Cost | Recall |
|:---|:---|:---|
| Best single-signal rule | 2,879 | 0.647 |
| ML (GBM, cost-optimal T) | 2,210 | 0.940 |

### Sub-step 7 — Fleet Scale (100K sensors)
- Cost-optimal threshold: **0.15**
- F1-optimal threshold: **0.80**
- Daily cost gap: ~1.8M units/day
- Key insight: F1 ≠ cost-optimal when FN >> FP cost

---

## Key Design Decisions

| Decision | Rationale |
|:---|:---|
| No random train/test split | Time-series data — must respect temporal ordering |
| OLS + Fourier features instead of ARIMA | statsmodels unavailable; Fourier OLS is mathematically equivalent reduced form |
| Recall as primary metric (sensor) | Emergency repair >> planned inspection; FN cost = 10× FP |
| Threshold = 0.35 (Step 5) vs 0.15 (Step 7) | Step 5 balances recall/precision; Step 7 optimises fleet-scale business cost |
| Forward-fill NaN ≤ 5 steps | Sensor continuity is valid assumption for short gaps; longer gaps may span state changes |

---

