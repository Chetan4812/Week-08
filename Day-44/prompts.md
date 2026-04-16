# AI Usage Log
**Week 08 · Monday | Time Series & Sensor Modelling**

---

## Sub-step 1 — Stationarity Testing

**Prompt used:**
> "I need to test stationarity of a time-series in Python without statsmodels (not available). Explain how ADF test works mathematically and show how to implement the regression manually using numpy.linalg.lstsq."

**AI output summary:**
The AI correctly explained that ADF tests stationarity by regressing Δy_t on y_{t-1} and lagged differences, then testing H₀: β = 0 (unit root present = non-stationary). It provided a numpy-based OLS implementation using `lstsq`.

**Critique:**
- ✅ Mathematics was correct — ADF as an OLS regression is standard.
- ✅ The `lstsq` approach is correct for parameter estimation.
- ⚠️ The AI used symmetric t-distribution for p-value which is approximate — exact ADF critical values require Dickey-Fuller tables. I retained the approximation with a note in the code.
- ❌ AI did not handle the edge case where a lagged-difference column is all-zero (constant series). Added `pinv` fallback.

**Changes made:** Added try/except around the OLS block; added note that p-value is approximate (t-distribution, not DF-distribution).

---

## Sub-step 3 — Model Selection Justification

**Prompt used:**
> "Given a time series with upward linear trend (slope 0.69/day, R²=0.51) and weekly seasonality (CoV=0.08 across days of week), which model family should I use? Justify. statsmodels is unavailable — can I implement it with OLS + Fourier features?"

**AI output summary:**
AI recommended SARIMA, then confirmed that OLS with Fourier features (sin/cos at weekly and annual frequencies) is a valid reduced-form approximation that captures the same patterns without requiring `statsmodels`.

**Critique:**
- ✅ Fourier feature approach is correct — it is the foundation of Prophet's seasonal component.
- ✅ AI correctly pointed out that d=1 differencing is needed for trend, which I handled via the linear `t` feature (equivalent for OLS).
- ⚠️ AI suggested using all 3 Fourier harmonics — I reduced to k=1 to avoid overfitting on 730 observations.

**Changes made:** Used k=1 Fourier terms (2 features: sin + cos) instead of AI's k=3 suggestion.

---

## Sub-step 5 — Failure Risk Feature Engineering

**Prompt used:**
> "For a binary sensor-failure classification problem (NORMAL vs at-risk within 24h), what rolling features are most predictive? The sensors have gradual drift before failure. I have 52 sensor channels sampled at 1-minute intervals."

**AI output summary:**
AI suggested rolling mean, std, max, and rate-of-change (first difference) over a sliding window. It also suggested autocorrelation features but noted these would be expensive to compute at 52 × 15K scale.

**Critique:**
- ✅ Mean, std, max, ROC are all sound — they capture drift magnitude and direction.
- ⚠️ AI suggested 30-minute window — I used 60-minute to capture slower sensor drift patterns that precede failure in the data.
- ❌ AI recommended autocorrelation features — I excluded these as they add 52 × ~10 = 520 features and would require much more training data to be useful. The 204 features from mean/std/max/ROC were sufficient.

**Changes made:** Window=60, excluded autocorrelation features. Added `_roc` (rate-of-change) which AI suggested last but is actually the most diagnostic feature for gradual drift.

---

## Sub-step 7 — Fleet-Scale Cost Analysis

**Prompt used:**
> "How do I calculate expected daily business cost of a binary classifier deployed on 100K sensors? FN cost = 10x FP cost. Show how to find the threshold that minimises expected cost and compare it to the F1-optimal threshold."

**AI output summary:**
AI provided the expected cost formula: `E[cost] = fleet_size × (FP_rate × FP_cost + FN_rate × FN_cost)` and correctly noted that F1 treats both error types symmetrically, while cost-optimal thresholds should be set lower (more aggressive) when FN >> FP cost.

**Critique:**
- ✅ Expected cost formula is correct.
- ✅ The observation that F1 and cost-optimal thresholds diverge when costs are asymmetric is accurate — confirmed empirically (F1-opt: 0.80 vs cost-opt: 0.15).
- ⚠️ AI modelled FP_rate and FN_rate as global rates across all timesteps. I refined this to account for class prevalence separately (pos_rate term) to avoid conflating the two error modes.

**Changes made:** Separated `fp_rate × (1 - pos_rate)` and `fn_rate × pos_rate` in the cost calculation for mathematical correctness.
