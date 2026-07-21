# HSI Dual-Tower LSTM Forecasting Project (Handover Guide)

## 0. 10-Minute Quick Read

This project is a time-series forecasting system for the Hang Seng Index (`^HSI`). Its core model is **DualTowerPatternAwareLSTM**, which jointly predicts multi-horizon returns and future volatility, with scenario backtesting and review workflows included.  
The execution order is `Step0 -> Step1 -> Step1.5 -> Step2 -> Step3 -> Step6 -> Step8 -> Step9`. All critical artifacts are stored in `model_artifacts/` for reproducibility, audit, inference, and operations.

---

## 1. Project Overview

### 1.1 Core Objective and Design Intent

- **Core objective**: jointly model **multi-horizon returns (1/5/10/15/20/25/30 days)** and **future volatility** for `^HSI`, supporting market timing, risk warning, and trend tracking.
- **Design intent**:
  - Reduce short-term noise sensitivity from single-horizon forecasts.
  - Move beyond point forecasts by quantifying uncertainty.
  - Mitigate backtest leakage using time-based splits and scenario-level retraining.

### 1.2 System Architecture (Modules and Relationships)

#### Module A: Data Ingestion and Caching (Step0)
- Data source: primarily `yfinance`, with fallback options; local cache under `yf_cache/`.
- Goal: stable OHLCV and macro/cross-asset data retrieval with reduced repeated downloads.

#### Module B: Factor Engineering (Step1)
- Core function: `detect_trading_patterns(...)`.
- Goal: build candidate factors from technical, volume-price, volatility, macro, and cross-market domains.

#### Module C: Feature Selection (Step1.5)
- Two-stage pipeline: **MI screening -> correlation pruning -> PI refinement**.
- Output: `selected_features*.json` for strict train/inference feature alignment.

#### Module D: Data Preprocessing (Step2)
- Time split, normalization, sequence construction, and target creation (7 return targets + 1 volatility target).
- Output: model-ready tensors and scaler artifacts.

#### Module E: Model Training and HPO (Step3.1/3.2/3.3)
- Model: `DualTowerPatternAwareLSTM` (feature attention + sequence encoder + multi-head outputs).
- Optional HPO with Optuna.
- Output: `best_model.pth`, hyperparameter/config files, and metrics.

#### Module F: Inference and Decision Outputs (Step6/Step8)
- Autoregressive multi-step forecasting using trained artifacts.
- Output: forecast path, quantile bands, decision tables, and textual reports.

#### Module G: Scenario Backtesting and Review (Step9 + review scripts)
- Step9 walk-forward retraining to enforce anti-leakage audit.
- `step8_review.py` and `daily_1d_review.py` for maturity review and hit-rate summaries.

### 1.3 Technology Selection Rationale

- **PyTorch**: flexible for multi-task sequence modeling and dual-tower architecture.
- **Pandas/Numpy**: efficient time-series transformation and rolling computations.
- **Scikit-learn**: mature tooling for scaling, MI, random forest, and PI-based ranking.
- **Optuna**: automated hyperparameter search with lower manual tuning cost.
- **Notebook + scripted execution**: balances research iteration and routine automation.
- **Local cache strategy**: improves runtime stability and reduces external API dependency risk.

### 1.4 Core Value and Use Cases

- **Core value**
  - Joint direction + volatility perspective for return-risk decisions.
  - More credible evaluation through scenario-level retraining backtests.
  - Structured artifacts for clean handover, reproducibility, and audit.
- **Use cases**
  - Daily market timing and position sizing.
  - Short-to-mid trend analysis and risk alerting.
  - Model iteration comparison and post-run review.

---

## 2. Factor Landscape

> This section explains core factor groups by definition, role, source, generation method, parameters, scenarios, and relationships.

### 2.1 Factor Overview

- Factor families: **macro, technical, volume-price, volatility, correlation, and cross-market**.
- Common rolling windows: `5`, `30`, `60`, and `252` trading days.
- Final feature subset is selected and versioned via `selected_features*.json`.

### 2.2 Core Factor List (Representative)

#### A. Macro / Cross-Market Factors

- **`US10Y_Close`**
  - Definition: 10Y US Treasury yield level.
  - Role: captures global rate regime and valuation pressure.
  - Source: external index (`^TNX`).
  - Scenario: medium/long-term trend interpretation.

- **`USDCNH_Close` / `CNH_Change_5D`**
  - Definition: offshore RMB level and short-term change.
  - Role: reflects FX pressure and capital-flow dynamics.
  - Source: `CNH=F`.
  - Scenario: risk-off/risk-on transitions and flow-sensitive periods.

- **`VIX_Quantile`**
  - Definition: VIX percentile in a rolling window.
  - Role: quantifies market stress.
  - Source: `^VIX`.
  - Parameter: typically `252`-day window.
  - Scenario: short-term risk warning and volatility spikes.

- **`ES_SPX_Basis`**
  - Definition: US futures-spot structure spread proxy.
  - Role: reflects sentiment and expectation shifts.
  - Scenario: periods of stronger cross-market coupling.

#### B. Technical / Momentum Factors

- **`BB_Pos`**
  - Definition: normalized relative position within Bollinger Bands.
  - Role: captures overbought/oversold and mean-reversion pressure.
  - Generation: rolling mean/std and band-based normalization.

- **`RSI_Velocity`**
  - Definition: first-order change of RSI.
  - Role: captures momentum turning points.

- **`MACD_Bearish_Cross`**
  - Definition: bearish MACD cross event.
  - Role: trend reversal signal reinforcement.

- **`Returns_ZScore`**
  - Definition: return scaled by volatility context.
  - Role: improves comparability under heteroskedastic conditions.

#### C. Volume-Price / Volatility Factors

- **`Vol_Rel_Short` / `Vol_Rel_5`**
  - Definition: current volume relative to short-window average.
  - Role: detects volume breakout or exhaustion.

- **`Volatility_Cluster`**
  - Definition: volatility clustering intensity.
  - Role: models medium-term path uncertainty.

- **`BB_Width_Compressed`**
  - Definition: compressed Bollinger width state.
  - Role: indicates potential directional breakout setup.

#### D. Correlation Factors

- **`Beta_SPX`**
  - Definition: rolling exposure to SPX.
  - Role: measures external shock transmission strength.

- **`Correlation_VIX`**
  - Definition: rolling correlation with VIX.
  - Role: reflects regime-dependent risk preference shifts.

### 2.3 Factor Acquisition and Generation

- Data retrieval and cache-first loading from `yf_cache/`.
- Missing-value alignment and fallback handling for key fields.
- Factor generation from rolling/statistical/cross-market transformations.
- Feature selection by MI + PI with correlation de-duplication.

### 2.4 Inter-Factor Relationships (Handover Focus)

- **Short horizon (1D/5D)**: sentiment and technical factors dominate (e.g., `VIX_Quantile`, `RSI_Velocity`).
- **Mid horizon (10D/20D)**: volatility structure and correlation factors gain weight.
- **Long horizon (25D/30D)**: macro and FX linkage factors become more influential.
- Factor importance is regime-dependent and should be re-validated via backtest and review.

---

## 3. How to Use the Project

### 3.1 Environment Setup

#### 3.1.1 Recommended Runtime
- Python 3.9+ (team-wide version consistency is recommended).
- Jupyter and `pip` available.

#### 3.1.2 Dependency Installation

```bash
pip install torch pandas numpy scikit-learn yfinance TA-Lib optuna matplotlib seaborn scipy
```

> Note: `TA-Lib` installation varies by OS. Install system-level `ta-lib` first if wheel build fails.

#### 3.1.3 Key Paths and Files
- Main workflow: `LSTM_twotarget_v3.ipynb`
- Review scripts: `step8_review.py`, `daily_1d_review.py`
- Reference analysis: `HSI_Model_Trend_Analysis_Report.md`
- Artifact root: `model_artifacts/`

#### 3.1.4 Common Environment Variables

- **`RUN_AUTO`**: auto mode switch (recommended `1`).
- **`TICKER`**: symbol (e.g., `^HSI`).
- **`START` / `END`**: time range.
- **`INCLUDE_TODAY`**: include latest day or not.
- **`RUN_FS`**: run feature selection.
- **`RUN_HPO`**: run hyperparameter optimization.
- **`VOL_STRATEGY` / `VOL_VALUE`**: manual volume fallback policy and value.

#### 3.1.5 Input Templates (Date Range / Missing Data Fallback)

**A. Date range inputs (highest priority)**
- **`START`**: start date in `YYYY-MM-DD`; at least 5 years is recommended.
- **`END`**: end date; leave empty for latest available data.
- **`INCLUDE_TODAY`**: `Y/N` to control whether the latest day is included.

**B. Missing volume input (common in production)**
- Use **`VOL_STRATEGY=manual`** + **`VOL_VALUE=<numeric>`**.
- When the source does not return latest volume, this fallback prevents pipeline interruption.

**C. Missing volatility input (important handover note)**
- No manual volatility value is required by default; volatility targets are rebuilt from historical price series in-pipeline.
- If volatility fields are missing/empty, handle in this order:
  1. Backfill OHLC market data and rerun Step0/Step1.
  2. Regenerate volatility-related features and targets in Step2.
  3. Validate volatility-related metrics in `run_metrics.json`.
- If manual fallback is temporarily required, use short-window historical volatility estimates and log the source explicitly.

### 3.2 Startup Methods

#### 3.2.1 Interactive Local Run (Development)
1. Open `LSTM_twotarget_v3.ipynb`.
2. Execute cells in step order.
3. Verify newly created run folders under `model_artifacts/`.

#### 3.2.2 Parameter Input Method (Notebook Parameter Cell)

Set the following inputs in the parameter section of `LSTM_twotarget_v3.ipynb`:

- **Fixed historical range replay**
  - `RUN_AUTO=1`
  - `TICKER="^HSI"`
  - `START="2015-01-01"`
  - `END="2024-12-31"`
  - `INCLUDE_TODAY="N"`
  - `VOL_STRATEGY="manual"`
  - `VOL_VALUE="3500000000"`
  - `RUN_FS=0`
  - `RUN_HPO=0`

- **Daily incremental run (empty END)**
  - `RUN_AUTO=1`
  - `TICKER="^HSI"`
  - `START="2006-01-01"`
  - `END=""`
  - `INCLUDE_TODAY="Y"`
  - `VOL_STRATEGY="manual"`
  - `VOL_VALUE="12345678"`
  - `RUN_FS=0`
  - `RUN_HPO=0`

#### 3.2.3 Review Run

```bash
python daily_1d_review.py
```

This command triggers `step8_review.py` and generates maturity-review summary tables.

### 3.3 Core Workflow Usage

#### Function A: Training and Model Artifacts
1. Run Step0 to Step3.
2. Confirm generated files:
   - **`best_model.pth`**
   - **`best_hyperparameters.json`**
   - **`selected_features*.json`**
   - **`run_metrics.json`**

#### Function B: Inference and Decision Outputs
1. Run Step6 and Step8.
2. Inspect forecast path, quantile bands, and proportional decision reports.

#### Function C: Scenario Backtesting and Audit
1. Run Step9 (walk-forward).
2. Compare scenario metrics, focusing on MAE and directional accuracy shifts.

### 3.4 Key Parameters

- **Sequence length**: `SEQUENCE_LENGTH=30`
- **Prediction horizons**: `[1,5,10,15,20,25,30]`
- **Training/early stop**: `epochs=200`, `patience=15`
- **Batch size**: `batch_size=64`
- **Loss structure**: weighted combination of returns, volatility, and direction losses

### 3.5 Common Issues and Fixes

#### Issue 1: Missing feature files (`selected_features*.json not found`)
- Cause: feature selection not executed or path misconfigured.
- Fix:
  - set **`RUN_FS=1`** and regenerate;
  - or point to a verified historical feature file.

#### Issue 2: Dimension mismatch
- Cause: inconsistent feature set between training and inference.
- Fix:
  - strictly align by `selected_features_applied.json`;
  - rerun Step1.5 + Step2 if needed.

#### Issue 3: Data pull failure or missing latest volume
- Cause: external API instability.
- Fix:
  - use cache fallback;
  - set **`VOL_STRATEGY=manual`** with **`VOL_VALUE`**.

#### Issue 4: Missing volatility fields or abnormal volatility output
- Cause: incomplete OHLC data, insufficient rolling window history, or skipped volatility-feature regeneration.
- Fix:
  - verify OHLC completeness (at least Open/High/Low/Close);
  - extend date range (earlier `START`) and rerun Step0 to Step2;
  - re-check logs and metrics to confirm volatility head recovery.

#### Issue 5: Notebook parameters do not take effect
- Cause: parameter cell not re-executed, or step execution order is broken.
- Fix:
  - re-run the parameter cell after editing inputs;
  - then execute the core step cells in order.

### 3.6 Maintenance Recommendations

- Keep each run folder in `model_artifacts/` complete; avoid copying weights alone.
- Any factor-engineering update must be followed by feature-selection and inference-alignment checks.
- Run review scripts regularly to monitor real-world drift and hit-rate changes.

---

## 4. Appendix: Artifact Index

- `model_artifacts/<TICKER>/<DATE>/run_<TIME>/config.json`
- `model_artifacts/<TICKER>/<DATE>/run_<TIME>/best_hyperparameters.json`
- `model_artifacts/<TICKER>/<DATE>/run_<TIME>/run_metrics.json`
- `model_artifacts/<TICKER>/<DATE>/run_<TIME>/selected_features_latest.json`
- `model_artifacts/<TICKER>/<DATE>/run_<TIME>/Proportional_Inference_Report.txt`
- `model_artifacts/Step8_Review_Summary.csv`

For strategy-level interpretation, read this together with `HSI_Model_Trend_Analysis_Report.md`.
