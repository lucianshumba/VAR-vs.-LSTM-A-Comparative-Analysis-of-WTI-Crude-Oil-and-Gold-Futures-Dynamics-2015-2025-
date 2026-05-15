# VAR vs. LSTM: A Comparative Analysis of WTI Crude Oil and Gold Futures Dynamics (2015–2025)

> A rigorous empirical comparison of classical econometric and machine learning approaches to multivariate commodity time series forecasting. Empirical comparison of VAR(11) and LSTM models for forecasting WTI Crude Oil &amp; Gold Futures log-returns (2015–2025). Includes stationarity testing, Granger causality analysis, impulse response functions, and 60-day out-of-sample evaluation. Both models achieve comparable RMSE/MAE; LSTM yields lower MAPE.

---

## Overview

This project investigates whether machine learning methods outperform classical econometric models in forecasting short-horizon commodity return dynamics — and at what cost to interpretability.

Using **2,394 daily observations** of WTI Crude Oil and Gold Futures prices (January 2015 to December 2025), we implement and compare:

- **VAR(11)** — Vector Autoregression, selected via AIC
- **LSTM** — Stacked two-layer Long Short-Term Memory neural network

Both models are evaluated on an identical **60-day out-of-sample window**, covering a period that includes the 2015–16 oil collapse, the 2020 COVID-19 crash, the 2021–22 commodity supercycle, and the 2022–25 normalisation.

---

## Key Findings

| Metric | VAR Oil | LSTM Oil | VAR Gold | LSTM Gold |
|--------|---------|----------|----------|-----------|
| RMSE | 0.01643 | 0.01647 | 0.01724 | **0.01723** |
| MAE | 0.01312 | 0.01318 | **0.01241** | 0.01239 |
| MAPE | 102.2% | **96.8%** | 138.3% | **101.7%** |

- RMSE and MAE differences are **below 0.5%** — both models are competitive benchmarks
- LSTM achieves **5.4 pp lower MAPE on Oil** and **36.6 pp lower MAPE on Gold**
- VAR identifies **Granger causality from Gold to Oil** at lags 8–11 (p < 0.005) — Gold leads Oil by approximately 8–11 trading days, reflecting macroeconomic sentiment transmission
- Oil does **not** Granger-cause Gold at any lag

---

## Project Structure

```
├── data/
│   ├── DCOILWTICO.csv                   # WTI Oil daily prices (FRED)
│   └── Gold_Futures_Historical_Data.csv # Gold Futures daily prices (Investing.com)
│
├── figures/
│   ├── fig1_eda_prices_returns.png      # Price levels, returns, scatter, rolling correlation
│   ├── fig2_distributions_acf.png       # Histograms, QQ plots, ACF, PACF
│   ├── fig3_irf.png                     # VAR(11) Impulse Response Functions
│   ├── fig4_var_forecast.png            # VAR out-of-sample forecast
│   ├── fig5_lstm_forecast.png           # LSTM out-of-sample forecast
│   └── fig6_lstm_diagnostics.png        # LSTM training curve & residuals
│
├── figures_code.py                      # Reproduces all 6 figures end-to-end
├── VAR-vs-LSTM-Report.docx              # Full technical report (4–7 pages)
└── README.md
```

---

## Methodology

### Data Preparation
- Prices merged on common trading dates; 2,394 observations retained
- April 20, 2020 negative WTI price (−$36.98/bbl) forward-filled (COVID-19 storage anomaly)
- Log-returns computed: `r_t = ln(P_t / P_{t-1})`
- Both series confirmed **I(1)** via ADF and KPSS tests
- Engle-Granger cointegration test: p = 0.417 → **not cointegrated** → VAR on returns appropriate

### VAR Model
- Bivariate system of Oil and Gold log-returns
- Lag order p = 11 selected by AIC
- Estimated by OLS on training data
- Outputs: coefficient estimates, Granger causality tests, Impulse Response Functions (IRFs)

### LSTM Model
- 20-day lookback window (consistent with VAR lag structure)
- Architecture: LSTM(64) → Dropout(0.2) → LSTM(32) → Dropout(0.2) → Dense(16, ReLU) → Dense(2)
- Trained with Adam (lr = 0.001), MSE loss
- Early stopping (patience = 15) — converged at epoch 16
- Learning rate reduction on plateau (factor = 0.5)
- Inputs standardised via `StandardScaler`

---

## Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib statsmodels scipy scikit-learn tensorflow
```

### Data

Download the two data files and place them in the project root (or `data/` folder):

| File | Source | Series |
|------|--------|--------|
| `DCOILWTICO.csv` | [FRED](https://fred.stlouisfed.org/series/DCOILWTICO) | WTI Crude Oil Daily Price |
| `Gold_Futures_Historical_Data.csv` | [Investing.com](https://www.investing.com/commodities/gold-historical-data) | Gold Futures Daily Close |

Set the date range to **January 1, 2015 – December 31, 2025** when downloading.

> **Note:** The Gold CSV from Investing.com uses `MM/DD/YYYY` date format and comma-formatted prices (e.g. `1,850.30`). The script handles this automatically.

### Run

```bash
python figures_code.py
```

This will:
1. Load and clean both data files
2. Compute log-returns and apply the train/test split
3. Fit the VAR(11) model and generate IRFs and forecasts
4. Train the LSTM model (runs ~16 epochs with early stopping)
5. Save all 6 figures to the working directory

---

## Figures

### Figure 1 — Exploratory Data Analysis
Price levels, log-returns, cross-sectional return scatter, and 60-day rolling correlation between WTI Oil and Gold (2015–2025).

### Figure 2 — Distribution Diagnostics
Return histograms (with normal overlay), QQ plots, ACF, and PACF for both series. Oil exhibits extreme kurtosis (89.1) driven by the COVID-19 crash; Gold is considerably more symmetric (kurtosis 3.4).

### Figure 3 — VAR(11) Impulse Response Functions
2×2 IRF grid showing how each series responds to a one-standard-deviation shock in the other. Gold-to-Oil cross-IRF reveals a mild delayed negative response consistent with a safe-haven transmission channel.

### Figure 4 — VAR Out-of-Sample Forecast
VAR(11) 60-day ahead forecast vs. actual log-returns for both series.

### Figure 5 — LSTM Out-of-Sample Forecast
LSTM 60-day ahead forecast vs. actual log-returns for both series on the identical test window.

### Figure 6 — LSTM Diagnostics
Training/validation loss curve confirming convergence without overfitting; residual bar plots for both series showing no systematic bias.

---

## Results Summary

### Granger Causality

| Direction | Lag 8 | Lag 9 | Lag 10 | Lag 11 |
|-----------|-------|-------|--------|--------|
| Gold → Oil (p-value) | 0.004\*\* | 0.002\*\* | 0.002\*\* | 0.000\*\*\* |
| Oil → Gold (p-value) | > 0.15 | > 0.22 | > 0.25 | > 0.15 |

\*\* p < 0.01, \*\*\* p < 0.001

Gold unidirectionally Granger-causes Oil, suggesting that gold market sentiment — reflecting inflation expectations, dollar dynamics, and global risk appetite — transmits to oil prices with an approximately 8–11 trading day lag.

### VAR Residual Diagnostics
| Test | Statistic | Conclusion |
|------|-----------|------------|
| Durbin-Watson (Oil) | 2.005 | No autocorrelation |
| Durbin-Watson (Gold) | 2.002 | No autocorrelation |
| Whiteness Test p-value | 0.014 | Mild residual structure |
| Normality (JB) p-value | 0.000 | Non-normal (expected: fat tails) |

---

## Economic Interpretation

**VAR strengths:** Full structural interpretability — coefficients, Granger causality, and IRFs provide direct economic insight. Preferred for policy analysis, risk management, and regulatory reporting.

**LSTM strengths:** Implicitly captures volatility clustering, nonlinear regime dynamics, and asymmetric responses that linear VAR cannot model by construction. Yields superior MAPE, particularly for Gold.

**Key takeaway:** Model choice in commodity forecasting should be governed by the end-use objective. The two approaches are complementary rather than substitutes — VAR for inference, LSTM for predictive accuracy.

---

## Limitations

- 60-day evaluation window may not generalise across all market regimes
- LSTM results are seed-dependent; single seed used for reproducibility
- Neither model incorporates macroeconomic covariates (e.g. DXY, VIX, interest rates)
- Transaction costs and liquidity constraints not modelled

---

## References

- Engle, R. F., & Granger, C. W. J. (1987). Co-integration and error correction. *Econometrica*, 55(2), 251–276.
- Granger, C. W. J. (1969). Investigating causal relations by econometric models. *Econometrica*, 37(3), 424–438.
- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735–1780.
- Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Springer.
- Sims, C. A. (1980). Macroeconomics and reality. *Econometrica*, 48(1), 1–48.
- Zhang, G. P. (2003). Time series forecasting using a hybrid ARIMA and neural network model. *Neurocomputing*, 50, 159–175.

---

## License

This project is for academic purposes. Data is sourced from publicly available repositories (FRED, Investing.com) under their respective terms of use.

---
## Author
**Lucia N Shumba**
