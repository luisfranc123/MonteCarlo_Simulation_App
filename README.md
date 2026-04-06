# Monte Carlo Portfolio Simulator

**Arin Risk Advisors, LLC** — Portfolio Construction Analysis

> *"Where smart money becomes brilliant."*

---

## What This App Does

This application simulates how a diversified investment portfolio might behave over a 10-year period under different market conditions and hedging strategies. It uses a **Monte Carlo simulation** to generate thousands of possible future return paths, rather than trying to predict a single outcome. The results are displayed as probability distributions — visual curves that show the range of likely outcomes for each strategy.

---

## Background — Key Concepts 

### What Is a Portfolio Distribution?

When we run thousands of simulations, each one produces a slightly different portfolio return. If we collect all those returns and plot them on a chart, we get a **distribution** — a curve showing which return values are most likely (the tall part of the curve) and which are rare (the thin tails at each end).

A narrow, tall distribution is good — it means returns are predictable and clustered tightly around the expected value. A wide, flat distribution means more uncertainty — outcomes could range broadly from very good to very bad.

### What Is a Collar Strategy?

A collar is an options-based hedging technique. It combines two contracts:

- A **put option** — this gives you the right to sell an asset at a pre-agreed price. It acts as insurance: if the asset falls sharply, the put limits your loss.
- A **call option** — this gives someone else the right to buy the asset from you at a pre-agreed price. Selling it generates income that helps pay for the put.

Together they create a "collar" around your returns — your downside is protected, but your upside is capped. The net effect is a tighter, more predictable return distribution. 

### What Is the Sharpe Ratio?

The Sharpe ratio measures how much return you earn per unit of risk taken. A higher Sharpe ratio means you are being better compensated for the risk you bear. It is calculated as:

```
Sharpe Ratio = (Portfolio Return − Risk-Free Rate) / Standard Deviation
```

The risk-free rate represents what you could earn with zero risk — typically a short-term government bond. Only the return above that baseline is worth attributing to investment skill or strategy.

### What Is Skewness and Kurtosis?

**Skewness** describes the asymmetry of a return distribution. A negative skew means the left tail (losses) is longer than the right tail (gains) — bad events tend to be more extreme than good ones. The collar strategies in this app intentionally produce negative skewness because they cap the upside while providing partial protection on the downside.

**Kurtosis** describes how fat the tails of a distribution are relative to a normal bell curve. Positive kurtosis means extreme events happen more often than expected. Negative kurtosis — which is what the collar strategies produce — means extreme events are rare, and returns cluster tightly around the mean. This is a desirable property for risk-averse investors.

---

## The Four Simulation Cases

| Case | Correlation | Hedging |
|---|---|---|
| Base Case | Fixed | None |
| Dynamic Beta Case | Floating / Stressed | None |
| Volatility Managed Case | Floating / Stressed | All assets |
| Equity Hedged Case | Floating / Stressed | Equity only |

**Base Case** — the simplest scenario. Each asset's return is derived from the S&P 500 using a fixed statistical relationship (beta). Correlations between assets stay constant throughout.

**Dynamic Beta Case** — a more realistic scenario. Correlations between assets are allowed to fluctuate randomly. Crucially, when markets fall sharply (S&P 500 down more than 2% in a month), correlations are stressed upward — capturing the well-documented phenomenon that assets tend to move together during crises, reducing the benefit of diversification precisely when you need it most.

**Volatility Managed Case** — the Dynamic Beta returns with a monthly collar applied to every asset in the portfolio. Returns are truncated to a ±5% band each month, net of the options cost.

**Equity Hedged Case** — the same as Volatility Managed, but the collar is only applied to equity assets. Fixed income and alternative assets run unhedged.

---

## Portfolio Assets

The simulator uses 21 assets spanning equities, fixed income, and alternatives — based on the Carson Institutional Portfolio constructed by Arin Risk Advisors.

| Category | Assets |
|---|---|
| US Equity | SPY, MDY, IWM |
| International Equity | EFA, VWO |
| Real Assets | DBC, VNQ, AMLP |
| Fixed Income | AGG, HYG, BWX, IBND, ELD, BSV, FLOT, TIP |
| Alternatives | WTMF, GLD, PSP, QAI (×2) |

Market data is fetched live from Yahoo Finance using a 5-year historical window. Expected returns, volatility, and correlations are computed from this data rather than hardcoded — meaning the simulation reflects current market conditions, not a fixed historical snapshot.

---

## Project Structure

```
Monte_Carlo_Simulation/
│
├── app.py              # Streamlit entry point — the user interface
├── data.py             # Asset definitions, yfinance data fetching
├── simulation.py       # Monte Carlo engine — all four cases
├── options.py          # Black-Scholes pricing and collar logic
├── analytics.py        # Portfolio statistics computation
├── charts.py           # Plotly visualisations
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## How to Run Locally

**Install dependencies**
```bash
pip install -r requirements.txt
```

**Run the app**
```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## Dependencies

| Library | Version | Purpose |
|---|---|---|
| `streamlit` | 1.55.0 | Web application framework |
| `numpy` | 2.4.3 | Numerical computation and random sampling |
| `pandas` | 2.3.3 | Data manipulation and DataFrame operations |
| `scipy` | 1.17.1 | Statistical functions (KDE, skewness, kurtosis) |
| `plotly` | 6.6.0 | Interactive charts and tables |
| `yfinance` | 1.2.0 | Live market data from Yahoo Finance |

---

## About Arin Risk Advisors

Arin Risk Advisors, LLC is a financial risk advisory firm specialising in volatility management and options-based portfolio strategies. The firm helps fiduciaries and institutional investors construct resilient portfolios that balance return objectives with downside protection.

---

*Built with Python and Streamlit. Market data provided by Yahoo Finance.*

---

## References

Arin Risk Advisors, LLC. (2016). Carson institutional portfolio construction write-up (Version 1.2) [Internal report].
