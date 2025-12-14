# Microstructure-Driven Intraday Relative Value

## Overview

This project studies **short-horizon relative value strategies** that exploit **temporary intraday price dislocations** between economically related instruments, with explicit attention to **market microstructure and execution costs**.

The goal is **not** to maximize standalone PnL, but to demonstrate a **buy-side research mindset**: identifying why short-term mispricings arise, testing whether they survive realistic trading frictions, and evaluating robustness under different liquidity regimes.

---

## Research Questions

* Do intraday price dislocations between closely related assets mean-revert after accounting for microstructure effects?
* How do **order flow imbalance, liquidity stress, and time-of-day effects** condition the speed and reliability of reversion?
* Does the apparent edge persist once **conservative execution costs** are applied?

---

## Strategy Universe

### 1. Single-Name Intraday Relative Value

* **MS – GS** (U.S. equities)
* Same sector, similar liquidity and institutional flow
* Designed to isolate microstructure-driven dislocations rather than fundamental divergence

### 2. Cross-Market Relative Value (Extension)

* **Hong Kong H-Index vs China A-Index proxies**
* Focuses on relative value under **market segmentation, investor heterogeneity, and trading frictions**
* Framed as short-horizon dislocation trading, not arbitrage

---

## Alpha Construction

* Intraday fair value defined via rolling, short-horizon beta
* Relative value signal based on standardized spread deviations
* Entry conditioned on **order flow imbalance and liquidity filters** to distinguish temporary dislocations from persistent moves

Holding periods are **minutes to hours**, with hard end-of-day exits.

---

## Execution Modeling

Due to the absence of consolidated historical SIP NBBO, execution is simulated using a **synthetic top-of-book model**:

* Bid–ask spreads calibrated to realistic intraday levels
* Spread widening during volatility and low-liquidity regimes
* Aggressive execution assumed at synthetic bid/ask with additional slippage and fees

Results are reported **gross and net of costs**, with explicit **cost sensitivity analysis**.

This execution model is intended to **stress-test signal robustness**, not to replicate true exchange-level microstructure.

---

## Risk Management

* Dollar-neutral position sizing
* Volatility-scaled exposure
* Maximum holding time and daily loss limits
* No overnight risk

---

## Evaluation Metrics

Beyond headline PnL, the analysis focuses on:

* Spread decay and mean-reversion half-life
* Win rate and average hold time
* Turnover and cost breakdown
* Net PnL per unit of spread
* Performance stability across intraday regimes

---

## Project Structure

```
microstructure_rv/
  data/            # raw and cleaned market data
  features/        # microstructure and spread features
  signals/         # relative value signal logic
  execution/       # synthetic top-of-book and cost models
  backtest/        # event-driven backtesting engine
  analysis/        # evaluation and reporting
```

---

## Disclaimer

This project is for **research and educational purposes only**. It does not constitute investment advice and is not intended to represent live trading performance.

---

## Key Takeaway

> Temporary intraday mispricings exist, but only a subset survive realistic microstructure frictions. The focus of this project is identifying and validating that subset.
