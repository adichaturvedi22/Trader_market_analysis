# Trader Performance vs Market Sentiment — Primetrade.ai Assignment

**Dataset:** 211,224 trades from Hyperliquid (May 2023 – May 2025) · 32 unique traders · Bitcoin Fear/Greed Index (daily)

---

## Methodology

Both datasets were loaded and profiled — no missing values or duplicates were found in either. The trader data contained timestamps in `DD-MM-YYYY HH:MM` (IST) format which were parsed and stripped to date level. The Fear/Greed index already had a clean date column. Both were merged on date, retaining 211,218 of 211,224 trades (99.997% overlap across 479 trading days).

The five-class sentiment labels (Extreme Fear, Fear, Neutral, Greed, Extreme Greed) were collapsed into three (Fear, Neutral, Greed) for cleaner comparisons. Key metrics were then engineered at the trader × day level: daily PnL, win rate, average trade size, trade frequency, long/short ratio, and a drawdown proxy using cumulative PnL peak tracking.

With only 32 unique traders, KMeans clustering was not reliable so rule-based segmentation was applied instead — traders were split into Consistent Winners (win rate ≥ 45%), Hyper-Active Traders (avg trades/day ≥ 80), and Passive/Infrequent traders. Statistical differences between Fear and Greed days were tested using the Mann-Whitney U test (non-parametric, appropriate for skewed PnL distributions). A Gradient Boosting classifier was also trained to predict next-day profitability using sentiment and behavior features.

---

## Insights

**1. Fear-day overtrading destroys value**
Traders placed 37% more trades on Fear days compared to Greed days (105 vs 77 avg/day, p=0.039) yet earned 2.2× less median daily PnL ($123 vs $265). The extra volume is reactive and unprofitable — not alpha-generating.

**2. Contrarian long bias during Fear is penalised**
Long ratio jumped from 49% on Greed days to 55% on Fear days (p=0.004), indicating traders were aggressively buying dips. However this behavior correlated with worse outcomes, suggesting the market continued falling after these entries. Consistent Winners showed the smallest increase in long bias during Fear — they stayed closer to neutral.

**3. Consistent Winners are sentiment-resilient; Hyper-Active traders are not**
The PnL heatmap shows Hyper-Active Traders suffer the sharpest drop in performance on Fear days relative to Greed days. Consistent Winners maintained relatively stable PnL across sentiment regimes, suggesting their edge comes from discipline rather than riding market momentum.

---

## Strategy Recommendations

**Rule 1 — Fear Throttle** *(for Hyper-Active Traders)*
Cut daily trade count by ~50% whenever the Fear/Greed index is below 45. The data shows this segment's overtrading on Fear days is the single biggest drag on their returns. Fewer, higher-conviction trades during adverse sentiment reduces fee drag and drawdown exposure.

**Rule 2 — Greed Tilt** *(for Consistent Winners and Passive traders)*
When the Fear/Greed index crosses above 55, allow long position sizes up to 1.3× the normal limit. Greed days consistently produce better PnL across all segments and the market rewards directional long bias under positive sentiment. Revert to 1.0× sizing when the index drops below 50. Do not apply this during Fear days — the data shows long bias during Fear is actively penalised.

---

*Full analysis and code in `primetrade_analysis_v2.ipynb` · Charts generated via `generate_charts.py`*
