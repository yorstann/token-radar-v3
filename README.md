[![make.fun · token-radar-01](https://make.fun/badge/token-radar-01.svg)](https://make.fun/bounty/token-radar-01?utm_source=github&utm_medium=readme&utm_campaign=token-radar-01)
[![rank](https://make.fun/badge/token-radar-01/yorstann/token-radar-v3/rank.svg)](https://make.fun/bounty/token-radar-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=token-radar-01)
[![reward](https://make.fun/badge/token-radar-01/yorstann/token-radar-v3/reward.svg)](https://make.fun/bounty/token-radar-01?utm_source=github&utm_medium=readme&utm_campaign=token-radar-01)

# Token Launch Radar

An entry for the make.fun bounty [Build the best token launch radar](https://make.fun/bounty/token-radar-01).

The radar sweeps a sealed snapshot of 758 DeFi protocols (DefiLlama data, as of 2026-11-14) and ranks all **342 protocols that record no token yet** by what a launch could be worth — then tells the reader exactly how to farm each one. What is new in this version: the price is not asserted, it is **fitted** — every number in the model was estimated from the snapshot's own priced protocols and survived (or failed) a held-out test whose results are printed in the dashboard, not buried.

## The fitted model

`log(market cap) = category offset + intercept + Σ kept terms`

- **Fitting set.** 128 protocols in the snapshot already trade with a positive market cap; 95 of them also record a listing date and form the fitting set. The other 33 are held back as an extra check, predicted with the fitting set's median age standing in for the date they do not record.
- **Category offset.** Each category's mean `log(mcap / TVL)`, shrunk toward the global mean with a pseudo-count of 8, so a category priced by two names cannot shout. The global mean multiple is **0.229× TVL**. Categories with no priced peer (18 of the board's) fall back to it, and the dashboard says so.
- **Term selection.** Four candidate terms were offered — log TVL, log age, log(1 + chains), and 7-day momentum as `tanh(change/20)`. A term was kept only if it lowered the held-out median absolute percentage error in **both** validations (5-fold cross-validation, folds by slug order; and a time split at the median listing date, 2024-02-18), each refitting offsets and coefficients on its training part alone. **Kept: log TVL, coefficient 0.519945** (intercept 8.527695) — market cap scales sub-linearly with TVL. Dropped, with their held-out deltas (CV / time-split median error): log age 79.97% → 83.80% / 86.13% → 88.88%; log chains 79.97% → 82.56% / 86.13% → 86.32%; momentum 79.97% → 79.67% / 86.13% → 87.84% (it helped one test and hurt the other, so it went).
- **Honest accuracy.** Held-out, the fitted model scores R² (log) **0.1166** and median error **79.97%** in cross-validation; R² (log) **−0.0027** and median error **86.13%** on the time split; median error **89.63%** on the 33 date-less protocols. That is the real spread of launch pricing — which is why every board row carries a **P10–P90 band** from the out-of-fold residuals (−2.34 … +2.09 in log space; the band held the actual cap 78.9% of the time held-out, against a nominal 80%). The fit prices the typical protocol; the largest exchange-scale caps sit far above the line, and the dashboard's predicted-vs-actual table shows that too.
- **Shadow guardrail.** After the fit, products of an already-tokenized parent — the wrapped/staked/bridged and exchange categories the watchlist rule sets aside, versioned names, wrapped/staked names — are priced at **0.35×** and flagged (144 rows): they record no token, but they are not clean launch candidates whatever the regression says.

The board covers the house watchlist **40/40** and ranks all 342 pre-token protocols; each row carries a concrete farm plan (6 steps with a dollar cost band and a time estimate) generated from its category, chains, TVL tier, age and seven-day print — including a risk sentence worded from that print when TVL falls 15%+, spikes 20%+, or is not reported at all.

## The dashboard

A dark radar page in three movements: a KPI strip with the fit's headline numbers; a **Methodology & fit diagnostics** section (the model story, coefficient table, fitted category multiples, the term-selection table with both validations' errors before and after, the validation table, and predicted-vs-actual for the ten largest priced protocols); and the board itself — sortable and filterable (search, category, chain, watchlist-only, hide-shadow), with a per-row Signal (the row's implied-value percentile on this board; context only, it never moves a rank), its P10–P90 band, and a detail view per protocol showing the price decomposed into additive log-space parts (category offset, TVL scale, shadow discount) next to its farm plan as a checklist. A final strip shows the largest already-trading protocols — the fit's training pool — for scale.

## Run it

```sh
mkdir -p out
MAKEFUN_OUTPUT_DIR=out node main.ts --input <path-to-snapshot.json>
```

Node 22.18+, standard library only. The program fits the model at runtime and is deterministic: two runs over the same snapshot write byte-identical `board.json`, dashboard data, and `dossier.json`.
