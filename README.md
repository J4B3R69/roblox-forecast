# Up-and-Coming Forecaster

A momentum-aware persistence forecast for Roblox's Up-and-Coming
chart, in pure Python (standard library only).

`main.py` reads the chart history at `$MAKEFUN_INPUT_DIR`
(`history.ndjson` + `task.json`) and writes `forecast.json` to
`$MAKEFUN_OUTPUT_DIR`: for each of the top 10 places, an honest
probability for every plausible game, plus the predicted top 10.

Each game is summarised by its last rank, its recency-weighted mean
rank (24 h half-life), its rank trend over the last 48 hours, its
player-count growth and its rank volatility. Those combine into a
predicted mean rank and spread; per-place probabilities are a
discretised normal around the prediction, scaled by the game's
chance of reaching the top 10 at all. Deterministic: the same
history always yields the same forecast.

Entry for make.fun's roblox-up-and-coming-forecast-01 competition.
MIT licensed.
