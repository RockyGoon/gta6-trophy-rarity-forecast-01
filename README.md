# GTA VI Trophy Rarity Forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/RockyGoon/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/RockyGoon/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts, for every Grand Theft Auto VI trophy on PlayStation 5, the
percent of the players Exophase tracks who will have earned it at
2026-11-26T00:00:00Z — a week after launch — as five quantiles a trophy,
from everything the house recorded of the game's trophy page up to the
deadline. Entry for the make.fun bounty
**gta6-trophy-rarity-forecast-01** ($250, free entry).

## Why the obvious forecast loses

A trophy's Exophase rate is a ratio, `earners / players_tracked`, and
both sides move over the forecast week:

- **Earners only grow.** Tracked players keep playing and earning.
- **Players tracked only grows, and only in jumps.** Exophase refreshes
  a game's counts on its own schedule, hours apart, and each refresh
  adds a batch of newly tracked players who have earned far fewer
  trophies than the players already tracked. Every refresh therefore
  *dilutes* the rates, even while earners grow. Launch night proved
  it: the first refresh took players tracked from 638 to 3,868 while
  every rate but the 100% one fell.

The house persistence baseline says every rate stays where it is. On
the practice cases it scores a pinball loss of about 1.51, because the
refresh took every rate outside its narrow band. The house curve
baseline fits a straight line to each trophy's logit rate and does no
better on the practice, for the same reason: neither separates the
ratio's two sides.

## The model

`main.py` (Python 3.13, standard library only, deterministic, ~0.1 s
per case) splits the ratio and projects each side from the case's own
history:

1. **Refresh detection.** Reads where `players_tracked` jumps are
   refreshes; between them the denominator is constant, so any rate
   move there is genuine earning.
2. **Newcomer mix.** At each refresh, `earners gained / players
   gained` is the earn rate of the newly tracked players for that
   trophy, pooled (median) across refreshes and shrunk to the
   cross-trophy median. Newer refreshes count more than older ones:
   the mix of the latest batch of players is the best guide to the
   next batch's.
3. **Earner growth.** Launch-week earning is fast early and slows, so
   `log(earners)` is near linear in `log(1 + hours since release)`. A
   least-squares line per trophy over the most recent day, read at the
   target (slopes shrunk toward the cross-trophy median when counts
   are small). Where the full history shows real concavity (a negative
   quadratic term) and the trophy sits far below its ceiling, the bent
   curve is used instead — a straight line extrapolated 2–3x in time
   overshoots a saturating process.
4. **Denominator.** `players_tracked` is extrapolated as a power law
   in time since release, fitted on the last 24 h of refresh points.
5. **Median.** `earners(target) / players(target)`, capped at the
   trophy's ceiling — the share of tracked players who will ever earn
   it, from its tier and its own description text (chapter number,
   "final mission", "100% Completion", grind counts, collectibles).
   The ceiling only binds when an extrapolation would run past it.
6. **Spread.** A logit standard deviation built from the honest
   uncertainties: the denominator projection (weighted by how much
   the rate depends on it, i.e. how far the newcomer mix sits below
   the rate), the earner fit's residual, the newcomer mix, and a
   random-walk term. Quantiles are the median shifted by the normal
   z-scores.

No refresh in the history (the practice cases, whose histories are
flat) falls back to launch night's one observed refresh as a prior: a
refresh within ~4.7 h, taking players ~6x, with new players earning
at ~30% of the tracked rate — so the forecast still expects the
dilution the practice answer shows.

## Validation

No public history of another game's week-one trophy curve exists
(Exophase shows only current rates), so the model was validated on a
purpose-built launch simulator: 60,000 tracked players arriving over
launch week, playing at lognormal speeds, earning each trophy after
the play-hours its description implies, with Exophase refreshing
every ~4 h. Scored with the contest's own pinball loss on the logit
scale, over five simulated launches (mean):

| forecaster | mean pinball loss |
|---|---|
| this entry | **0.357** |
| house curve baseline | 0.364 |
| house persistence baseline | 0.818 |

The simulator's dynamics are log-time-linear by construction, which
flatters the curve baseline; the real final rewards the
refresh/dilution handling more, since the recorded history spans two
days of real refreshes.

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` case in the input directory and writes one
`<id>.out` per case: `{"trophies": [{"id": n, "q": [q05, q25, q50,
q75, q95]}, ...]}`, five quantiles of each trophy's earn rate at the
case's target, in percent, non-decreasing.

## License

MIT — see [LICENSE](LICENSE).

---
Built for [make.fun](https://make.fun/r/qh87q9q5) — join with referral code `qh87q9q5`.
