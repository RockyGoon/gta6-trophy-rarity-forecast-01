# GTA VI Trophy Rarity Forecast — Round 3 (v3)

Forecast how many GTA VI players earn each trophy in week one, as
probabilities: for each of the 55 trophies, the 5/25/50/75/95%
quantiles of the share of players (in percent) who will have earned
it by the target time.

## Running it

```sh
MAKEFUN_INPUT_DIR=<dir with *.in cases> \
MAKEFUN_OUTPUT_DIR=<dir for *.out files> \
python3 main.py
```

Reads every `*.in` case in `MAKEFUN_INPUT_DIR` and writes one
`<id>.out` JSON file per case to `MAKEFUN_OUTPUT_DIR`:

```json
{"trophies": [{"id": 1, "q": [0.0303, 0.0613, 0.1, 0.1632, 0.3299]}, ...]}
```

Stdlib only, no network, no credentials. Deterministic: two runs on
the same input produce byte-identical output. Runtime is ~0.06 s per
case (limit 8 s); outputs are ~2.8–3.0 KB, 55 trophies, 5 quantiles
each, non-decreasing, within [0.01, 99.99].

## Model

Two regimes, picked by whether the tracked-player count ever jumps
(a "refresh") inside the history window.

**No refresh seen (early page).** The player base is assumed to jump
toward 3,868 tracked players — the size it reaches once the page has
been flat for a while (full jump once flat ≥ 2.5 h and horizon ≥ 1 h,
otherwise the jump ramps in linearly over 4.7 h of horizon). 103 new
earners are split across the trophies as the per-trophy median of
five allocations — earliness with story missions late (20 h),
earliness with story missions early (6 h), proportional to current
earners, uniform, and a burst (chapter 1 takes 40, the first six
story missions 6 each, the remaining 27 by earliness) — blended
50/50 toward the early-earliness allocation. The first-mission
trophy ("gone fishing") gets no new earners. Spread is a fixed 0.18
on the logit scale.

**Refreshes seen.** The player base is projected with a log-log
power law fitted on the last 24 h of refresh points and anchored at
the last history slot (slope clamped to [0.05, 1.0], projection
clamped to [last, 10× last]). Each trophy's earners grow by a
log-linear fit of log(earners + 0.5) on log(1 + hours) over the last
24 h (48 h, then all history as fallback), with the slope shrunk
toward the cross-trophy median slope (weight min(1, earners_last /
20); shrink first, then clamp to [0, 3]) and the intercept refit. A
concavity gate — a quadratic fit on the last 48 h — blends a concave
projection in geometrically (quad^0.6 × linear^0.4) when the
quadratic coefficient is below −0.05, the trophy has ≥ 20 earners,
is under 45% of its effective ceiling and under a 30% rate, and the
quadratic projection lies between the last earners and the linear
projection. Projections are capped by a per-trophy ceiling (max of
its class ceiling and its last observed rate), floored at a 0.10%
rate when it already has ≥ 1 earner, and never below its last
observed earners. Spread combines newcomer-rate sensitivity (median
per-refresh earners gained / players gained), fit residuals
(clamped to [0.20, 1.0]), a fixed 0.30 term and a random-walk term,
scaled by 0.6.

**Quantiles.** The 5/25/50/75/95% points of a logit-normal around
the median rate, using the full-precision standard-normal quantiles
(−1.6448536269514722, −0.6744897501960817, 0, 0.6744897501960817,
1.6448536269514722), rounded to 4 decimals, clamped to [0.01, 99.99]
and forced non-decreasing. A trophy already held by ≥ 99.99% is
forecast at 99.99 with spread 0.01. Integral values print without a
decimal point (`85`, not `85.0`).

## Class ceilings and hours to earn

Ceilings (first match wins, on the lowercased description plus tier):
platinum 0.02; "100% completion" 0.05; chapters 1–4 → 0.85 / 0.70 /
0.60 / 0.50; "final mission" + gold 0.40; "complete the final mission
for" 0.45; "mission strands for strangers" 0.20; "250 stunts" or
"gold medal in all" 0.12; "100 locations" or "collectible card sets"
0.12; "complete 50 challenges" 0.20; "discover 100 vehicles" or
"explore all of" 0.25; "purchase all garage" or "register 10
vehicles" 0.25; "meditate at every spot" or "unique animal" 0.25;
"win 20 different races" or "drone race at every location" 0.20;
"rob 5 bank vaults" 0.20; "laundering marked bills" or "spend
$100,000" 0.25; "complete" + "mission" 0.55; gold 0.35; silver 0.20;
bronze 0.40.

Hours to earn (same first-match order): "gone fishing" 0.5; "100%
completion" 90; platinum 95; chapters 1–4 → 3 / 8 / 16 / 26; "final
mission" + gold 40; "try each type of activity" or "complete an
arcade game" 2; "fishing challenge" 4; "250 stunts" 45; "gold medal
in all" 30; "100 locations" or "collectible card sets" 35; "complete
50 challenges" 50; "discover 100 vehicles" or "explore all of" 25;
"mission strands for strangers" 30; "win 20 different races" 20;
"rob 5 bank vaults" 35; "laundering marked bills" or "spend
$100,000" 22; a "complete …" trophy that is neither a mission nor a
chapter takes the story-mission hours (6 early / 20 late);
"complete" + "mission" 20; default 12.

## Scores (validated v3 model)

Practice cases (no refresh): mean pinball 0.1052. Simulated
10-seed cases (refreshes): mean pinball 0.1974, worst 0.2437 — down
from v2's 0.3986 / 0.6032. No peeking: nothing in the model hardcodes
answers or true rates.

## Validation (this rebuild, 20 Nov 2026)

Outputs are byte-identical to the validated v3 reference outputs on
all 12 cases: practice `p00` / `p01` (2,783 B each) and sim seeds
1–10 `f00.out` (2,934–2,986 B). Two consecutive runs on the kit
input dir are byte-identical; runtime ~0.06 s per run.
