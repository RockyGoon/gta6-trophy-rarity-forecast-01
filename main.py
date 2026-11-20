#!/usr/bin/env python3
"""Forecast how many GTA VI players earn each trophy in week one.

Reads every ``*.in`` case in ``MAKEFUN_INPUT_DIR`` and writes one
``<id>.out`` JSON file per case to ``MAKEFUN_OUTPUT_DIR``::

    {"trophies": [{"id": 1, "q": [q05, q25, q50, q75, q95]}, ...]}

Model (v3)
----------
Two regimes, picked by whether the tracked-player count ever jumps
(a "refresh") inside the history window.

* No refresh seen (early page, a few hours of history): the player
  base is assumed to jump toward 3,868 tracked players (the size it
  reaches once the page has been flat for a while), and 103 new
  earners are split across the trophies as the per-trophy median of
  five allocations (late earliness, early earliness, proportional to
  current earners, uniform, burst), blended 50/50 toward the early
  earliness allocation.  Spread is a fixed 0.18 on the logit scale.
* Refreshes seen: the player base is projected with a log-log power
  law fitted on the last 24 h of refresh points and anchored at the
  last history slot; each trophy's earners grow by a log-linear fit
  of log(earners + 0.5) on log(1 + hours) over the last 24 h (48 h,
  then all history as fallback), with the slope shrunk toward the
  cross-trophy median slope (shrink first, then clamp to [0, 3]) and
  the intercept refit.  A concavity gate (quadratic fit on the last
  48 h) blends a concave projection in geometrically when it fires.
  Projections are capped by a per-trophy ceiling (max of its class
  ceiling and its last observed rate), floored at a 0.10% rate, and
  never below the last observed earners.  Spread combines
  newcomer-rate sensitivity, fit residuals, a fixed term and a
  random-walk term, scaled by 0.6.

Quantiles are the 5/25/50/75/95% points of a logit-normal around the
median rate, rounded to 4 decimals, clamped to [0.01, 99.99] and
forced non-decreasing.  A trophy already held by >= 99.99% is
forecast at 99.99 with spread 0.01.

Stdlib only.  Deterministic: same input bytes give same output bytes.
"""

import json
import math
import os
from datetime import datetime, timezone
from statistics import median

RELEASE = datetime(2026, 11, 19, tzinfo=timezone.utc)

# 5/25/50/75/95% points of the standard normal (full precision).
Z_VALUES = [-1.6448536269514722, -0.6744897501960817, 0.0,
            0.6744897501960817, 1.6448536269514722]

# Regime 2 (no refresh) constants.
FLAT_JUMP = 3868.0 / 638.0      # player-base jump once the page is flat
FLAT_HOURS = 2.5                # page flat long enough for the full jump
RAMP_HOURS = 4.7                # horizon over which the jump ramps in
NEW_EARNERS = 103.0             # new earners split across the trophies
SPREAD_NO_REFRESH = 0.18        # fixed logit spread, regime 2

# Regime 1 (refreshes) constants.
HORIZON_REF_DAYS = 4.52         # reference horizon for the player term
SPREAD_SCALE = 0.6              # overall spread multiplier, regime 1
SPREAD_FIXED = 0.30             # fixed spread component
SPREAD_RW = 0.25                # random-walk spread coefficient
PLAYER_COEF = 0.55              # player-uncertainty coefficient


# ---------------------------------------------------------------- helpers

def parse_ts(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_release(text):
    return (parse_ts(text) - RELEASE).total_seconds() / 3600.0


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def logit(p):
    return math.log(p / (1.0 - p))


def quantiles(med_rate, sigma):
    """5/25/50/75/95% points of a logit-normal around the median rate.

    Values are percentages, rounded to 4 decimals, clamped to
    [0.01, 99.99] and forced non-decreasing.
    """
    qs = []
    prev = -1.0
    for z in Z_VALUES:
        v = 100.0 * sigmoid(logit(med_rate) + z * sigma)
        v = round(v, 4)
        v = clamp(v, 0.01, 99.99)
        if v < prev:
            v = prev
        prev = v
        qs.append(v)
    return qs


def lsq(xs, ys):
    """Least squares y ~ a + b*x; returns (a, b, residuals)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx if sxx else 0.0
    a = my - b * mx
    return a, b, [y - (a + b * x) for x, y in zip(xs, ys)]


def quad_fit(xs, ys):
    """Least squares y ~ c0 + c1*x + c2*x^2; returns (coeffs, residuals)."""
    s = [0.0] * 5
    t = [0.0] * 3
    for x, y in zip(xs, ys):
        powers = [1.0, x, x * x, x * x * x, x * x * x * x]
        for i in range(5):
            s[i] += powers[i]
        for i in range(3):
            t[i] += y * powers[i]
    m = [[s[0], s[1], s[2], t[0]],
         [s[1], s[2], s[3], t[1]],
         [s[2], s[3], s[4], t[2]]]
    for c in range(3):
        piv = max(range(c, 3), key=lambda r: abs(m[r][c]))
        if abs(m[piv][c]) < 1e-12:
            return None
        m[c], m[piv] = m[piv], m[c]
        for r in range(3):
            if r != c:
                f = m[r][c] / m[c][c]
                for k in range(c, 4):
                    m[r][k] -= f * m[c][k]
    coeffs = [m[i][3] / m[i][i] for i in range(3)]
    resid = [y - (coeffs[0] + coeffs[1] * x + coeffs[2] * x * x)
             for x, y in zip(xs, ys)]
    return coeffs, resid


def ceiling(desc, tier):
    """Class ceiling for a trophy (fraction of players). First match wins."""
    d = desc.lower()
    t = tier.lower()
    if t == "platinum":
        return 0.02
    if "100% completion" in d:
        return 0.05
    for chapter, cap in (("chapter 1", 0.85), ("chapter 2", 0.70),
                         ("chapter 3", 0.60), ("chapter 4", 0.50)):
        if chapter in d:
            return cap
    if "final mission" in d and t == "gold":
        return 0.40
    if "complete the final mission for" in d:
        return 0.45
    if "mission strands for strangers" in d:
        return 0.20
    if "250 stunts" in d or "gold medal in all" in d:
        return 0.12
    if "100 locations" in d or "collectible card sets" in d:
        return 0.12
    if "complete 50 challenges" in d:
        return 0.20
    if "discover 100 vehicles" in d or "explore all of" in d:
        return 0.25
    if "purchase all garage" in d or "register 10 vehicles" in d:
        return 0.25
    if "meditate at every spot" in d or "unique animal" in d:
        return 0.25
    if "win 20 different races" in d or "drone race at every location" in d:
        return 0.20
    if "rob 5 bank vaults" in d:
        return 0.20
    if "laundering marked bills" in d or "spend $100,000" in d:
        return 0.25
    if "complete" in d and "mission" in d:
        return 0.55
    if t == "gold":
        return 0.35
    if t == "silver":
        return 0.20
    return 0.40


def hours_to_earn(desc, tier, mission_hours):
    """Hours a typical player needs for this trophy. First match wins."""
    d = desc.lower()
    t = tier.lower()
    if "gone fishing" in d:
        return 0.5
    if "100% completion" in d:
        return 90.0
    if t == "platinum":
        return 95.0
    for chapter, h in (("chapter 1", 3.0), ("chapter 2", 8.0),
                       ("chapter 3", 16.0), ("chapter 4", 26.0)):
        if chapter in d:
            return h
    if "final mission" in d and t == "gold":
        return 40.0
    if "try each type of activity" in d or "complete an arcade game" in d:
        return 2.0
    if "fishing challenge" in d:
        return 4.0
    if "250 stunts" in d:
        return 45.0
    if "gold medal in all" in d:
        return 30.0
    if "100 locations" in d or "collectible card sets" in d:
        return 35.0
    if "complete 50 challenges" in d:
        return 50.0
    if "discover 100 vehicles" in d or "explore all of" in d:
        return 25.0
    if "mission strands for strangers" in d:
        return 30.0
    if "win 20 different races" in d:
        return 20.0
    if "rob 5 bank vaults" in d:
        return 35.0
    if "laundering marked bills" in d or "spend $100,000" in d:
        return 22.0
    if d.startswith("complete ") and "mission" not in d and "chapter" not in d:
        return float(mission_hours)
    if "complete" in d and "mission" in d:
        return 20.0
    return 12.0


# ---------------------------------------------------------------- regime 2
# No refresh seen: early page, player base jumps, 103 new earners split
# across the trophies as the median of five allocations.

def forecast_no_refresh(case):
    trophies = case["trophies"]
    hist = case["history"]
    horizon_h = case["horizon_minutes"] / 60.0
    p_last = hist[-1]["players_tracked"]
    e_last = [hist[-1]["rates"][i] / 100.0 * p_last
              for i in range(len(trophies))]
    last_rate = [hist[-1]["rates"][i] / 100.0 for i in range(len(trophies))]
    # The last history slot is a bit before `at`; `at` is the forecast
    # origin, so flat time runs from the first slot to `at`.
    flat_h = (parse_ts(case["at"]) - parse_ts(hist[0]["at"])).total_seconds() / 3600.0
    if flat_h >= FLAT_HOURS and horizon_h >= 1.0:
        mult = FLAT_JUMP
    else:
        mult = 1.0 + (FLAT_JUMP - 1.0) * min(horizon_h, RAMP_HOURS) / RAMP_HOURS
    p_target = p_last * mult

    # The first-mission trophy ("gone fishing") gets no new earners.
    eligible = [i for i, tr in enumerate(trophies)
                if "gone fishing" not in tr["description"].lower()]

    def earliness(mission_hours, pool):
        weights = [math.exp(-hours_to_earn(trophies[i]["description"],
                                           trophies[i]["tier"],
                                           mission_hours) / 4.0)
                   for i in pool]
        total = sum(weights)
        return [NEW_EARNERS * w / total for w in weights]

    alloc_late = earliness(20, eligible)
    alloc_early = earliness(6, eligible)
    earners_total = sum(e_last[i] for i in eligible)
    alloc_prop = [NEW_EARNERS * e_last[i] / earners_total for i in eligible]
    alloc_unif = [NEW_EARNERS / len(eligible)] * len(eligible)

    # Burst: chapter 1 takes 40, the first six story missions take 6
    # each, the remaining 27 go by earliness (story missions early).
    story = [i for i in eligible
             if trophies[i]["description"].lower().startswith("complete ")
             and "mission" not in trophies[i]["description"].lower()
             and "chapter" not in trophies[i]["description"].lower()]
    chapter1 = [i for i in eligible
                if "chapter 1" in trophies[i]["description"].lower()]
    first_six = sorted(story, key=lambda i: trophies[i]["id"])[:6]
    rest = [i for i in eligible if i not in chapter1 and i not in first_six]
    rest_shares = earliness(6, rest)
    alloc_burst = {}
    for i in chapter1:
        alloc_burst[i] = 40.0
    for i in first_six:
        alloc_burst[i] = 6.0
    for k, i in enumerate(rest):
        alloc_burst[i] = 27.0 * rest_shares[k] / NEW_EARNERS

    out = []
    for pos, i in enumerate(eligible):
        five = [alloc_late[pos], alloc_early[pos], alloc_prop[pos],
                alloc_unif[pos], alloc_burst[i]]
        new = 0.5 * median(five) + 0.5 * alloc_early[pos]
        med_rate = (e_last[i] + new) / p_target
        if last_rate[i] >= 0.9999:
            med_rate = 0.9999
            sigma = 0.01
        else:
            sigma = SPREAD_NO_REFRESH
        out.append({"id": trophies[i]["id"], "q": quantiles(med_rate, sigma)})
    # The excluded trophy keeps its last earners (and the 99.99% rule).
    for i, tr in enumerate(trophies):
        if i not in eligible:
            med_rate = e_last[i] / p_target
            if last_rate[i] >= 0.9999:
                med_rate = 0.9999
                sigma = 0.01
            else:
                sigma = SPREAD_NO_REFRESH
            out.append({"id": tr["id"], "q": quantiles(med_rate, sigma)})
    out.sort(key=lambda row: row["id"])
    return {"trophies": out}


# ---------------------------------------------------------------- regime 1
# Refreshes seen: power-law player projection anchored at the last
# history slot, per-trophy log-linear earner growth with shrinkage
# toward the pool slope, a concavity gate, ceilings/floors, and a
# four-component spread.

def forecast_with_refresh(case):
    trophies = case["trophies"]
    hist = case["history"]
    n = len(trophies)
    horizon_days = case["horizon_minutes"] / 1440.0
    p_last = hist[-1]["players_tracked"]
    e_last = [hist[-1]["rates"][i] / 100.0 * p_last for i in range(n)]
    last_rate = [hist[-1]["rates"][i] / 100.0 for i in range(n)]
    last_t = hours_since_release(hist[-1]["at"])
    target_t = hours_since_release(case["target"])
    x_target = math.log1p(target_t)
    xs_all = [math.log1p(hours_since_release(s["at"])) for s in hist]
    last_dt = parse_ts(hist[-1]["at"])

    refresh_idx = [j for j in range(1, len(hist))
                   if hist[j]["players_tracked"] != hist[j - 1]["players_tracked"]]

    # --- player base: log-log power law, anchored at the last slot.
    def refresh_points(idxs):
        return [(hours_since_release(hist[j]["at"]),
                 hist[j]["players_tracked"]) for j in idxs]

    pts = [(t, p) for (t, p) in refresh_points(refresh_idx)
           if t >= last_t - 24.0]
    if len(pts) < 2:
        pts = refresh_points(refresh_idx)
    if len(pts) < 2:
        pts = [(hours_since_release(hist[0]["at"]), hist[0]["players_tracked"]),
               (last_t, p_last)]
    _, slope, _ = lsq([math.log1p(t) for t, _ in pts],
                      [math.log(p) for _, p in pts])
    slope = clamp(slope, 0.05, 1.0)
    p_target = p_last * ((1.0 + target_t) / (1.0 + last_t)) ** slope
    p_target = clamp(p_target, p_last, 10.0 * p_last)

    # --- per-refresh newcomer earn rate, median-pooled over all refreshes.
    newcomer = []
    for i in range(n):
        rates = []
        for j in refresh_idx:
            d_e = (hist[j]["rates"][i] / 100.0 * hist[j]["players_tracked"]
                   - hist[j - 1]["rates"][i] / 100.0 * hist[j - 1]["players_tracked"])
            d_p = hist[j]["players_tracked"] - hist[j - 1]["players_tracked"]
            rates.append(d_e / d_p)
        newcomer.append(median(rates))

    # --- per-trophy log-linear earner growth.
    fits = []
    for i in range(n):
        ys_all = [math.log(s["rates"][i] / 100.0 * s["players_tracked"] + 0.5)
                  for s in hist]
        chosen = None
        for window in (24, 48, None):
            if window is None:
                idx = list(range(len(hist)))
            else:
                idx = [k for k in range(len(hist))
                       if (last_dt - parse_ts(hist[k]["at"])).total_seconds() / 3600.0 <= window]
            if len(idx) >= 5:
                chosen = idx
                break
        xs = [xs_all[k] for k in chosen]
        ys = [ys_all[k] for k in chosen]
        a, b, resid = lsq(xs, ys)
        fits.append((a, b, resid, chosen, ys_all))
    pool_slope = clamp(median([f[1] for f in fits]), 0.0, 3.0)

    out = []
    for i in range(n):
        _, b_fit, resid, chosen, ys_all = fits[i]
        weight = min(1.0, e_last[i] / 20.0)
        b_shrunk = clamp(weight * b_fit + (1.0 - weight) * pool_slope, 0.0, 3.0)
        xs = [xs_all[k] for k in chosen]
        ys = [ys_all[k] for k in chosen]
        a2 = sum(ys) / len(ys) - b_shrunk * (sum(xs) / len(xs))
        e_lin = math.exp(a2 + b_shrunk * x_target) - 0.5
        sd_lin = (sum(r * r for r in resid) / len(resid)) ** 0.5
        sd_use = sd_lin

        # --- concavity gate: quadratic on the last 48 h.
        idx48 = [k for k in range(len(hist))
                 if (last_dt - parse_ts(hist[k]["at"])).total_seconds() / 3600.0 <= 48]
        cap_eff = max(ceiling(trophies[i]["description"], trophies[i]["tier"]),
                      last_rate[i])
        if len(idx48) >= 8:
            xsq = [xs_all[k] for k in idx48]
            ysq = [ys_all[k] for k in idx48]
            qfit = quad_fit(xsq, ysq)
            if qfit is not None:
                (c0, c1, c2), qres = qfit
                util = e_last[i] / (cap_eff * p_target) if cap_eff * p_target > 0 else 1.0
                if (e_last[i] >= 20 and util < 0.45 and last_rate[i] < 0.30
                        and c2 < -0.05):
                    yq = c0 + c1 * x_target + c2 * x_target * x_target
                    e_quad = math.exp(yq) - 0.5
                    if e_last[i] < e_quad < e_lin:
                        e_lin = (e_quad ** 0.6) * (e_lin ** 0.4)
                        sd_quad = (sum(r * r for r in qres) / len(qres)) ** 0.5
                        sd_use = max(sd_lin, sd_quad)

        # --- caps and floors.
        e_t = e_lin
        if e_t > cap_eff * p_target:
            e_t = cap_eff * p_target
        if e_last[i] >= 1 and e_t < 0.001 * p_target:
            e_t = 0.001 * p_target
        if e_t < e_last[i]:
            e_t = e_last[i]
        if e_t <= 0:
            e_t = max(e_last[i], 0.5)
        med_rate = e_t / p_target

        if last_rate[i] >= 0.9999:
            med_rate = 0.9999
            sigma = 0.01
        else:
            mix = newcomer[i]
            sens = clamp(1.0 - mix / max(med_rate, 0.0001), 0.0, 1.0)
            sig_p = PLAYER_COEF * math.sqrt(horizon_days / HORIZON_REF_DAYS) * sens
            sig_g = clamp(sd_use * 1.5, 0.20, 1.0)
            sig_rw = SPREAD_RW * math.sqrt(horizon_days)
            sigma = SPREAD_SCALE * math.sqrt(sig_p ** 2 + sig_g ** 2
                                             + SPREAD_FIXED ** 2 + sig_rw ** 2)
        out.append({"id": trophies[i]["id"], "q": quantiles(med_rate, sigma)})
    return {"trophies": out}


# ---------------------------------------------------------------- entry point

def forecast(case):
    hist = case["history"]
    refreshes = any(hist[j]["players_tracked"] != hist[j - 1]["players_tracked"]
                    for j in range(1, len(hist)))
    if refreshes:
        return forecast_with_refresh(case)
    return forecast_no_refresh(case)


def _jsonable(value):
    """Compact-JSON numbers: integral floats print as ints (85, not 85.0)."""
    if isinstance(value, float):
        return int(value) if value == int(value) else value
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "output")
    os.makedirs(out_dir, exist_ok=True)
    for name in sorted(os.listdir(in_dir)):
        if not name.endswith(".in"):
            continue
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        result = forecast(case)
        out_name = str(case.get("id") or name[:-3]) + ".out"
        with open(os.path.join(out_dir, out_name), "w", encoding="utf-8") as fh:
            fh.write(json.dumps(_jsonable(result), separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
