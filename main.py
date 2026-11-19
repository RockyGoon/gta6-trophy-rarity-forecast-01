"""GTA VI trophy-rarity forecaster — make.fun gta6-trophy-rarity-forecast-01.

For every GTA VI trophy on PS5, forecasts its Exophase earn rate at the
case's target as five quantiles, scored by pinball loss on the logit scale.

Why a ratio split wins
----------------------
An Exophase rate is earners / players_tracked, and the two sides move
differently between a case's checkpoint and its target:

* Earners only grow — tracked players keep playing and earning trophies.
* Players tracked only grows, and only in jumps. Exophase refreshes a
  game's counts on its own schedule, hours apart, and each refresh adds
  a batch of newly tracked players who have earned far fewer trophies
  than the players already tracked. A refresh therefore *dilutes* every
  rate even while earners grow. Launch night proved it: the first
  refresh took players tracked 638 -> 3,868 while every rate but the
  100% one fell.

The house persistence baseline (the kit starter) says every rate stays
put, so on the practice cases — whose histories are flat launch night
and whose answers are the refreshed page — it scores about 1.51. This
program splits the ratio and projects each side from the case's own
history instead:

1. Refreshes. A read whose players_tracked jumps is a refresh. Between
   refreshes the denominator is constant, so any rate move there is
   genuine earning.
2. Newcomer mix. At each refresh, the earners gained per player gained
   is the new batch's earn rate for that trophy. Pool it across
   refreshes (recency-weighted median — the latest batch's mix is the
   best guide to the next batch's) and shrink a trophy's sparse
   estimate toward the pool median.
3. Earner growth. Launch-week earning is fast early and slows, so
   log(earners) is near linear in log(1 + hours since release). A
   least-squares line per trophy over the most recent day, read at the
   target, with slopes shrunk toward the cross-trophy median when
   counts are small. Where the full history bends down (a concave
   quadratic) and the trophy sits far below its ceiling, the bent curve
   replaces the line — a straight line extrapolated 2-3x in time
   overshoots a saturating process.
4. Denominator. players_tracked is extrapolated as a power law in time
   since release, fitted on the last 24 h of refresh points.
5. Median. earners(target) / players(target), capped at the trophy's
   ceiling — the share of tracked players who will ever earn it, from
   its tier and its own description text. The ceiling only binds when
   an extrapolation would run past it.
6. Spread. A logit standard deviation built from the honest
   uncertainties: the denominator projection (weighted by how much the
   rate depends on it — how far the newcomer mix sits below the rate),
   the earner fit's residual, the newcomer mix, and a random-walk term.

No refresh in the history (the practice cases) falls back to launch
night's one observed refresh as a prior: a refresh within ~4.7 h,
taking players ~6x, with new players earning at ~30% of the tracked
rate — so the forecast still expects the dilution the practice answer
shows.

Deterministic, standard library only, ~0.1 s per case.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in row order.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z_SCORES = [-1.6448536269514722, -0.6744897501960817, 0.0,
            0.6744897501960817, 1.6448536269514722]

# Release day began 2026-11-19T00:00:00Z.
RELEASE_EPOCH = 1795046400

# Launch night's one observed refresh, used as a prior when a case's
# history holds no refresh at all: players jumped ~6x (638 -> 3,868)
# within ~4.7 h, and the new batch earned at ~30% of the tracked rate.
PRIOR_REFRESH_HOURS = 4.7
PRIOR_REFRESH_GROWTH = 5.0   # players multiply by (1 + 5) on a refresh
PRIOR_NEWCOMER_RATE = 0.30   # new players' rate, as a share of the rate

# Spread components, in logit units.
SIGMA_DENOMINATOR = 0.55     # denominator projection, scaled by horizon
SIGMA_NEWCOMER = 0.30        # newcomer-mix uncertainty
SIGMA_RANDOM_WALK = 0.25     # per square-root day
SIGMA_NO_REFRESH = 0.45      # extra when no refresh was ever seen
SIGMA_FULL_TROPHY = 0.05     # the trophy everyone holds stays ~100%


def parse_ts(value: str) -> float:
    """ISO 8601 UTC timestamp to Unix seconds."""
    return datetime.fromisoformat(
        value.strip().replace("Z", "+00:00")).timestamp()


def clamp_pct(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float):
    """Round to 4 decimals, half up, as JavaScript's Math.round does.

    A whole number comes back as an int, so json.dumps writes 100 the
    way JavaScript writes 100 (not 100.0).
    """
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if float(value).is_integer() else value


def median(values):
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return None
    if n % 2:
        return vals[n // 2]
    return 0.5 * (vals[n // 2 - 1] + vals[n // 2])


def weighted_median(pairs):
    """Median of (value, weight) pairs: the value where the cumulative
    weight first reaches half the total. None when there are no pairs."""
    if not pairs:
        return None
    total = sum(w for _, w in pairs)
    if total <= 0:
        return None
    seen = 0.0
    for value, weight in sorted(pairs):
        seen += weight
        if seen >= 0.5 * total:
            return value
    return sorted(pairs)[-1][0]


# --- ceilings: the share of tracked players who will ever earn a trophy ---

# (phrases, ceiling) in priority order; the first phrase found wins.
# A rule's phrases are alternatives — any one of them matches.
CEILING_RULES = [
    (("100% completion",), 0.05),
    (("complete chapter 1",), 0.85),
    (("complete chapter 2",), 0.70),
    (("complete chapter 3",), 0.60),
    (("complete chapter 4",), 0.50),
    (("complete the final mission for",), 0.45),
    (("mission strands for strangers",), 0.20),
    (("250 stunts", "gold medal in all"), 0.12),
    (("100 locations", "collectible card sets"), 0.12),
    (("complete 50 challenges",), 0.20),
    (("discover 100 vehicles", "explore all of"), 0.25),
    (("purchase all garage", "register 10 vehicles"), 0.25),
    (("meditate at every spot", "unique animal"), 0.25),
    (("win 20 different races", "drone race at every location"), 0.20),
    (("rob 5 bank vaults",), 0.20),
    (("laundering marked bills", "spend $100,000"), 0.25),
]

TIER_CEILINGS = {"gold": 0.35, "silver": 0.20, "bronze": 0.40}


def trophy_ceiling(trophy: dict) -> float:
    """Saturating ceiling from the trophy's tier and description text."""
    text = ((trophy.get("description") or "") + " "
            + (trophy.get("name") or "")).lower()
    tier = trophy.get("tier", "bronze")
    if tier == "platinum":
        return 0.02
    if "final mission" in text and tier == "gold":
        return 0.40
    for phrases, cap in CEILING_RULES:
        if any(p in text for p in phrases):
            return cap
    if "complete " in text and "mission" in text:
        return 0.55
    return TIER_CEILINGS.get(tier, 0.40)


# --- small least-squares helpers ---

def line_fit(xs, ys):
    """Least squares y = a + b x. Returns (slope, intercept, resid_sd,
    mean_x, mean_y), or None when the fit is degenerate (< 3 points or
    no spread in x). The centroid is returned so a later slope change
    can move the intercept with it and keep the line through the data."""
    if len(xs) < 3:
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    den = sum((x - mx) ** 2 for x in xs)
    if den <= 1e-9:
        return None
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    inter = my - slope * mx
    resid = math.sqrt(sum((y - (inter + slope * x)) ** 2
                          for x, y in zip(xs, ys)) / len(xs))
    return slope, inter, resid, mx, my


def _solve3(mat, vec):
    """Solve a 3x3 system by Gaussian elimination; None if singular."""
    m = [row[:] + [vec[i]] for i, row in enumerate(mat)]
    for col in range(3):
        piv = max(range(col, 3), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            return None
        m[col], m[piv] = m[piv], m[col]
        for r in range(3):
            if r == col:
                continue
            f = m[r][col] / m[col][col]
            for cc in range(col, 4):
                m[r][cc] -= f * m[col][cc]
    return [m[i][3] / m[i][i] for i in range(3)]


def concave_quad_fit(xs, ys):
    """Least squares y = a + b x + c x^2. Only a concave fit (c < -0.05)
    is returned, as (a, b, c, resid_sd); anything else is None."""
    n = len(xs)
    if n < 8:
        return None
    powers = [sum(x ** k for x in xs) for k in range(5)]
    moments = [sum(ys),
               sum(x * y for x, y in zip(xs, ys)),
               sum(x * x * y for x, y in zip(xs, ys))]
    mat = [[powers[0], powers[1], powers[2]],
           [powers[1], powers[2], powers[3]],
           [powers[2], powers[3], powers[4]]]
    coef = _solve3(mat, moments)
    if coef is None:
        return None
    a0, b0, c0 = coef
    if c0 >= -0.05:
        return None
    resid = math.sqrt(sum((y - (a0 + b0 * x + c0 * x * x)) ** 2
                          for x, y in zip(xs, ys)) / n)
    return a0, b0, c0, resid


def power_law_slope(points):
    """Least-squares slope of log(P) on log(1 + hours) for (hours,
    players) points, clamped to a sane band; None when unidentifiable."""
    if len(points) < 2:
        return None
    xs = [math.log1p(h) for h, _ in points]
    ys = [math.log(max(1.0, p)) for _, p in points]
    fit = line_fit(xs, ys)
    if fit is None:
        return None
    return min(1.0, max(0.05, fit[0]))


# --- case preparation ---

def build_series(case: dict) -> dict:
    """Turn a case into aligned time series: hours since release,
    players tracked, and earners per trophy per read (None where a read
    lacked that trophy's rate)."""
    trophies = case["trophies"]
    history = case["history"]
    n = len(trophies)
    times = []
    players = []
    earners = []
    for read in history:
        times.append((parse_ts(read["at"]) - RELEASE_EPOCH) / 3600.0)
        p = float(read["players_tracked"])
        players.append(p)
        rates = read["rates"]
        earners.append([
            (rates[i] / 100.0 * p) if rates[i] is not None else None
            for i in range(n)
        ])
    last = len(history) - 1
    p_last = players[last]
    e_last = [earners[last][i] if earners[last][i] is not None else 0.0
              for i in range(n)]
    r_last = [clamp_pct(e_last[i] / p_last * 100.0) if p_last else 0.0
              for i in range(n)]
    return {
        "n": n,
        "times": times,
        "players": players,
        "earners": earners,
        "t_last": times[last],
        "t_target": (parse_ts(case["target"]) - RELEASE_EPOCH) / 3600.0,
        "horizon_days": case["horizon_minutes"] / 1440.0,
        "p_last": p_last,
        "e_last": e_last,
        "r_last": r_last,
    }


def find_refreshes(players):
    """Indices k where read k is the first read after a players_tracked
    jump (a refresh)."""
    out = []
    for k in range(1, len(players)):
        if players[k] > players[k - 1] + max(2.0, 0.001 * players[k - 1]):
            out.append(k)
    return out


def newcomer_mix(series: dict, refreshes):
    """Per-trophy newcomer earn rate s_i: at each refresh, the earners
    gained per player gained. Pooled across refreshes with a
    recency-weighted median (weight halves every 24 h — the latest
    batch's mix is the best guide to the next batch's, since the
    player mix drifts from early adopters toward the wider launch
    audience) and shrunk toward the cross-trophy median; with no
    refresh observed, the launch-night prior (new players earn ~30%
    of the tracked rate)."""
    n = series["n"]
    players = series["players"]
    earners = series["earners"]
    times = series["times"]
    t_last = series["t_last"]
    per_trophy = [[] for _ in range(n)]
    for k in refreshes:
        dp = players[k] - players[k - 1]
        if dp <= 0:
            continue
        # Recency weight: halves every 24 h of age at the checkpoint.
        w = 0.5 ** ((t_last - times[k]) / 24.0)
        for i in range(n):
            a, b = earners[k - 1][i], earners[k][i]
            if a is None or b is None:
                continue
            per_trophy[i].append(
                (min(1.0, max(0.0, (b - a) / dp)), w))
    pool = weighted_median([pair for row in per_trophy for pair in row])
    s_i = []
    for i in range(n):
        m = weighted_median(per_trophy[i])
        if m is None:
            # No refresh observed for this trophy.
            if series["r_last"][i] >= 99.99:
                m = 1.0
            else:
                m = min(0.9, max(0.0005,
                                 PRIOR_NEWCOMER_RATE
                                 * series["r_last"][i] / 100.0))
        elif pool is not None and len(per_trophy[i]) < 2:
            m = 0.5 * m + 0.5 * pool
        s_i.append(m)
    return s_i


def earner_fits(series: dict):
    """Per-trophy log-time earner growth fits plus the cross-trophy
    median slope and the concavity fits used by the gate below."""
    n = series["n"]
    times = series["times"]
    earners = series["earners"]
    t_last = series["t_last"]
    fits = []
    slopes = []
    for i in range(n):
        # Fit on the most recent day: launch-week earning bends down
        # in log-log space (the slope over the whole history is the
        # explosive first hours and overshoots the target), so the
        # local slope is the honest extrapolant. Widen the window only
        # when the recent day holds too few reads to fit.
        fit = None
        for window in (24.0, 48.0, 1e9):
            xs, ys = [], []
            for k in range(len(times)):
                if times[k] < t_last - window:
                    continue
                e = earners[k][i]
                if e is None:
                    continue
                xs.append(math.log1p(max(0.0, times[k])))
                ys.append(math.log(e + 0.5))
            fit = line_fit(xs, ys)
            if fit is not None and len(xs) >= 5:
                break
        fits.append(fit)
        if fit is not None:
            slopes.append(fit[0])
    slope_med = median(slopes) if slopes else 0.5
    slope_med = min(3.0, max(0.0, slope_med))
    quads = []
    for i in range(n):
        xs = [math.log1p(max(0.0, times[k])) for k in range(len(times))
              if earners[k][i] is not None]
        ys = [math.log(earners[k][i] + 0.5) for k in range(len(times))
              if earners[k][i] is not None]
        quads.append(concave_quad_fit(xs, ys))
    return fits, slope_med, quads


def players_at_target(series: dict, refreshes):
    """Project players_tracked to the target as a power law in time
    since release, fitted on refresh points in the last 24 h; with no
    refresh in the history, expect one refresh (the launch-night prior)
    scaled by how much of its window the horizon covers."""
    times = series["times"]
    players = series["players"]
    t_last = series["t_last"]
    t_target = series["t_target"]
    p_last = series["p_last"]
    cut = t_last - 24.0
    pts = [(times[k], players[k]) for k in refreshes if times[k] >= cut]
    if len(pts) < 2:
        pts = [(times[k], players[k]) for k in refreshes]
    if len(pts) < 2:
        pts = [(times[-1], p_last), (times[0], players[0])]
    alpha = power_law_slope(pts)
    if alpha is None:
        alpha = 0.30
    p_target = p_last * ((1.0 + t_target) / (1.0 + t_last)) ** alpha
    p_target = min(p_target, p_last * 10.0)
    p_target = max(p_target, p_last)
    if not refreshes:
        horizon_hours = series["horizon_days"] * 24.0
        p_target = p_last * (1.0 + PRIOR_REFRESH_GROWTH
                             * min(1.0, horizon_hours / PRIOR_REFRESH_HOURS))
    return p_target


def forecast_trophy(i: int, trophy: dict, series: dict, s_i, fit,
                    slope_med: float, quad, p_target: float) -> dict:
    """One trophy's five quantiles: median earners at the target over
    projected players, capped at the ceiling, with a logit spread from
    the moving parts."""
    cap = trophy_ceiling(trophy)
    e_last = series["e_last"][i]
    r_last = series["r_last"][i]
    t_target = series["t_target"]
    if fit is not None:
        slope, inter, resid, mx, my = fit
        # Small counts -> shrink the slope toward the pool median, and
        # move the intercept with it so the line still passes through
        # the data's centroid.
        w = min(1.0, e_last / 20.0)
        slope = min(3.0, max(0.0, w * slope + (1 - w) * slope_med))
        inter = my - slope * mx
        e_target = math.exp(inter + slope
                            * math.log1p(max(0.0, t_target))) - 0.5
        resid_sd = resid
        # Concavity gate: where the full history bends down and the
        # trophy is far from its ceiling, the bent curve is the honest
        # extrapolant; near the ceiling the cap does the work, and for
        # tiny counts the quadratic is noise.
        u_now = e_last / max(1.0, cap * p_target)
        if (quad is not None and e_last >= 20 and u_now < 0.45
                and r_last < 30.0):
            a0, b0, c0, resid_q = quad
            x_t = math.log1p(max(0.0, t_target))
            e_q = math.exp(a0 + b0 * x_t + c0 * x_t * x_t) - 0.5
            if e_last <= e_q < e_target:
                e_target = e_q
                resid_sd = max(resid_sd, resid_q)
    else:
        # Too few reads to fit: hold earners flat (the denominator
        # projection still moves the rate).
        e_target = e_last
        resid_sd = 0.5
    e_target = min(max(e_target, e_last), cap * p_target)
    if e_target <= 0:
        e_target = max(e_last, 0.5)
    rate_med = clamp_pct(e_target / p_target * 100.0)

    # The trophy everyone holds stays (nearly) 100%.
    if r_last >= 99.99:
        rate_med = 99.9

    # Spread: denominator projection (weighted by how much the rate
    # depends on it — how far the newcomer mix sits below the rate),
    # earner-growth residual, newcomer mix, and a random walk.
    sens = min(1.0, max(0.0, 1.0 - s_i[i] / max(rate_med / 100.0, 1e-4)))
    sigma_p = SIGMA_DENOMINATOR * math.sqrt(
        series["horizon_days"] / 4.52) * sens
    sigma_g = min(1.0, max(0.20, resid_sd * 1.5))
    sigma = math.sqrt(sigma_p ** 2 + sigma_g ** 2 + SIGMA_NEWCOMER ** 2
                      + (SIGMA_RANDOM_WALK
                         * math.sqrt(series["horizon_days"])) ** 2)
    return rate_med, sigma


def forecast_case(case: dict) -> dict:
    if not case["history"]:
        raise ValueError("case has no history")
    series = build_series(case)
    refreshes = find_refreshes(series["players"])
    s_i = newcomer_mix(series, refreshes)
    fits, slope_med, quads = earner_fits(series)
    p_target = players_at_target(series, refreshes)

    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rate_med, sigma = forecast_trophy(
            i, trophy, series, s_i, fits[i], slope_med, quads[i], p_target)
        if not refreshes:
            sigma = math.sqrt(sigma ** 2 + SIGMA_NO_REFRESH ** 2)
        if series["r_last"][i] >= 99.99:
            sigma = SIGMA_FULL_TROPHY
        lg = logit(rate_med)
        qs = [round4(clamp_pct(inv_logit(lg + z * sigma))) for z in Z_SCORES]
        # Keep the row non-decreasing after rounding.
        for j in range(1, 5):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        out = OUTPUT_DIR / f"{path.stem}.out"
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n",
                       encoding="utf-8")


if __name__ == "__main__":
    main()
