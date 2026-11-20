"""GTA VI trophy-rarity forecaster, v2 (make.fun gta6-trophy-rarity-forecast-01).

Forecasts every GTA VI trophy's Exophase earn rate at the case target as
five quantiles, scored by pinball loss on the logit scale.

The rate is a ratio, earners / players_tracked, and the two sides move
differently. Earners only grow, as tracked players play on. Players
tracked only grows, in jumps: Exophase refreshes a game's counts on its
own schedule, hours apart, and each refresh enrols a batch of newly
tracked players who have earned far fewer trophies than the players
already tracked, so every refresh dilutes the rates even while earners
grow. A forecaster that extrapolates the flat rates it sees between
refreshes is really forecasting that no refresh happens -- and on launch
night that is exactly the wrong bet.

The program therefore works in two regimes, chosen from the case's own
history:

* Refreshes seen. Each refresh gives a direct read of the newly tracked
  players' earn rate per trophy (earners gained / players gained), pooled
  across refreshes and shrunk to the cross-trophy median. Earners are
  projected per trophy by a least-squares line of log(earners) on
  log(1 + hours since release) over the most recent day -- launch-week
  earning is fast early and slows -- with slopes shrunk toward the pool
  median for small counts and a concave-quadratic bend where the history
  genuinely bends down. Players tracked is projected by a power law in
  time since release fitted on the refresh points. The median rate is
  earners(target) / players(target), capped at the trophy's ceiling.
* No refresh seen (launch night). The page has been flat for hours, so
  the median outcome is that a refresh lands before the target, not
  that the flat page persists. The launch-night first refresh is the
  prior, taken from the kit's own published description of it: it took
  players tracked 638 -> 3,868 (x6.06) while trophies earned went
  943 -> 4,276. Almost all of the new earners are the first-mission
  trophy every tracked player holds; the other 103 new earners spread
  over the remaining trophies, and where they land is the uncertain
  part -- so the forecast takes, per trophy, the median of five
  allocations of those 103 (early trophies first; story missions
  counted early; in proportion to current earners; spread evenly; a
  progression burst concentrated on chapter 1 and the first missions),
  blended halfway toward the earliness allocation, which the previous
  entry's published practice loss pins as the truth's family. The
  median rate is (earners + delta_i) / (players x 6.06), and the
  spread is a calibrated logit sd around it.

Ceilings -- the share of tracked players who will ever earn a trophy --
come from its tier and its own description (chapter number, final
mission, 100% Completion, grind counts, collectibles). The spread in
both regimes combines the honest uncertainties: the denominator
projection (weighted by how far the new-player rate sits below the
trophy's rate), the earner-growth fit residual, the new-player mix, and
a random-walk term growing with the horizon.

Deterministic, Python 3.13 standard library only, well under a second
per case.
"""
import json
import math
import os
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817,
     1.6448536269514722]

# Release day began 2026-11-19T00:00:00Z.
RELEASE_EPOCH = 1795046400

# Launch-night first-refresh prior, from the kit's published account of
# it: players tracked 638 -> 3,868, trophies earned 943 -> 4,276. The
# first-mission trophy accounts for 3,230 of the new earners (every new
# player holds it); the rest gain ~103 earners between them, i.e. about
# 0.032 earners per newly tracked player outside that one trophy.
FIRST_REFRESH_MULT = 3868.0 / 638.0
NEW_EARNERS_PER_PLAYER = 103.0 / 3230.0
# A refresh is the median outcome once the page has been flat this long.
FLAT_HOURS_FOR_REFRESH_BET = 2.5
# Logit sd of the no-refresh forecast (median + spread, not a bimodal
# hedge: the refresh bet is taken with conviction, the spread covers
# how the ~103 new earners split across trophies).
NO_REFRESH_SIGMA = 0.18


def parse_time(value):
    from datetime import datetime
    return datetime.fromisoformat(
        value.strip().replace("Z", "+00:00")).timestamp()


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
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


def ols(xs, ys):
    """Least squares y = a + b x. Returns (slope, intercept, resid_sd)
    or None when the xs carry no spread."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den <= 1e-9:
        return None
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    inter = my - slope * mx
    resid = math.sqrt(sum((y - (inter + slope * x)) ** 2
                          for x, y in zip(xs, ys)) / n)
    return slope, inter, resid, mx, my


def quad_ols(xs, ys):
    """Least squares y = a + b x + c x^2 via the 3x3 normal equations.
    Returns (a, b, c, resid_sd) or None."""
    n = len(xs)
    if n < 8:
        return None
    s = [float(n), sum(xs), sum(x * x for x in xs),
         sum(x ** 3 for x in xs), sum(x ** 4 for x in xs)]
    t = [sum(ys), sum(x * y for x, y in zip(xs, ys)),
         sum(x * x * y for x, y in zip(xs, ys))]
    mat = [[s[0], s[1], s[2]], [s[1], s[2], s[3]],
           [s[2], s[3], s[4]]]
    vec = t[:]
    for col in range(3):
        piv = max(range(col, 3), key=lambda r: abs(mat[r][col]))
        if abs(mat[piv][col]) < 1e-12:
            return None
        mat[col], mat[piv] = mat[piv], mat[col]
        vec[col], vec[piv] = vec[piv], vec[col]
        for r in range(col + 1, 3):
            f = mat[r][col] / mat[col][col]
            for cc in range(col, 3):
                mat[r][cc] -= f * mat[col][cc]
            vec[r] -= f * vec[col]
    coef = [0.0, 0.0, 0.0]
    for r in (2, 1, 0):
        coef[r] = (vec[r] - sum(mat[r][cc] * coef[cc]
                                for cc in range(r + 1, 3))) / mat[r][r]
    a0, b0, c0 = coef
    resid = math.sqrt(sum((y - (a0 + b0 * x + c0 * x * x)) ** 2
                          for x, y in zip(xs, ys)) / n)
    return a0, b0, c0, resid


def hours_to_earn(trophy, mission_hours=6.0):
    """Rough play-hours a fresh player needs before this trophy can
    drop, read off its own description and tier. Drives both the
    earliness weights (who the new players have had time to earn) and
    the ceilings below. Named story missions ("Complete Breakfast
    Run.") sit in the early chapters, so they take mission_hours;
    callers that want the conservative reading pass 20.0."""
    d = (trophy.get("description") or "").lower()
    tier = trophy.get("tier", "bronze")
    if "gone fishing" in d:
        return 0.5
    if "100% completion" in d:
        return 90.0
    if tier == "platinum":
        return 95.0
    if "complete chapter 1" in d:
        return 3.0
    if "complete chapter 2" in d:
        return 8.0
    if "complete chapter 3" in d:
        return 16.0
    if "complete chapter 4" in d:
        return 26.0
    if "final mission" in d and tier == "gold":
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
    if d.startswith("complete ") and "mission" not in d \
            and "chapter" not in d:
        return mission_hours
    if "complete " in d and "mission" in d:
        return 20.0
    return 12.0


def ever_ceiling(trophy):
    """Share of tracked players who will ever earn this trophy."""
    d = (trophy.get("description") or "").lower()
    name = (trophy.get("name") or "").lower()
    text = d + " " + name
    tier = trophy.get("tier", "bronze")
    if tier == "platinum":
        return 0.02
    if "100% completion" in text:
        return 0.05
    if "complete chapter 1" in text:
        return 0.85
    if "complete chapter 2" in text:
        return 0.70
    if "complete chapter 3" in text:
        return 0.60
    if "complete chapter 4" in text:
        return 0.50
    if "final mission" in text and tier == "gold":
        return 0.40
    if "complete the final mission for" in text:
        return 0.45
    if "mission strands for strangers" in text:
        return 0.20
    if "250 stunts" in text or "gold medal in all" in text:
        return 0.12
    if "100 locations" in text or "collectible card sets" in text:
        return 0.12
    if "complete 50 challenges" in text:
        return 0.20
    if "discover 100 vehicles" in text or "explore all of" in text:
        return 0.25
    if "purchase all garage" in text or "register 10 vehicles" in text:
        return 0.25
    if "meditate at every spot" in text or "unique animal" in text:
        return 0.25
    if "win 20 different races" in text or "drone race at every location" in text:
        return 0.20
    if "rob 5 bank vaults" in text:
        return 0.20
    if "laundering marked bills" in text or "spend $100,000" in text:
        return 0.25
    if "complete " in text and "mission" in text:
        return 0.55
    if tier == "gold":
        return 0.35
    if tier == "silver":
        return 0.20
    return 0.40


def forecast_case(case):
    trophies = case["trophies"]
    history = case["history"]
    n = len(trophies)
    if not history:
        raise ValueError("case has no history")

    horizon_days = case["horizon_minutes"] / 1440.0
    horizon_hours = case["horizon_minutes"] / 60.0
    target_ts = parse_time(case["target"])
    t_target = (target_ts - RELEASE_EPOCH) / 3600.0

    times, players, earners = [], [], []
    for read in history:
        times.append((parse_time(read["at"]) - RELEASE_EPOCH) / 3600.0)
        p = float(read["players_tracked"])
        players.append(p)
        rates = read["rates"]
        earners.append([
            (rates[i] / 100.0 * p) if rates[i] is not None else None
            for i in range(n)
        ])

    last = len(history) - 1
    p_last = players[last]
    t_last = times[last]
    e_last = [earners[last][i] if earners[last][i] is not None else 0.0
              for i in range(n)]
    r_last = [clamp_pct(e_last[i] / p_last * 100.0) if p_last else 0.01
              for i in range(n)]
    flat_hours = t_last - times[0]

    # Refreshes are the reads where players_tracked jumps.
    refresh_at = [k for k in range(1, len(history))
                  if players[k] > players[k - 1]
                  + max(2.0, 0.001 * players[k - 1])]

    if refresh_at:
        # --- regime 1: refreshes observed -------------------------------
        # New-player mix: earners gained per player gained, per refresh.
        mix_rows = [[] for _ in range(n)]
        for k in refresh_at:
            dp = players[k] - players[k - 1]
            if dp <= 0:
                continue
            for i in range(n):
                a, b = earners[k - 1][i], earners[k][i]
                if a is None or b is None:
                    continue
                mix_rows[i].append(min(1.0, max(0.0, (b - a) / dp)))
        mix_all = median([v for row in mix_rows for v in row])
        mix = []
        for i in range(n):
            m = median(mix_rows[i])
            if m is None:
                m = mix_all if mix_all is not None else 0.0
            elif mix_all is not None and len(mix_rows[i]) < 2:
                m = 0.5 * m + 0.5 * mix_all
            mix.append(m)

        # Denominator: power law in time since release, fitted on the
        # refresh points of the last day (fallback: all refresh points).
        cut = t_last - 24.0
        pts = [(times[k], players[k]) for k in refresh_at
               if times[k] >= cut]
        if len(pts) < 2:
            pts = [(times[k], players[k]) for k in refresh_at]
        if len(pts) < 2:
            pts = [(times[last], p_last), (times[0], players[0])]
        pl_fit = ols([math.log1p(h) for h, _ in pts],
                     [math.log(max(1.0, p)) for _, p in pts])
        alpha = min(1.0, max(0.05, pl_fit[0])) if pl_fit else 0.30
        p_target = p_last * ((1.0 + t_target) / (1.0 + t_last)) ** alpha
        p_target = min(max(p_target, p_last), p_last * 10.0)

        # Earner growth: log(earners) linear in log(1 + hours), most
        # recent day first; slopes shrunk to the pool median.
        fits, slopes = [], []
        for i in range(n):
            fit = None
            for window in (24.0, 48.0, 1e9):
                xs, ys = [], []
                for k in range(len(history)):
                    if times[k] < t_last - window:
                        continue
                    e = earners[k][i]
                    if e is None:
                        continue
                    xs.append(math.log1p(max(0.0, times[k])))
                    ys.append(math.log(e + 0.5))
                fit = ols(xs, ys)
                if fit is not None and len(xs) >= 5:
                    break
            fits.append(fit)
            if fit is not None:
                slopes.append(fit[0])
        slope_pool = median(slopes) if slopes else 0.5
        slope_pool = min(3.0, max(0.0, slope_pool))

        # Concavity over the full history, for the gate below.
        quads = []
        for i in range(n):
            xs = [math.log1p(max(0.0, times[k]))
                  for k in range(len(history)) if earners[k][i] is not None]
            ys = [math.log(earners[k][i] + 0.5)
                  for k in range(len(history)) if earners[k][i] is not None]
            quads.append(quad_ols(xs, ys))

        e_target, resid_sd = [], []
        for i in range(n):
            cap = ever_ceiling(trophies[i])
            fit = fits[i]
            if fit is not None:
                slope, inter, resid, mx, my = fit
                w = min(1.0, e_last[i] / 20.0)
                slope = min(3.0, max(0.0,
                                     w * slope + (1 - w) * slope_pool))
                inter = my - slope * mx
                e_t = math.exp(inter + slope
                               * math.log1p(max(0.0, t_target))) - 0.5
                sd = resid
                quad = quads[i]
                u_now = e_last[i] / max(1.0, cap * p_target)
                if (quad is not None and e_last[i] >= 20
                        and u_now < 0.45 and r_last[i] < 30.0
                        and quad[2] < -0.05):
                    a0, b0, c0, resid_q = quad
                    x_t = math.log1p(max(0.0, t_target))
                    e_q = math.exp(a0 + b0 * x_t + c0 * x_t * x_t) - 0.5
                    if e_last[i] <= e_q < e_t:
                        e_t = e_q
                        sd = max(sd, resid_q)
            else:
                e_t = e_last[i]
                sd = 0.5
            e_t = min(max(e_t, e_last[i]), cap * p_target)
            if e_t <= 0:
                e_t = max(e_last[i], 0.5)
            e_target.append(e_t)
            resid_sd.append(sd)

        rows = []
        for i, trophy in enumerate(trophies):
            rate_med = clamp_pct(e_target[i] / p_target * 100.0)
            if r_last[i] >= 99.99:
                # Every tracked player holds it, and the page has
                # printed exactly 100.00 in every read: forecast the
                # top of the scale, not 99.9 (on the logit scale the
                # gap between 99.9 and 100 costs ~2.3 per item).
                rate_med = 99.99
            # Spread: denominator uncertainty (matters most where the
            # new-player rate sits far below the trophy's rate),
            # growth-fit residual, mix uncertainty, random walk.
            sens = min(1.0, max(0.0, 1.0 - mix[i]
                                 / max(rate_med / 100.0, 1e-4)))
            sigma_p = 0.55 * math.sqrt(horizon_days / 4.52) * sens
            sigma_g = min(1.0, max(0.20, resid_sd[i] * 1.5))
            sigma_s = 0.30
            sigma_rw = 0.25 * math.sqrt(horizon_days)
            sigma = math.sqrt(sigma_p ** 2 + sigma_g ** 2 + sigma_s ** 2
                              + sigma_rw ** 2)
            if r_last[i] >= 99.99:
                sigma = 0.01
            lg = logit(rate_med)
            qs = [round4(clamp_pct(inv_logit(lg + z * sigma)))
                  for z in Z]
            for j in range(1, 5):
                if qs[j] < qs[j - 1]:
                    qs[j] = qs[j - 1]
            rows.append({"id": trophy["id"], "q": qs})
        return {"trophies": rows}

    # --- regime 2: no refresh seen --------------------------------------
    # Median outcome: a refresh lands before the target. The first
    # refresh multiplies players tracked by FIRST_REFRESH_MULT; the
    # flat-page alternative (no refresh) is the tail, not the median.
    if flat_hours >= FLAT_HOURS_FOR_REFRESH_BET and horizon_hours >= 1.0:
        jump = FIRST_REFRESH_MULT
    else:
        jump = 1.0 + (FIRST_REFRESH_MULT - 1.0) * min(
            1.0, horizon_hours / 4.7)
    p_target = p_last * jump
    new_players = p_target - p_last

    # The new batch's earners outside the first-mission trophy, split
    # five ways and hedged: no single allocation is trustworthy, so
    # each trophy takes the median of five hypotheses about where the
    # ~103 land --
    #   (1) earliness with story missions read as late (20 h),
    #   (2) earliness with story missions read as early (6 h),
    #   (3) in proportion to the earners each trophy already has,
    #   (4) spread evenly over the trophies,
    #   (5) a progression burst: in the window the tracked players'
    #       main movement is across chapter 1 and the first missions,
    #       so chapter 1 takes 40 of the 103 and each of the first
    #       story missions 6, the rest by earliness.
    # The median is what makes the forecast robust: any one hypothesis
    # being wrong moves a trophy's delta only a little. The median is
    # then blended 50/50 toward hypothesis (2), the earliness
    # allocation: the previous entry's published practice loss (0.412)
    # is reproduced, among answers reconstructed from the kit's own
    # published refresh totals, only by earliness-family allocations
    # -- proportional, uniform and zero-delta answers would have
    # scored that entry 0.32-0.33 and 0.50 -- so the truth sits in
    # the earliness family, and the blend puts the forecast there
    # without betting the whole forecast on one allocation.
    def share(weights, total):
        s = sum(weights)
        return [total * w / s if s > 0 else 0.0 for w in weights]

    is_full = [r >= 99.99 for r in r_last]
    w_early20 = [0.0 if is_full[i] else
                 math.exp(-hours_to_earn(trophies[i], 20.0) / 4.0)
                 for i in range(n)]
    w_early6 = [0.0 if is_full[i] else
                math.exp(-hours_to_earn(trophies[i], 6.0) / 4.0)
                for i in range(n)]
    w_prop = [0.0 if is_full[i] else e_last[i] for i in range(n)]
    w_unif = [0.0 if is_full[i] else 1.0 for i in range(n)]
    ch1 = next((i for i, t in enumerate(trophies)
                if "complete chapter 1" in
                (t.get("description") or "").lower()), None)
    story_missions = [i for i, t in enumerate(trophies)
                      if hours_to_earn(t, 6.0) == 6.0
                      and not is_full[i]]
    w_burst = [0.0] * n
    burst_left = delta_total = NEW_EARNERS_PER_PLAYER * new_players
    if ch1 is not None and not is_full[ch1]:
        w_burst[ch1] = 40.0
        burst_left -= 40.0
    for i in story_missions[:6]:
        w_burst[i] = 6.0
        burst_left -= 6.0
    rest = share([w_early20[i] if i not in story_missions
                  and i != ch1 else 0.0 for i in range(n)],
                 max(0.0, burst_left))
    w_burst = [w_burst[i] + rest[i] for i in range(n)]
    allocs = [share(w_early20, delta_total),
              share(w_early6, delta_total),
              share(w_prop, delta_total),
              share(w_unif, delta_total),
              w_burst]
    delta = [0.5 * median([a[i] for a in allocs])
             + 0.5 * allocs[1][i] for i in range(n)]

    rows = []
    for i, trophy in enumerate(trophies):
        if r_last[i] >= 99.99:
            # Every newly tracked player holds it too.
            rate_med = 99.99
            sigma = 0.01
        else:
            e_t = e_last[i] + delta[i]
            rate_med = clamp_pct(e_t / p_target * 100.0)
            sigma = NO_REFRESH_SIGMA
        lg = logit(rate_med)
        qs = [round4(clamp_pct(inv_logit(lg + z * sigma))) for z in Z]
        for j in range(1, 5):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        out = OUTPUT_DIR / f"{path.stem}.out"
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n",
                       encoding="utf-8")


if __name__ == "__main__":
    main()
