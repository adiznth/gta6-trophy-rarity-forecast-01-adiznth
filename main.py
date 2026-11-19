"""GTA VI trophy rarity forecaster (make.fun gta6-trophy-rarity-forecast-01).

Forecasts, for every PlayStation 5 trophy of Grand Theft Auto VI, five
quantiles of its earn rate on Exophase at the case's target, from the house's
reads of the trophy page up to the case's checkpoint.

An earn rate is a ratio, earners over players tracked, and the two sides move
for different reasons. Players tracked grows as the launch's audience piles
in (diluting every rate), while earners grow as those players progress
through the game (lifting each trophy on its own schedule). So the program
fits the two sides separately and divides:

  - log players-tracked is fit against log(1 + hours since launch) over the
    recent reads, and extrapolated to the target;
  - log earners of each trophy is fit the same way, its slope shrunk toward
    the players slope (in the long run a trophy's earners can only grow as
    fast as the player base, so a steeper slope is a catch-up that fades),
    and extrapolated to the target;
  - the median forecast is 100 * earners / players at the target.

Three things keep the extrapolation honest. First, the fits use only the
recent reads (the last 36 hours): early launch growth is explosive, and a
slope fitted over the whole history is a slope from a phase that has already
ended. Second, the extrapolation is damped: the slope applies over a
log-time distance that grows only logarithmically past the data, because
both the player influx and the progression catch-up decelerate. Third,
earners never decrease (they are a cumulative count), the players count
never shrinks, and the forecast's move from the last read is capped, so a
wild fit cannot explode. A trophy nobody has earned yet gets a small
positive median from its tier's rarest earned trophy, with the widest
spread: it is probably about to start earning, just far below the rest.

The spread around the median combines a random-walk term that grows with
the horizon (as the house baseline does), the residuals of the two fits,
and an extra term for how far past the data the target lies, so the five
quantiles are an honest band, not a guess.

Runs as `python3 -P -s /bot/main.py` (Python 3.13, standard library only) in
the grader's sandbox, no network. Reads every <id>.in in the input directory
($MAKEFUN_INPUT_DIR, else /input) and writes <id>.out per case to the output
directory ($MAKEFUN_OUTPUT_DIR, else /output). Deterministic: no randomness,
no wall-clock reads, one pass over the data.
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

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Launch-day origin for log time: the start of 2026-11-19 UTC.
LAUNCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Only the recent reads carry the current trend; older ones are a phase that
# has ended. Recency half-life inside that window, in hours.
WINDOW_HOURS = 36.0
HALF_LIFE_HOURS = 12.0
# How strongly a trophy's slope is pulled to the players slope.
SHRINK_K = 6.0
# Damping of the extrapolation: a log-time step dx past the data counts as
# DAMP_TAU * ln(1 + dx / DAMP_TAU). Near the data it is dx; far past it, it
# grows only logarithmically, because growth decelerates.
DAMP_TAU = 0.7
# Caps that keep an extrapolation inside the plausible: at most 300x the
# players tracked, a logit move of 3 from the last read (about a 20x rate
# move), and a spread between 0.6 and 2.0 logit units.
MAX_PLAYERS_GROWTH = 300.0
MAX_LOGIT_MOVE = 3.0
SIGMA_MIN = 0.6
SIGMA_MAX = 2.0
# Random-walk spread, logit units per square-root day (as the baseline's).
SIGMA_RW = 0.5
# Extra spread per unit of log-time past the last read.
SIGMA_EXTRAP = 0.12


def clamp(x: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, x))


def logit(pct: float) -> float:
    p = clamp(pct, 0.01, 99.99)
    return math.log(p / (100.0 - p))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float):
    """Round to 4 decimals, half up, as JavaScript's Math.round does it."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_launch(moment: datetime) -> float:
    return max(0.0, (moment - LAUNCH).total_seconds() / 3600.0)


def log_time(moment: datetime) -> float:
    return math.log1p(hours_since_launch(moment))


def damped(dx: float) -> float:
    """A log-time step past the data, damped: dx nearby, ~log(dx) far away."""
    if dx <= 0:
        return dx
    return DAMP_TAU * math.log1p(dx / DAMP_TAU)


def weighted_fit(xs, ys, ws):
    """Weighted least squares of y on x.

    Returns (slope, xbar, ybar, resid_std, n_eff): the slope, the weighted
    means (the anchor the forecast hangs from), the weighted residual
    standard deviation, and the effective sample size. With fewer than two
    distinct x values there is no slope to fit: slope 0, resid 0.
    """
    total_w = sum(ws)
    if total_w <= 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    xbar = sum(w * x for x, w in zip(xs, ws)) / total_w
    ybar = sum(w * y for y, w in zip(ys, ws)) / total_w
    sxx = sum(w * (x - xbar) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        resid = math.sqrt(
            sum(w * (y - ybar) ** 2 for y, w in zip(ys, ws)) / total_w
        )
        return 0.0, xbar, ybar, resid, total_w
    slope = sum(w * (x - xbar) * (y - ybar) for x, y, w in zip(xs, ys, ws)) / sxx
    resid = math.sqrt(
        sum(w * (y - (ybar + slope * (x - xbar))) ** 2 for x, y, w in zip(xs, ys, ws))
        / total_w
    )
    return slope, xbar, ybar, resid, total_w


def forecast_case(case: dict, name: str) -> dict:
    trophies = case["trophies"]
    history = sorted(case["history"], key=lambda r: r["at"])
    if not history:
        raise ValueError(f"{name} has no history")
    target = parse_time(case["target"])
    x_target = log_time(target)
    horizon_days = case["horizon_minutes"] / 1440.0
    last_read = history[-1]
    t_last = parse_time(last_read["at"])
    x_last = log_time(t_last)

    # The recent window: reads from the last 36 hours (all of them, when the
    # history is shorter). Weights decay with a 12-hour half-life inside it.
    window = [r for r in history
              if (t_last - parse_time(r["at"])).total_seconds() <= WINDOW_HOURS * 3600.0]
    if len(window) < 3:
        window = history
    xs = [log_time(parse_time(r["at"])) for r in window]
    weights = [
        0.5 ** ((t_last - parse_time(r["at"])).total_seconds() / 3600.0 / HALF_LIFE_HOURS)
        for r in window
    ]

    # --- Players tracked: one shared fit, it anchors every trophy. ---
    n_last = last_read["players_tracked"] or 0
    n_points = [
        (x, math.log(r["players_tracked"]), w)
        for x, r, w in zip(xs, window, weights)
        if r["players_tracked"] and r["players_tracked"] > 0
    ]
    if len(n_points) >= 2:
        slope_n, xbar_n, ybar_n, resid_n, _ = weighted_fit(
            [p[0] for p in n_points], [p[1] for p in n_points], [p[2] for p in n_points]
        )
        slope_n = clamp(slope_n, 0.0, 3.5)
        log_n_target = ybar_n + slope_n * damped(x_target - xbar_n)
        n_target = math.exp(log_n_target)
    else:
        slope_n, resid_n = 0.0, 0.0
        n_target = float(n_last) if n_last > 0 else 1.0
    n_floor = float(n_last) if n_last > 0 else 1.0
    n_target = clamp(n_target, n_floor, n_floor * MAX_PLAYERS_GROWTH)

    # --- Earners per trophy: fit, shrink the slope to the players slope. ---
    fits = []
    for i, trophy in enumerate(trophies):
        points = []
        for x, r, w in zip(xs, window, weights):
            rate = r["rates"][i]
            if rate is None or not r["players_tracked"]:
                continue
            earners = max(0.0, rate * r["players_tracked"] / 100.0)
            points.append((x, math.log(earners + 0.5), w))
        if len(points) >= 2:
            slope, xbar, ybar, resid, n_eff = weighted_fit(
                [p[0] for p in points], [p[1] for p in points], [p[2] for p in points]
            )
        elif len(points) == 1:
            slope, xbar, ybar, resid, n_eff = 0.0, points[0][0], points[0][1], 0.0, 1.0
        else:
            slope, xbar, ybar, resid, n_eff = 0.0, x_last, 0.0, 0.0, 0.0
        last_rate = None
        last_earners = 0.0
        for r in reversed(history):
            if r["rates"][i] is not None and r["players_tracked"]:
                last_rate = r["rates"][i]
                last_earners = max(0.0, last_rate * r["players_tracked"] / 100.0)
                break
        fits.append({
            "tier": trophy["tier"],
            "slope": slope,
            "xbar": xbar,
            "ybar": ybar,
            "resid": resid,
            "n_eff": n_eff,
            "last_rate": last_rate,
            "last_earners": last_earners,
        })

    # The rarest earned rate per tier, for trophies nobody has earned yet.
    tier_rarest = {}
    for tier in {f["tier"] for f in fits}:
        earned = [f["last_rate"] for f in fits
                  if f["tier"] == tier and f["last_rate"] is not None and f["last_rate"] > 0]
        tier_rarest[tier] = min(earned) if earned else None

    rows = []
    for trophy, fit in zip(trophies, fits):
        if fit["last_rate"] is None or fit["last_earners"] <= 0:
            # Nobody has earned this one yet: guess a quarter of its tier's
            # rarest earned rate (at least 0.05%), with the widest spread.
            rarest = tier_rarest.get(trophy["tier"])
            median = 0.25 * rarest if rarest else 0.05
            median = clamp(median, 0.05, 1.0)
            sigma = SIGMA_MAX
        else:
            w = fit["n_eff"] / (fit["n_eff"] + SHRINK_K) if fit["n_eff"] > 0 else 0.0
            slope = w * fit["slope"] + (1.0 - w) * slope_n
            slope = clamp(slope, slope_n - 1.0, slope_n + 2.0)
            log_e_target = fit["ybar"] + slope * damped(x_target - fit["xbar"])
            # The fit ran on log(earners + 0.5), so undo the +0.5 here:
            # without it a flat history would forecast half an extra
            # earner per trophy.
            e_target = math.exp(log_e_target) - 0.5
            # Earners are cumulative: never forecast fewer than the last read saw.
            e_target = max(e_target, fit["last_earners"])
            median = 100.0 * e_target / n_target
            median = clamp(median, 0.01, 99.99)
            # Cap how far the median may move from the last read.
            lo = inv_logit(logit(fit["last_rate"]) - MAX_LOGIT_MOVE)
            hi = inv_logit(logit(fit["last_rate"]) + MAX_LOGIT_MOVE)
            median = clamp(median, lo, hi)
            # Spread: random walk over the horizon, fit residuals, and how
            # far past the data the target lies.
            base = SIGMA_RW * math.sqrt(horizon_days)
            resid = math.sqrt(max(fit["resid"], 0.10) ** 2 + max(resid_n, 0.10) ** 2)
            extrap = SIGMA_EXTRAP * max(0.0, x_target - x_last)
            sigma = clamp(math.sqrt(base * base + resid * resid + extrap * extrap),
                          SIGMA_MIN, SIGMA_MAX)
        lg = logit(median)
        qs = [round4(clamp(inv_logit(lg + z * sigma), 0.0, 100.0)) for z in Z]
        # Rounding can only tie neighbours, never invert them.
        for k in range(1, len(qs)):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
