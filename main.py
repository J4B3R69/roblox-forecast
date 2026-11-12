#!/usr/bin/env python3
"""Roblox Up-and-Coming forecaster - momentum-aware persistence.

Reads the chart history at $MAKEFUN_INPUT_DIR (history.ndjson +
task.json) and writes forecast.json to $MAKEFUN_OUTPUT_DIR: for each
of the top 10 places, an honest probability for every plausible game,
plus the predicted top 10 in order.

Model: a game is summarised by its last rank, its recency-weighted
mean rank, its recent rank trend, its player-count growth and its
rank volatility. Those combine into a predicted mean rank mu and a
spread sigma; the per-place probabilities are a discretised normal
around mu, scaled by the game's chance of being in the top 10 at all.
Pure standard library, no network, deterministic: the same history
always yields the same forecast.
"""

import json
import math
import os

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

HALF_LIFE_H = 24.0      # recency weight half-life, hours
ABSENT_RANK = 45.0      # rank imputed when a game misses a snapshot
MAX_PLACE_ROWS = 60     # rows kept per place (rule allows 100)
MIN_P = 0.002           # smallest probability worth naming


def parse_time(s):
    # "2026-11-07T20:53:19.000Z" -> epoch seconds (UTC)
    import calendar
    import time as _time

    fmt = "%Y-%m-%dT%H:%M:%S"
    head = s[:19]
    frac = 0.0
    if len(s) > 20 and s[19] == ".":
        digits = ""
        for ch in s[20:]:
            if ch.isdigit():
                digits += ch
            else:
                break
        frac = float("0." + digits) if digits else 0.0
    return calendar.timegm(_time.strptime(head, fmt)) + frac


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def load_history(path):
    snaps = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows = obj.get("up_and_coming") or []
            chart = {}
            for r in rows:
                try:
                    chart[int(r["universe_id"])] = (
                        int(r["rank"]),
                        float(r.get("playing") or 0),
                    )
                except (KeyError, TypeError, ValueError):
                    continue
            if chart:
                snaps.append((parse_time(obj["at"]), chart))
    snaps.sort(key=lambda s: s[0])
    return snaps


def game_features(snaps):
    """Return {universe_id: feature dict} from the snapshot list."""
    t_end = snaps[-1][0]
    games = {}
    for t, chart in snaps:
        age_h = (t_end - t) / 3600.0
        w = 0.5 ** (age_h / HALF_LIFE_H)
        for gid, (rank, playing) in chart.items():
            g = games.setdefault(gid, {
                "sw": 0.0, "swr": 0.0, "swr2": 0.0,
                "recent": [], "logp": [],
                "last_rank": None, "last_playing": 0.0,
                "seen_last": False, "seen": 0,
            })
            g["sw"] += w
            g["swr"] += w * rank
            g["swr2"] += w * rank * rank
            g["recent"].append((t, rank))
            if playing > 0:
                g["logp"].append((t, math.log(playing)))
            g["seen"] += 1
        # mark who is in this snapshot for last-snapshot tracking
        for gid, (rank, playing) in chart.items():
            games[gid]["last_rank"] = rank
            games[gid]["last_playing"] = playing
            games[gid]["seen_last"] = (t == t_end)
    return games, t_end


def summarise(games, t_end):
    out = {}
    for gid, g in games.items():
        if g["sw"] <= 0:
            continue
        wmean = g["swr"] / g["sw"]
        var = max(0.0, g["swr2"] / g["sw"] - wmean * wmean)
        vol = math.sqrt(var)
        last_rank = g["last_rank"] if g["last_rank"] is not None else ABSENT_RANK
        if not g["seen_last"]:
            last_rank = max(last_rank, wmean + 6.0)
        # trend over the last 48 hours only: mean rank of the older
        # half of that window vs the newer half (positive = climbing)
        rec = [(t, r) for (t, r) in g["recent"] if t >= t_end - 48 * 3600]
        trend = 0.0
        if len(rec) >= 4:
            mid = rec[len(rec) // 2][0]
            old = [r for (t, r) in rec if t < mid]
            new = [r for (t, r) in rec if t >= mid]
            if old and new:
                trend = (sum(old) / len(old)) - (sum(new) / len(new))
        # player-count growth per day from a two-point log fit
        growth = 0.0
        lp = g["logp"]
        if len(lp) >= 2:
            (t0, v0), (t1, v1) = lp[0], lp[-1]
            span_d = (t1 - t0) / 86400.0
            if span_d > 0.25:
                growth = (v1 - v0) / span_d
        mu = 0.55 * last_rank + 0.45 * wmean
        mu -= 0.45 * max(-6.0, min(6.0, trend))
        mu -= 1.2 * max(-1.0, min(1.0, growth))
        mu = max(1.0, min(30.0, mu))
        sigma = max(2.0, min(6.0, 1.8 + 0.55 * vol))
        out[gid] = {
            "mu": mu, "sigma": sigma,
            "last_rank": last_rank, "last_playing": g["last_playing"],
        }
    return out


def build_forecast(stats, places):
    cands = {g: s for g, s in stats.items() if s["mu"] <= 24.0}
    per_place = {k: [] for k in range(1, places + 1)}
    for gid, s in cands.items():
        mu, sigma = s["mu"], s["sigma"]
        p_in = norm_cdf((places + 0.5 - mu) / sigma)
        if p_in < 0.01:
            continue
        dens = []
        total = 0.0
        for k in range(1, places + 1):
            d = norm_cdf((k + 0.5 - mu) / sigma) - norm_cdf((k - 0.5 - mu) / sigma)
            dens.append(d)
            total += d
        if total <= 0:
            continue
        for k in range(1, places + 1):
            p = p_in * dens[k - 1] / total
            if p >= MIN_P:
                per_place[k].append((gid, p))
    rows = []
    for k in range(1, places + 1):
        lst = sorted(per_place[k], key=lambda kv: -kv[1])[:MAX_PLACE_ROWS]
        ssum = sum(p for _, p in lst)
        scale = 0.97 / ssum if ssum > 0.97 else 1.0
        for gid, p in lst:
            rows.append({"universe_id": gid, "place": k,
                         "p": round(p * scale, 6)})
    order = sorted(cands.items(),
                   key=lambda kv: (kv[1]["mu"], -kv[1]["last_playing"]))
    top10 = [gid for gid, _ in order[:places]]
    return {"probabilities": rows, "top10": top10}


def main():
    with open(os.path.join(INPUT_DIR, "task.json"), "r", encoding="utf-8") as f:
        task = json.load(f)
    places = int(task.get("places", 10))
    out_name = task.get("output", "forecast.json")
    snaps = load_history(os.path.join(INPUT_DIR, "history.ndjson"))
    games, t_end = game_features(snaps)
    stats = summarise(games, t_end)
    forecast = build_forecast(stats, places)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, out_name), "w", encoding="utf-8") as f:
        f.write(json.dumps(forecast) + "\n")


if __name__ == "__main__":
    main()
