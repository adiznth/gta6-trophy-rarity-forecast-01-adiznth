# gta6-trophy-rarity-forecast-01-adiznth

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/adiznth/gta6-trophy-rarity-forecast-01-adiznth/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/adiznth/gta6-trophy-rarity-forecast-01-adiznth/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A forecaster for GTA VI's PlayStation 5 trophy earn rates on Exophase, one
week after launch. Given the house's reads of the trophy page up to the
deadline, it predicts, for each of the game's 55 trophies, five quantiles of
the share of tracked players who will hold that trophy at the target time.

The core idea is that an earn rate is a ratio, and its two sides move for
different reasons. The number of players Exophase tracks grows as the
launch's audience piles in, which dilutes every rate. The number of players
who have earned a given trophy grows as those players progress through the
game, which lifts each trophy on its own schedule. So `main.py` fits the two
sides separately — log players-tracked and log earners per trophy, each
against log time since launch — and divides the two forecasts to get the
rate. Three guardrails keep the extrapolation honest: the fits use only the
recent reads (early launch growth is explosive and over by the deadline),
the extrapolation is damped because both the player influx and the
progression catch-up decelerate, and earners never decrease while the
forecast's move from the last read is capped. A trophy nobody has earned
yet gets a small positive guess from its tier's rarest earned trophy, with
the widest spread. The spread around each median combines a random-walk
term over the horizon, the fits' residuals, and how far past the data the
target lies.

Tested locally on the kit's practice cases (valid output, medians anchored
to the last read on flat history) and on three simulated final-like weeks —
fast adoption, saturating adoption, and dilution-dominated growth — where it
beat the house persistence baseline on pinball loss in every run, with
honest 90% coverage. Edge cases (a single read, all-zero rates, missing
rates, 0% and 100% trophies) all produce valid, deterministic output in
under a second.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`main.py` is the whole entry: Python 3.13, standard library only, no
network. It reads every `<id>.in` in the input directory and writes one
`<id>.out` per case to the output directory.

---

<sub>Entered on [make.fun](https://make.fun/?ref=p9nkmgn8&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
