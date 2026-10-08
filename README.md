# Conv-Cup '26 — FIFA of AI: Planning-Based Soccer Bot

A bot for the **Conv-Cup '26 "FIFA of AI"** contest (Machine Learning Division, CyberLabs). The game is a deterministic 2D soccer simulation: two players, one ball, two goals, and mirrored random obstacles. Submissions play head-to-head matches on hidden seeds.

This repository holds our bot, the tools we built to test it, and a record of how and why it changed.


---

## Table of contents

1. [The task](#1-the-task)
2. [Approach](#2-approach)
3. [Repository layout](#3-repository-layout)
4. [How to run things](#4-how-to-run-things)
5. [Development history: what changed and why](#5-development-history-what-changed-and-why)
6. [Present state](#6-present-state)
7. [Results so far](#7-results-so-far)
8. [Expected results and how we will judge them](#8-expected-results-and-how-we-will-judge-them)
9. [Known issues and honest caveats](#9-known-issues-and-honest-caveats)

---

## 1. The task

- **Format:** team contest, 2–4 participants.
- **Game:** deterministic 2D soccer. Same seed + same starting state + same actions → same outcome.
- **Actions per step:** 8 directions plus STAY for movement. If you hold the ball you may also kick in one of 8 directions with power 1, 2 or 3 (kick distance 32 / 64 / 96).
- **Rules that shape strategy:**
  - Players move 4 units per step; the ball moves 8 per step.
  - A player who holds the ball for 10 steps is forced to kick.
  - A newly gained ball is protected from tackles for 3 steps.
  - A goal ends the play, resets positions, and the conceding player gets the next kickoff.
  - A stationary loose ball is claimed by the nearest player within 5 units. If nobody closes in on it for ~20 steps, it is dropped at the centre.
  - A match ends at the maximum iterations (400) or maximum goals (7).
- **Limits:** one action per game iteration, 2 seconds per decision, no external APIs, no dependencies outside the allowed list.
- **Evaluation:** head-to-head matches on hidden seeds and obstacle layouts. Ties go to extra iterations.

The official environment is in [`Anukul1712/Mlfootball_env`](https://github.com/Anukul1712/Mlfootball_env). This repo builds on its `participants` folder.

---

## 2. Approach

This bot is **model-based planning, not a trained network.** A Q-table or neural net learns "in situation X do Y" from many games. Our bot instead looks ahead at every step, like a chess engine.

### 2.1 The pieces

1. **A world model (`sim.py`).** A line-by-line port of the engine's `step` function in pure Python. It reproduces the real engine exactly (see [verification](#71-forward-model-verification)), so the bot can imagine the future with no learning error.
2. **Candidate actions.** Each turn the bot lists what it could do: up to 9 moves, plus 24 kick options (8 directions × 3 powers) when it holds the ball. It also considers a kick combined with a move toward a ball-holding opponent, because the engine lets a tackle and a kick happen in the same step.
3. **Rollouts.** For each candidate, it simulates about 18 steps ahead. After the first step, it plays its own side with a scripted continuation policy (`mine_policy`) and the opponent with a guessed style.
4. **A hand-written value function.** At the end of each rollout it scores the position. A goal is ±1. Otherwise it counts possession, ball progress toward the goal, who can reach a loose ball first, and a small tempo bonus for time spent holding the ball.
5. **Opponent modelling.** The bot keeps five guesses for how the opponent plays:
   - the organizer's *practice*, *aggressive* and *counter* bots,
   - our own heuristic (`mine`),
   - a *goalkeeper* style that stays near its own goal.

   Each step it checks which guess best predicted the opponent's real move and trusts that guess more.
6. **A robust choice.** The final score of each action is 70% weighted average across guesses plus 30% worst case. This is what lets it generalise instead of betting on one opponent.
7. **Two-stage search for speed.** All candidates are screened cheaply against the two most likely opponent styles. Only the best 8 are evaluated against all five.

### 2.2 Why this approach

| Question | Answer |
|---|---|
| Why not reinforcement learning? | Training is slow in pure Python, tabular models overfit the opponents they trained against, and the contest uses hidden opponents and seeds. |
| Why planning? | The simulator is deterministic and fully known, so look-ahead search is exact. Obstacles are just geometry, so the bot generalises to new layouts without retraining. |
| Why model several opponents? | Hidden opponents will not match any one script. Weighting guesses online, and keeping the worst case in the score, avoids committing to a single style. |
| Why hand-written values? | It is quick to build and easy to inspect. The cost is tuning effort, and the weights may overfit the test opponents. |

### 2.3 Special engine behaviours we use

- **Tackle-then-kick in one step.** The engine applies a kick from the action even if the player gained possession by tackling in that same step. The planner includes such candidates when an opponent holds the ball nearby. This depends on the exact engine behaviour; see [Known issues](#9-known-issues-and-honest-caveats).
- **Possession-timeout reset.** Kicking resets the possession counter. That created a degenerate loop we had to filter out (see the history below).

---

## 3. Repository layout

> This is the layout we recommend. Adjust it to match what you actually push.

```
.
├── README.md                     <- this file
├── .gitignore
├── participants/                 <- the organizers' starter files (soccer_env, config, tools)
│   ├── soccer_env/               <- engine (do not modify)
│   ├── config/
│   ├── evaluate.py               <- our scoreboard harness
│   └── my_team/                  <- THE SUBMISSION
│       ├── submission.json
│       ├── requirements.txt
│       ├── README.md             <- short submission documentation
│       └── team_bot/
│           ├── bot.py            <- protocol loop from the starter kit (unchanged)
│           ├── policy.py         <- entry point: Policy.load / choose_action
│           ├── planner.py        <- rollouts, opponent models, value function, search
│           ├── sim.py            <- exact forward model of the engine
│           ├── baseline.py       <- original starter policy, kept as safety fallback
│           └── models/
│               └── example_policy.json
└── tools/                        <- our testing scripts
    ├── verify_sim.py
    ├── bench.py
    ├── sweep.py
    ├── summarize.py
    ├── replay.py
    ├── dbg.py
    ├── quick.py
    ├── quick_camper.py
    └── shotlab.py
```

### 3.1 What each file does

**Submission (`my_team/team_bot/`)**

| File | Role |
|---|---|
| `sim.py` | Pure-Python forward model: player movement and contact resolution, tackles, kick timing, ball flight with wall and obstacle bounces, interceptions, loose-ball claims, drop-balls, goals and restarts. About 5 µs per step. |
| `planner.py` | The five opponent-style functions, our continuation policy, the exact-shot test, leaf value function, rollouts, candidate generation, opponent weighting and the two-stage search. All tunable numbers live in `DEFAULT_PARAMS`. |
| `policy.py` | Wraps the planner in the contest interface (`Policy.load`, `choose_action`). Resets per match, converts actions to the engine's format, and falls back to `tactical_action` if anything throws. |
| `baseline.py` | The original starter `policy.py`, kept as the fallback and for `tactical_action`. |
| `bot.py`, `submission.json` | Starter-kit process loop and launch description. |

**Tools (`tools/`)**

| File | Role |
|---|---|
| `verify_sim.py` | Plays random and scripted games in the real engine and the simulator side by side, and reports any difference. |
| `bench.py` | Plays the bot against the four organizer opponents on both sides. |
| `sweep.py` | Plays the bot against eight opponent styles (the four organizer ones, plus goal camper, random mover, mirror bot, and itself). Appends one JSON line per match. |
| `summarize.py` | Prints the win/draw/loss table, the non-wins, and the worst opponent. |
| `replay.py` | Step log for one match (positions, ball, action, events). |
| `dbg.py` | Stops a match at a chosen iteration and prints the planner's value for every candidate under every opponent model. |
| `quick.py`, `quick_camper.py` | Small, fast regression sets (8 matches and 4 camper matches). |
| `shotlab.py` | Physics experiment: maps which launch points beat a ball-tracking goalkeeper on an empty field. |

---

## 4. How to run things

Everything below assumes Python 3.10+ and that you run commands from the `participants` folder.

> **Path warning.** The scripts in `tools/` were written in a Linux sandbox and contain hardcoded paths such as `/home/claude/Mlfootball_env/participants` and `/home/claude/work/my_team`. Edit the `sys.path.insert(...)` lines and config paths near the top of each script to match your machine before running them. Fixing this properly (relative paths) is on the to-do list.

```powershell
# Watch the bot play (organizer tool)
python live_viewer.py --submission my_team/submission.json --opponent organizer-rl --seed 101

# Score against a reference opponent
python evaluate.py --opponent module --opponent-module reference_bot.policy --seeds 10

# Validate that the packaged bot runs through the real process protocol
python validate_submission.py --submission my_team/submission.json --matches-per-side 2

# Our tools
python tools/verify_sim.py 40            # forward-model exactness check
python tools/bench.py --seeds 3          # four organizer opponents
python tools/sweep.py --seeds 5 --out sweep.jsonl
python tools/summarize.py sweep.jsonl
python tools/replay.py camper 9101 player_1
python tools/dbg.py 62
python tools/quick_camper.py
python tools/shotlab.py
```

Timing: each match costs roughly 5–45 seconds on one CPU, depending on the opponent. A full 80-match sweep takes about half an hour or more.

---

## 5. Development history: what changed and why

### Step 0 — Baseline
The starter policy is a tabular Q-policy with a hand-written tactical fallback (dribble toward the goal, kick when close). It is our safety net and our reference point.

### Step 1 — Exact forward model (`sim.py`)
**Why:** every later decision depends on predicting the future correctly. A model that is only approximately right leads to confidently wrong plans.
**What:** port of the engine's step function.
**Check:** 160 matches / 61,731 steps, zero mismatches, every event type covered (tackles, contacts, drop-balls, goals, bounces, timeouts).

### Step 2 — First planner (`planner.py`)
**Why:** the fastest route to a strong bot that does not overfit to one opponent.
**What:** rollouts with four opponent styles, hand-written value function, candidate enumeration, opponent weighting, robust (mean + worst case) scoring.
**Problem:** about 25 seconds per match, too slow to benchmark on one CPU.

### Step 3 — Speed
**Why:** we needed many matches to measure anything.
**What:** two-stage search (cheap screen, then full evaluation of the best 8); horizon cut from 22 to 18; faster geometry helpers (coarser sampling, bounding-box rejection, fast paths).
**Result:** about 9–10 seconds per match at that point.

### Step 4 — Kick–bounce–intercept loop fix
**Found by:** replaying a drawn match (seed 9002 against the aggressive bot).
**Problem:** the bot stood next to an obstacle and kicked every step. The ball bounced straight back and was re-intercepted, which reset the possession counter, so the forced-kick timeout never fired. About 300 of 400 iterations were wasted. The aggressive bot did the same thing earlier in that match.
**Why the planner chose it:** "keep possession safely" scored well in its rollouts.
**Fix:** drop "boomerang" kicks, meaning kicks where the ball returns to the kicker within 3 steps after moving less than 15 units.

### Step 5 — Bigger test sweep
**Why:** 6 matches per opponent was too small and had no hard styles.
**What:** added `sweep.py` with goal camper, random mover, mirror bot and self-play, plus `summarize.py`.
**Finding:** against a pure goal camper, the planner drew all 10 matches 0–0.

### Step 6 — Fixing the camper stalemate (four problems stacked)

| # | Symptom | Cause | Fix |
|---|---|---|---|
| 1 | Bot alternated two opposite moves | Candidate values were nearly flat; tiny noise flipped the choice each step | Wider "who reaches the ball first" term, a time-to-ball penalty (`loose_time`), a small bonus for repeating the previous move (`stick_bonus`) |
| 2 | Bot waited instead of attacking | None of the four opponent models described a passive opponent, so weights stayed uniform and "wait for them to come" looked good | Added a fifth, goalkeeper-style opponent model; weights now lock on it against a camper |
| 3 | Fetching the ball looked worse than waiting | Horizon artifact: after fetching, the continuation made a pointless kick right before the rollout ended | Tempo reward for holding the ball (`tempo`), forward-only clearing kicks, and a rule that fetches a loose stationary ball unless the opponent is more than 3 steps closer (`fetch_margin`) |
| 4 | Still no goals against a keeper | Straight shots never beat a ball-tracking keeper (`shotlab.py` shows only narrow diagonal and wall-bank launch spots work). The continuation walked straight up the middle and its interception test assumed a perfect interceptor | Continuation now walks along the diagonal "shot line" toward the goal centre and fires only when an exact simulation against the modelled opponent says the shot scores |

**Result:** camper matches went from 0 wins / 10 draws to 6 wins / 4 draws / 0 losses, with goals 15–0 (see [Results](#72-sweeps-against-opponent-styles)).

### Parameters introduced in this process (`DEFAULT_PARAMS` in `planner.py`)

| Parameter | Value | Meaning |
|---|---|---|
| `horizon` | 18 | Rollout length in steps |
| `loose_adv`, `loose_prog`, `loose_time` | 0.35, 0.15, 0.008 | Loose-ball value: who gets it first, field progress, time-to-ball penalty |
| `poss_base`, `poss_prog` | 0.12, 0.45 | Value of holding the ball and of advancing it |
| `tempo` | 0.01 | Per-step reward for holding the ball, per-step penalty when the opponent does |
| `stick_bonus` | 0.01 | Tie-break for repeating the previous action |
| `fetch_margin` | −3.0 | Go fetch a stationary loose ball unless the opponent is more than this many steps closer |
| `robust_lambda` | 0.3 | Weight on the worst-case opponent model |
| `screen_models`, `screen_keep` | 2, 8 | Two-stage search sizes |
| `shot_checks` | 700 | Cap on exact shot simulations per decision |

---

## 6. Present state

- **Working:** exact forward model, five-model rollout planner, online opponent weighting, loop filter, keeper handling, exact-shot continuation.
- **Packaged:** a `my_team` folder and zip that follow the starter-kit layout.
- **Not yet done:** `validate_submission.py` and the ZIP checker have **not** been run on this build. A full re-run of the 80-match sweep on the latest planner was still in progress when this README was written.
- **Speed:** slowest single decision seen in recent sweeps was about 576 ms (earlier versions: under 200 ms). The limit is 2 s. The grading machine's speed is unknown.

---

## 7. Results so far

These numbers come from our own harness, using seeds 9000–9104 and both sides. They are small samples, so treat them as indications rather than guarantees. Results from different planner versions are labelled separately because we changed the code between runs.

### 7.1 Forward-model verification

| Check | Result |
|---|---|
| Matches / steps compared | 160 / 61,731 |
| Mismatches vs the real engine | **0** |
| Event types covered | tackle, player contact, drop-ball, possession timeout, goal, bounce, kick, interception, claim |

### 7.2 Sweeps against opponent styles

**Early planner (first benchmark, seeds 9000–9002, 6 matches per opponent, before the loop fix):**

| Opponent | W / D / L | Goals for–against |
|---|---|---|
| Practice | 6 / 0 / 0 | 38–0 |
| Aggressive | 5 / 1 / 0 | 20–0 |
| Counter | 5 / 1 / 0 | 36–1 |
| Organizer RL | 6 / 0 / 0 | 34–2 |

**Planner after the loop fix (5 seeds × 2 sides, 10 matches per opponent):**

| Opponent | W / D / L | Goals for–against |
|---|---|---|
| Random | 10 / 0 / 0 | 31–0 |
| Practice | 10 / 0 / 0 | 66–0 |
| Aggressive | 10 / 0 / 0 | 63–0 |
| Counter | 10 / 0 / 0 | 63–0 |
| Organizer RL | 10 / 0 / 0 | 59–2 |
| Mirror | 8 / 2 / 0 | 22–3 |
| Itself | 0 / 10 / 0 | 4–4 |
| **Goal camper** | **0 / 10 / 0** | **0–0** |

**Latest planner (after the camper fixes, partial — 58 of 80 matches done when last checked):**

| Opponent | W / D / L | Notes |
|---|---|---|
| Goal camper | 6 / 4 / 0 | goals 15–0 |
| Random | 10 / 0 / 0 | |
| Practice | 10 / 0 / 0 | |
| Aggressive | 8 / 0 / 0 | 8 of 10 played |
| Mirror | 8 / 2 / 0 | draws were 0–0 and 2–2 |
| Itself | 5 / 0 / 5 | losses mostly as player_1; cause not investigated |
| Counter, Organizer RL | not yet run with this version | |

> The sweep results above for the *latest* planner are incomplete. Please re-run `tools/sweep.py` on your own machine and replace this table with the finished numbers.

---

## 8. Expected results and how we will judge them

These are expectations, not measurements.

- **Against organizer-style and scripted bots:** the planner should win nearly all matches, as it has so far.
- **Against defensive opponents:** it should win more often than it draws now, but converting against a good keeper depends on narrow shot windows, so some draws are likely.
- **Against unseen opponents from other teams:** unknown. They may play unlike any of our five guessed styles. We can only say the design tries to avoid committing to one style.

How we will judge the bot:

1. Select on the **worst** opponent, not the average.
2. Use **30 or more seeds** on both sides per opponent.
3. Include new styles we have not tuned on.
4. Keep a held-out set of seeds and opponent styles that we never tune against.
5. Count draws as failures to investigate, since ties go to extra time.

---

## 9. Known issues and honest caveats

- **Four camper draws remain** (seeds 9101 as player_2, 9102 as both sides, 9103 as player_2). They have not been replayed, so the cause is unknown.
- **Self-play imbalance.** Two identical planners should be even, but the latest partial sweep shows losses concentrated on player_1. Not investigated.
- **Hardcoded game constants.** Ball radius, ball speed, possession radius, kick distances, the possession limit and the loose-ball restart count are fixed in `sim.py` to match `config/game.json`. The observation only provides field size, goal width, player radius and player speed. If the organizers change the other values, the simulator will predict wrong futures.
- **Tackle-then-kick quirk.** Some candidates rely on the engine applying a kick in the same step a tackle succeeds. If the organizers patch this, those candidates would be wasted, not harmful. There is currently no on/off switch for it.
- **No model-mismatch detector.** The bot does not compare its predictions with what actually happens and fall back when they diverge. This would be the safest guard against an engine change.
- **Greedy navigation.** Movement toward targets is greedy and not obstacle-aware, so it can stall behind obstacles. A grid-based path distance is the proposed fix.
- **Hand-tuned values.** The weights have not been systematically tuned and may overfit the test opponents.
- **Speed.** Decisions are now slower than earlier versions. Fine on our machine, unknown on the grader.
- **Version history.** The planner version used for the 10/10 sweep results was overwritten while we kept developing, and cannot be restored exactly. Use git history from now on.
- **Fetch rule is crude.** The "go fetch unless clearly slower" rule fixes a horizon artefact but is a hard-coded override of the search.
- **Not yet validated.** `validate_submission.py` and the ZIP checker have not been run on the latest build.

---


## Credits

- Environment and starter kit: Machine Learning Division, CyberLabs (`Anukul1712/Mlfootball_env`).
- Planner, forward model and test tooling: our team, developed with AI assistance.

*15 UCL / members: Shreyas Khanra, Aarav Gupta.*
