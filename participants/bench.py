"""bench.py: planner vs several opponents, both sides, many seeds (single process)."""
import sys, time, argparse, importlib, copy, random, os
from pathlib import Path

# Dynamically add the current folder (participants) and my_team to the system path
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("my_team"))

from soccer_env import GameConfig, SoccerEnv
from soccer_env.bots import practice_action, aggressive_action, counter_action
from organizer_rl_bot.policy import OrganizerRLOpponent

# Use a relative path to load the config file
config = GameConfig.from_json("config/game.json")

def make_opps(names):
    out = {}
    for n in names:
        if n == "practice": out[n] = lambda: practice_action
        elif n == "aggressive": out[n] = lambda: aggressive_action
        elif n == "counter": out[n] = lambda: counter_action
        elif n == "rl":
            o = OrganizerRLOpponent()
            def f(o=o):
                o.start_episode(); return o.choose_action
            out[n] = f
    return out

def play(ours, theirs, side, seed):
    env = SoccerEnv(config); obs = env.reset(seed=seed)
    opp = "player_2" if side == "player_1" else "player_1"
    mx = 0.0
    while not env.done:
        t = time.perf_counter(); a = ours(obs[side]); mx = max(mx, time.perf_counter() - t)
        b = theirs(obs[opp])
        acts = {side: a, opp: b}
        obs, info = env.step(acts["player_1"], acts["player_2"])
    sc = env.score
    return sc[side], sc[opp], mx

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--opps", default="practice,aggressive,counter,rl")
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--seed-start", type=int, default=9000)
    ap.add_argument("--params", default="{}")
    ap.add_argument("--module", default="team_bot.policy")
    a = ap.parse_args()
    import json
    mod = importlib.import_module(a.module)
    params = json.loads(a.params)
    opps = make_opps(a.opps.split(","))
    for name, fac in opps.items():
        w=d=l=gf=ga=0; worst=0.0; t0=time.time()
        for seed in range(a.seed_start, a.seed_start + a.seeds):
            for side in ("player_1", "player_2"):
                pol = mod.Policy(params=params)
                f, g, mx = play(pol.choose_action, fac(), side, seed)
                gf += f; ga += g; worst = max(worst, mx)
                if f > g: w += 1
                elif f == g: d += 1
                else: l += 1
        n = w + d + l
        print(f"{name:<11} W{w:>3} D{d:>3} L{l:>3}  GF {gf:>3} GA {ga:>3}  score {(w+0.5*d)/n:.2f}  slowest {worst*1000:.0f}ms  ({time.time()-t0:.0f}s)", flush=True)
