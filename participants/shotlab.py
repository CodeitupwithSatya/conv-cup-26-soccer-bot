import sys, os

# Dynamically add the current folder (participants) and my_team to the system path
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("my_team"))

import bench
from my_team.team_bot.sim import Sim, Cfg
from my_team.team_bot import planner as P

env = bench.SoccerEnv(bench.config)
obs = env.reset(seed=9100)
base = Sim.from_observation(obs["player_1"], Cfg.from_observation(obs["player_1"]))
base.obs = ()            # empty field to isolate the keeper geometry
def shot(x0, y0, d, power, kx=50.0):
    s = base.clone()
    s.x0, s.y0 = x0, y0; s.x1, s.y1 = kx, 132.0
    s.bx, s.by = x0, y0; s.poss = 0; s.ps = 4; s.vx = s.vy = s.rem = 0.0
    first = (0, d, power)
    s.step(first, P.keeper_policy(s, 1))
    for _ in range(25):
        if s.s0 or s.s1: break
        s.step((0,0,0), P.keeper_policy(s, 1))
    return s.s0 == 1
print("goals by launch point (UP_RIGHT=2 / UP_LEFT=8 / UP=1), power 2-3, keeper starting at x=50")
for d in (2, 8, 1):
    for y0 in (90, 100, 110, 118):
        row = []
        for x0 in range(6, 95, 6):
            ok = any(shot(x0, y0, d, p) for p in (2, 3))
            row.append("G" if ok else ".")
        print(f"dir {d} y0={y0:>3}: {''.join(row)}   (x0 = 6,12,...,90)")
