import sys, os


sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("my_team"))

import bench, sweep
from my_team.team_bot.policy import Policy
from my_team.team_bot import planner as P
from my_team.team_bot.sim import Sim, MOVE_NAMES

seed, side, stop = 9100, "player_1", int(sys.argv[1])
pol = Policy(); theirs = sweep.make("camper", seed)
env = bench.SoccerEnv(bench.config); obs = env.reset(seed=seed)
log = []
while not env.done and env.iteration < stop:
    a = pol.choose_action(obs[side]); b = theirs(obs["player_2"])
    log.append(a["move"])
    obs, _ = env.step(a, b)
print("last 8 moves:", log[-8:])
o = obs[side]; sim = Sim.from_observation(o, pol._cfg); me = 0
print("state it", sim.it, "me", (sim.x0, sim.y0), "opp", (sim.x1, sim.y1), "ball", (sim.bx, sim.by), "poss", sim.poss)
pl = pol._planner; w = pl.weights(); print("model weights", [round(x, 2) for x in w], "scores", [round(s, 1) for s in pl.scores])
rows = []
for c in P.candidate_actions(sim, me):
    vals = [P.rollout(sim, me, c, k, pl.params) for k in range(len(w))]
    mean = sum(x * v for x, v in zip(w, vals)); lam = pl.params["robust_lambda"]
    rows.append(((1 - lam) * mean + lam * min(vals), c, vals))
rows.sort(key=lambda r: -r[0])
for v, c, vals in rows:
    print(f"{MOVE_NAMES[c[0]]:<11} score {v:+.3f}  per-model {[round(x, 2) for x in vals]}")
