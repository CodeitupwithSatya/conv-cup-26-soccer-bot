"""sweep.py: planner vs many opponent styles (both sides, many seeds); appends one JSON line per match."""
import sys, json, time, random, argparse, os
sys.path.insert(0, os.path.abspath("."))

import bench
from my_team.team_bot.policy import Policy
from my_team.team_bot.baseline import direction_toward

ATTACK = {"player_1": "UP", "player_2": "DOWN"}

def _toward(me, tx, ty, dz=1.5):
    return direction_toward(tx - me["x"], ty - me["y"], dz)

def camper_factory():
    def act(obs):
        s = obs["state"]; pid = obs["player_id"]; me = s["players"][pid]; b = s["ball"]; W, H = s["field"]["width"], s["field"]["height"]
        atk = ATTACK[pid]
        if b["possession"] == pid:
            return {"move": "STAY", "kick": {"direction": atk, "power": 3}}
        gy = 8.0 if atk == "UP" else H - 8.0
        gx = min(max(b["x"], 36.0), 64.0)
        return {"move": _toward(me, gx, gy)}
    return act

def random_factory(seed=1):
    rng = random.Random(seed)
    moves = ["STAY","UP","UP_RIGHT","RIGHT","DOWN_RIGHT","DOWN","DOWN_LEFT","LEFT","UP_LEFT"]
    def act(obs):
        pid = obs["player_id"]; b = obs["state"]["ball"]
        a = {"move": rng.choice(moves)}
        if b["possession"] == pid and rng.random() < 0.5:
            a["kick"] = {"direction": rng.choice(moves[1:]), "power": rng.randint(1, 3)}
        return a
    return act

def mirror_factory():
    def act(obs):
        s = obs["state"]; pid = obs["player_id"]; oid = obs["opponent_id"]
        me = s["players"][pid]; op = s["players"][oid]; b = s["ball"]; W, H = s["field"]["width"], s["field"]["height"]
        atk = ATTACK[pid]
        if b["possession"] == pid:
            if abs(me["y"] - (H if atk == "UP" else 0)) < 70:
                return {"move": atk, "kick": {"direction": atk, "power": 3}}
            return {"move": atk}
        near = ((b["x"] - me["x"]) ** 2 + (b["y"] - me["y"]) ** 2) ** 0.5 < 25
        if near or b["possession"] is None:
            return {"move": _toward(me, b["x"], b["y"], 0.5)}
        ty = H * 0.25 if pid == "player_1" else H * 0.75
        return {"move": _toward(me, W - op["x"], ty)}
    return act

def make(name, seed):
    if name in ("practice", "aggressive", "counter", "rl"):
        return bench.make_opps([name])[name]()
    if name == "camper": return camper_factory()
    if name == "random": return random_factory(seed)
    if name == "mirror": return mirror_factory()
    if name == "self": return Policy().choose_action
    raise SystemExit(name)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--opps", default="camper,random,mirror,self,practice,aggressive,counter,rl")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed-start", type=int, default=9100)
    ap.add_argument("--out", default="sweep.jsonl")
    a = ap.parse_args()
    for name in a.opps.split(","):
        for seed in range(a.seed_start, a.seed_start + a.seeds):
            for side in ("player_1", "player_2"):
                t = time.time()
                f, g, mx = bench.play(Policy().choose_action, make(name, seed), side, seed)
                with open(a.out, "a") as fh:
                    fh.write(json.dumps({"opp": name, "seed": seed, "side": side, "gf": f, "ga": g, "ms": round(mx*1000), "sec": round(time.time()-t, 1)}) + "\n")
