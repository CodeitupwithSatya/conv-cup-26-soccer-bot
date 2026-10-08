import sys,os
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("my_team"))
import bench, sweep
from my_team.team_bot.policy import Policy
opp_name, seed, side = sys.argv[1], int(sys.argv[2]), sys.argv[3]
fac = lambda: sweep.make(opp_name, seed)
pol = Policy(); theirs = fac()
env = bench.SoccerEnv(bench.config); obs = env.reset(seed=seed)
other = "player_2" if side == "player_1" else "player_1"
print("obstacles:", [(round(o.x),round(o.y),round(o.width),round(o.height)) for o in env.obstacles])
i = 0
while not env.done:
    a = pol.choose_action(obs[side]); b = theirs(obs[other])
    acts = {side: a, other: b}
    obs, info = env.step(acts["player_1"], acts["player_2"])
    if i % 10 == 0 or info["events"] and any(e["type"] in ("goal","tackle","drop_ball","possession_timeout") for e in info["events"]):
        s = obs[side]["state"]; me = s["players"][side]; op = s["players"][other]; bl = s["ball"]
        print(f"it{s['iteration']:>3} me({me['x']:.0f},{me['y']:.0f}) opp({op['x']:.0f},{op['y']:.0f}) ball({bl['x']:.0f},{bl['y']:.0f}) poss={bl['possession']} st={bl['status']} act={a.get('move')}{'+K' if 'kick' in a else ''} ev={[e['type'] for e in info['events']]}")
    i += 1
print("final", env.score)
