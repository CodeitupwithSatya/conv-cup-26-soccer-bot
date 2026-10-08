import sys
sys.path.insert(0, "/home/claude/work")
import bench, sweep
from my_team.team_bot.policy import Policy
jobs = [("camper", 9100, "player_1"), ("camper", 9100, "player_2"), ("camper", 9101, "player_1"), ("camper", 9102, "player_2")]

for name, seed, side in jobs:
    f, g, mx = bench.play(Policy().choose_action, sweep.make(name, seed), side, seed)
    print(name, seed, side, f'{f}-{g}', f'{mx*1000:.0f}ms', flush=True)
