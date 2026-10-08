"""Check that team_bot.sim reproduces the real engine exactly (random + scripted play)."""
import sys, random, importlib
import os

# Dynamically add the current folder (participants) and my_team to the system path
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("my_team"))

from soccer_env import GameConfig, SoccerEnv
from soccer_env.bots import practice_action, aggressive_action, counter_action
from my_team.team_bot.sim import Sim, Cfg, MOVE_NAMES, NAME_TO_IDX

# Use a relative path to load the config file
config = GameConfig.from_json("config/game.json")

def to_tuple(a):
    mv = NAME_TO_IDX[a.get("move", "STAY")]
    k = a.get("kick")
    if k:
        return (mv, NAME_TO_IDX[k["direction"]], k["power"])
    return (mv, 0, 0)

def rnd_action(rng, obs):
    a = {"move": rng.choice(MOVE_NAMES)}
    if rng.random() < 0.35:
        a["kick"] = {"direction": rng.choice(MOVE_NAMES[1:]), "power": rng.randint(1, 3)}
    return a

def mixed(rng, style):
    def f(obs):
        if style == 0: return rnd_action(rng, obs)
        if style == 1: base = practice_action(obs)
        elif style == 2: base = aggressive_action(obs)
        else: base = counter_action(obs)
        if rng.random() < 0.15: return rnd_action(rng, obs)
        if rng.random() < 0.3 and "kick" not in base:
            base = dict(base); base["kick"] = {"direction": rng.choice(MOVE_NAMES[1:]), "power": rng.randint(1,3)}
        return base
    return f

def snapshot_env(env):
    pk = env.possession
    return (env.players["player_1"][0], env.players["player_1"][1], env.players["player_2"][0], env.players["player_2"][1],
            env.ball_position[0], env.ball_position[1], env.ball_velocity[0], env.ball_velocity[1], env.ball_remaining_distance,
            -1 if pk is None else (0 if pk == "player_1" else 1), env.possession_steps, env.loose_ball_steps,
            env.score["player_1"], env.score["player_2"], env.iteration, env.done)

def snapshot_sim(s):
    return (s.x0, s.y0, s.x1, s.y1, s.bx, s.by, s.vx, s.vy, s.rem, s.poss, s.ps, s.ls, s.s0, s.s1, s.it, s.done)

bad = 0; steps = 0; matches = 0
events = {}
for seed in range(int(sys.argv[1]) if len(sys.argv) > 1 else 60):
    for style in range(4):
        rng = random.Random(seed * 10 + style)
        env = SoccerEnv(config)
        obs = env.reset(seed=seed)
        cfg = Cfg()
        sim = Sim.from_observation(obs["player_1"], cfg)
        p1 = mixed(rng, style); p2 = mixed(rng, (style + seed) % 4)
        matches += 1
        while not env.done:
            a1, a2 = p1(obs["player_1"]), p2(obs["player_2"])
            obs, info = env.step(a1, a2)
            for e in info["events"]:
                events[e["type"]] = events.get(e["type"], 0) + 1
            sim.step(to_tuple(env.normalize_action(a1)), to_tuple(env.normalize_action(a2)))
            steps += 1
            e_, s_ = snapshot_env(env), snapshot_sim(sim)
            # ball position after goal is irrelevant: compare everything
            if any(abs(float(x) - float(y)) > 1e-6 for x, y in zip(e_, s_)):
                bad += 1
                if bad <= 5:
                    print("MISMATCH seed", seed, "style", style, "it", env.iteration, "\n env", e_, "\n sim", s_, info["events"])
                break
print("matches", matches, "steps", steps, "mismatches", bad)
print("event coverage:", dict(sorted(events.items())))
