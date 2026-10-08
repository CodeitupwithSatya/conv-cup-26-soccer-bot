"""Monte-Carlo style rollout planner built on the exact forward model in sim.py."""
from __future__ import annotations

import math
import time

from .sim import Sim, UNIT, RAW

_hypot = math.hypot

UP, UP_RIGHT, RIGHT, DOWN_RIGHT, DOWN, DOWN_LEFT, LEFT, UP_LEFT = 1, 2, 3, 4, 5, 6, 7, 8
_COMBINE = {(UP, LEFT): UP_LEFT, (UP, RIGHT): UP_RIGHT, (DOWN, LEFT): DOWN_LEFT, (DOWN, RIGHT): DOWN_RIGHT}
STAY_ACTION = (0, 0, 0)


# ---------------------------------------------------------------------------- geometry
def direction_toward(dx, dy, dead_zone=1.0):
    h = 0 if abs(dx) <= dead_zone else (RIGHT if dx > 0 else LEFT)
    v = 0 if abs(dy) <= dead_zone else (UP if dy > 0 else DOWN)
    if h and v:
        return _COMBINE[(v, h)]
    return v or h or 0


def fold(x, lo, hi):
    if lo <= x <= hi:
        return x
    span = hi - lo
    if span <= 0:
        return lo
    t = (x - lo) % (2 * span)
    if t > span:
        t = 2 * span - t
    return lo + t


def ball_at(sim, k):
    """Approximate ball centre after k more steps (walls only; ignores obstacles and players)."""
    c = sim.c
    if sim.poss >= 0 or sim.rem <= 0:
        return sim.bx, sim.by
    vl = _hypot(sim.vx, sim.vy)
    if vl == 0:
        return sim.bx, sim.by
    s = min(c.bs * k, sim.rem)
    x = fold(sim.bx + sim.vx / vl * s, c.br, c.W - c.br)
    y = fold(sim.by + sim.vy / vl * s, c.br, c.H - c.br)
    return x, y


def rest_point(sim):
    return ball_at(sim, 1000)


def safe_move(sim, me, preferred):
    """Port of the organizer helper: nearest unblocked direction (obstacle margin 0.25)."""
    c = sim.c
    x, y = (sim.x0, sim.y0) if me == 0 else (sim.x1, sim.y1)
    pv = RAW[preferred]
    best = None
    best_key = None
    for m in range(1, 9):
        u = UNIT[m]
        nx, ny = x + u[0] * c.ps, y + u[1] * c.ps
        r = c.pr
        if nx - r < 0 or nx + r > c.W or ny - r < 0 or ny + r > c.H:
            continue
        ok = True
        rr = r + 0.25
        rr2 = rr * rr
        for ax, ay, bx, by in sim.obs:
            if nx < ax - rr or nx > bx + rr or ny < ay - rr or ny > by + rr:
                continue
            qx = ax if nx < ax else (bx if nx > bx else nx)
            qy = ay if ny < ay else (by if ny > by else ny)
            if (nx - qx) ** 2 + (ny - qy) ** 2 < rr2:
                ok = False
                break
        if not ok:
            continue
        key = (RAW[m][0] * pv[0] + RAW[m][1] * pv[1], m == preferred)
        if best_key is None or key > best_key:
            best_key = key
            best = m
    return best if best is not None else 0


def move_toward(sim, me, tx, ty, stop_radius=0.0):
    """Valid move (engine validity) that gets closest to a target point."""
    c = sim.c
    x, y = (sim.x0, sim.y0) if me == 0 else (sim.x1, sim.y1)
    best = 0
    best_d = _hypot(tx - x, ty - y)
    if best_d <= stop_radius:
        return 0
    for m in range(1, 9):
        u = UNIT[m]
        nx, ny = x + u[0] * c.ps, y + u[1] * c.ps
        d = _hypot(tx - nx, ty - ny)
        if d < best_d - 1e-9 and sim.valid(nx, ny):
            best_d = d
            best = m
    return best


def prog(sim, me, y):
    """0 at my own goal line, 1 at the goal I attack."""
    return y / sim.c.H if me == 0 else 1.0 - y / sim.c.H


# ---------------------------------------------------------------------------- organizer-style models
def _pos(sim, me):
    return (sim.x0, sim.y0, sim.x1, sim.y1) if me == 0 else (sim.x1, sim.y1, sim.x0, sim.y0)


def practice_policy(sim, me):
    c = sim.c
    mx, my, ox, oy = _pos(sim, me)
    sign = 1 if me == 0 else -1
    attack = UP if me == 0 else DOWN
    if sim.poss == me:
        dg = c.H - my if sign > 0 else my
        od = _hypot(ox - mx, oy - my)
        ahead = sign * (oy - my) > 0 and abs(ox - mx) < 12 and od < 55
        side = LEFT if ox >= mx else RIGHT
        dribble = _COMBINE[(attack, side)] if ahead else attack
        move = safe_move(sim, me, dribble)
        if dg > 50 and sim.ps < 3:
            return (move, 0, 0)
        kd = dribble if ahead else direction_toward(c.W / 2 - mx, sign * dg, 6.0)
        return (move, kd, 3 if dg > 30 else 1)
    tx, ty = sim.bx, sim.by
    if sim.rem > 0:
        tx += sim.vx
        ty += sim.vy
    elif sim.poss == 1 - me:
        ty += (-1 if attack == UP else 1) * 3.0
        tx += -3.0 if ox > c.W / 2 else 3.0
    return (safe_move(sim, me, direction_toward(tx - mx, ty - my, 0.6)), 0, 0)


def aggressive_policy(sim, me):
    c = sim.c
    mx, my, ox, oy = _pos(sim, me)
    sign = 1 if me == 0 else -1
    attack = UP if me == 0 else DOWN
    if sim.poss == me:
        in_lane = sign * (oy - my) > 0 and abs(ox - mx) < 14
        if in_lane:
            d = _COMBINE[(attack, LEFT if ox >= mx else RIGHT)]
        else:
            d = direction_toward(c.W / 2 - mx, sign * c.H, 5.0)
        return (safe_move(sim, me, d), d, 3)
    tx, ty = sim.bx, sim.by
    if sim.rem > 0:
        tx += 2.0 * sim.vx
        ty += 2.0 * sim.vy
    return (safe_move(sim, me, direction_toward(tx - mx, ty - my, 0.25)), 0, 0)


def counter_policy(sim, me):
    c = sim.c
    mx, my, ox, oy = _pos(sim, me)
    sign = 1 if me == 0 else -1
    attack = UP if me == 0 else DOWN
    if sim.poss == me:
        side = RIGHT if mx <= c.W / 2 else LEFT
        brk = _COMBINE[(attack, side)]
        if sim.ps < 2:
            return (safe_move(sim, me, brk), 0, 0)
        gd = direction_toward(c.W / 2 - mx, sign * c.H, 4.0)
        return (safe_move(sim, me, brk), gd, 3)
    if sim.poss == 1 - me:
        if _hypot(ox - mx, oy - my) < 32:
            tx, ty = ox, oy
        else:
            own_goal_y = 0.0 if me == 0 else c.H
            tx, ty = (ox + c.W / 2) / 2, (oy + own_goal_y) / 2
    else:
        tx, ty = sim.bx, sim.by
        if sim.rem > 0:
            tx += sim.vx
            ty += sim.vy
    return (safe_move(sim, me, direction_toward(tx - mx, ty - my, 0.6)), 0, 0)


# ---------------------------------------------------------------------------- my continuation policy
def _path_clear(sim, sx, sy, d, dist_total, ox, oy, margin=0.0):
    """Could an opponent at (ox, oy) intercept a ball launched from (sx, sy)? (walls only, optimistic for opp)"""
    c = sim.c
    u0, u1 = UNIT[d]
    reach = c.br + c.pr + margin
    lo_x, hi_x, lo_y, hi_y = c.br, c.W - c.br, c.br, c.H - c.br
    s = 3.0
    bs = c.bs
    ps = c.ps
    while s <= dist_total:
        x = sx + u0 * s
        y = sy + u1 * s
        if x < lo_x or x > hi_x:
            x = fold(x, lo_x, hi_x)
        if y < lo_y or y > hi_y:
            y = fold(y, lo_y, hi_y)
        k = int((s + bs - 1e-9) // bs)
        r = reach + ps * (k if k > 1 else 1)
        if (x - ox) * (x - ox) + (y - oy) * (y - oy) <= r * r:
            return False
        s += 3.0
    return True


def goal_kick_options(sim, me, mx, my):
    """Analytic straight/bank shots that reach my attacking goal mouth (no obstacle check)."""
    c = sim.c
    out = []
    goal_up = me == 0
    clearance = c.pr + c.br + 0.05
    for d in range(1, 9):
        u = UNIT[d]
        if goal_up and u[1] <= 0.1:
            continue
        if not goal_up and u[1] >= -0.1:
            continue
        sx, sy = mx + u[0] * clearance, my + u[1] * clearance
        t = ((c.H - c.br - sy) / u[1]) if goal_up else ((c.br - sy) / u[1])
        if t < 0:
            continue
        x = fold(sx + u[0] * t, c.br, c.W - c.br)
        if not (c.gl + 1.0 <= x <= c.gr - 1.0):
            continue
        for p in (1, 2, 3):
            if c.kd[p - 1] >= t + 0.5:
                out.append((d, p, t, sx, sy))
                break
    return out


def _obstacle_blocked(sim, sx, sy, d, dist_total):
    c = sim.c
    u0, u1 = UNIT[d]
    br = c.br
    lo_x, hi_x, lo_y, hi_y = br, c.W - br, br, c.H - br
    lim = br + 0.3
    lim2 = lim * lim
    s = 0.0
    obs = sim.obs
    while s <= dist_total:
        x = sx + u0 * s
        y = sy + u1 * s
        if x < lo_x or x > hi_x:
            x = fold(x, lo_x, hi_x)
        if y < lo_y or y > hi_y:
            y = fold(y, lo_y, hi_y)
        for ax, ay, bx, by in obs:
            if x < ax - lim or x > bx + lim or y < ay - lim or y > by + lim:
                continue
            qx = ax if x < ax else (bx if x > bx else x)
            qy = ay if y < ay else (by if y > by else y)
            if (x - qx) ** 2 + (y - qy) ** 2 < lim2:
                return True
        s += 2.5
    return False


def exact_shot(sim, me, opp_fn, ctx):
    """Fire only if an exact simulation against the modelled opponent says the kick scores."""
    c = sim.c
    mx, my, ox, oy = _pos(sim, me)
    if abs((c.H if me == 0 else 0.0) - my) > 80.0 or ctx["count"][0] > ctx["limit"]:
        return None
    opp = 1 - me
    for d, p_, t, sx, sy in goal_kick_options(sim, me, mx, my):
        for power in range(p_, 4):
            key = (int(mx), int(my), int(ox), int(oy), sim.ps >= 3, d, power, ctx["model"])
            hit = ctx["cache"].get(key)
            if hit is None:
                ctx["count"][0] += 1
                s = sim.clone()
                base0, base1 = s.s0, s.s1
                kick = (0, d, power)
                if me == 0:
                    s.step(kick, opp_fn(s, opp))
                else:
                    s.step(opp_fn(s, opp), kick)
                for _ in range(14):
                    if s.s0 != base0 or s.s1 != base1 or s.done:
                        break
                    if me == 0:
                        s.step((0, 0, 0), opp_fn(s, opp))
                    else:
                        s.step(opp_fn(s, opp), (0, 0, 0))
                hit = (s.s0 > base0) if me == 0 else (s.s1 > base1)
                ctx["cache"][key] = hit
            if hit:
                return (0, d, power)
    return None


def launch_dribble(sim, me, mx, my):
    """Walk along the diagonal 'shot line' that ends at the middle of the goal mouth."""
    c = sim.c
    sign = 1 if me == 0 else -1
    yg = c.H - c.br if me == 0 else c.br
    L_me = sign * (yg - my)
    dxs = 1.0 if mx < c.W / 2 else -1.0
    lt = max(18.0, min(L_me, 46.0) - 8.0)
    tx = min(max(c.W / 2 - dxs * lt, 8.0), c.W - 8.0)
    ty = yg - sign * lt
    return move_toward(sim, me, tx, ty)


def mine_policy(sim, me, opp_fn=None, ctx=None):
    c = sim.c
    mx, my, ox, oy = _pos(sim, me)
    sign = 1 if me == 0 else -1
    attack = UP if me == 0 else DOWN
    own_goal_y = 0.0 if me == 0 else c.H
    if sim.poss == me:
        if opp_fn is not None and ctx is not None:
            shot = exact_shot(sim, me, opp_fn, ctx)
            if shot is not None:
                return shot
        elif abs(c.H - my if me == 0 else my) <= c.kd[2] + 6.0:
            for d, p, t, sx, sy in goal_kick_options(sim, me, mx, my):
                if _path_clear(sim, sx, sy, d, t, ox, oy) and not _obstacle_blocked(sim, sx, sy, d, t):
                    return (0, d, p)
        od = _hypot(ox - mx, oy - my)
        under_threat = sim.ps >= 2 and od < 15.0
        if sim.ps >= c.plim - 2 or under_threat:
            best = None
            best_score = -1e9
            clearance = c.pr + c.br + 0.05
            for d in range(1, 9):
                u = UNIT[d]
                if u[1] * sign < 0.5:
                    continue
                sx, sy = mx + u[0] * clearance, my + u[1] * clearance
                for p in (2, 3):
                    L = c.kd[p - 1]
                    rx = fold(sx + u[0] * L, c.br, c.W - c.br)
                    ry = fold(sy + u[1] * L, c.br, c.H - c.br)
                    dm = _hypot(rx - mx, ry - my)
                    do = _hypot(rx - ox, ry - oy)
                    score = 0.6 * prog(sim, me, ry) + 0.4 * max(-1.0, min(1.0, (do - dm) / 25.0))
                    if not _path_clear(sim, sx, sy, d, L, ox, oy, 0.0):
                        score -= 0.25
                    if _obstacle_blocked(sim, sx, sy, d, L):
                        score -= 0.3
                    if score > best_score:
                        best_score = score
                        best = (d, p)
            return (safe_move(sim, me, best[0]), best[0], best[1])
        if opp_fn is not None and ctx is not None and not under_threat:
            m = launch_dribble(sim, me, mx, my)
            if m:
                return (m, 0, 0)
        ahead = sign * (oy - my) > 0 and abs(ox - mx) < 14 and od < 40
        if ahead:
            pref = _COMBINE[(attack, LEFT if ox >= mx else RIGHT)]
        else:
            pref = direction_toward(c.W / 2 - mx, sign * 50.0, 8.0) or attack
            if pref in (LEFT, RIGHT):
                pref = _COMBINE[(attack, pref)]
        return (safe_move(sim, me, pref), 0, 0)
    if sim.poss == 1 - me:
        d = _hypot(ox - mx, oy - my)
        if d < 14.0:
            return (move_toward(sim, me, ox, oy), 0, 0)
        gx, gy = c.W / 2, own_goal_y
        vx, vy = gx - ox, gy - oy
        vl = _hypot(vx, vy) or 1.0
        off = 9.0
        return (move_toward(sim, me, ox + vx / vl * off, oy + vy / vl * off), 0, 0)
    if sim.rem > 0:
        reach = c.br + c.pr
        vl = _hypot(sim.vx, sim.vy) or 1.0
        ux, uy = sim.vx / vl, sim.vy / vl
        lo_x, hi_x, lo_y, hi_y = c.br, c.W - c.br, c.br, c.H - c.br
        px = py = None
        for k in range(1, 14):
            s = min(c.bs * k, sim.rem)
            px = fold(sim.bx + ux * s, lo_x, hi_x)
            py = fold(sim.by + uy * s, lo_y, hi_y)
            if _hypot(px - mx, py - my) - reach <= c.ps * k:
                return (move_toward(sim, me, px, py), 0, 0)
            if s >= sim.rem:
                break
        return (move_toward(sim, me, px, py), 0, 0)
    return (move_toward(sim, me, sim.bx, sim.by, 0.0), 0, 0)


def keeper_policy(sim, me):
    """Passive goal-line defender: tracks the ball's x in front of its own goal, clears when it has the ball."""
    c = sim.c
    if sim.poss == me:
        return (0, UP if me == 0 else DOWN, 3)
    gy = 8.0 if me == 0 else c.H - 8.0
    gx = min(max(sim.bx, 36.0), 64.0)
    return (move_toward(sim, me, gx, gy, 1.5), 0, 0)


MODEL_FUNCS = (practice_policy, aggressive_policy, counter_policy, mine_policy, keeper_policy)


# ---------------------------------------------------------------------------- evaluation
def leaf_value(sim, me, params):
    c = sim.c
    opp = 1 - me
    mx, my, ox, oy = _pos(sim, me)
    if sim.poss == me:
        p = prog(sim, me, sim.by)
        od = _hypot(ox - mx, oy - my)
        return params["poss_base"] + params["poss_prog"] * p + 0.08 * min(1.0, od / 30.0)
    if sim.poss == opp:
        p = prog(sim, me, sim.by)
        od = _hypot(ox - mx, oy - my)
        return -params["poss_base"] - params["poss_prog"] * (1.0 - p) - 0.04 * max(0.0, 1.0 - od / 30.0)
    rx, ry = rest_point(sim)
    reach = c.prad
    tm = max(0.0, _hypot(rx - mx, ry - my) - reach) / c.ps
    to = max(0.0, _hypot(rx - ox, ry - oy) - reach) / c.ps
    adv = max(-12.0, min(12.0, to - tm)) / 12.0
    p = prog(sim, me, ry)
    return (params["loose_adv"] * adv + params["loose_prog"] * (2.0 * p - 1.0)
            - params["loose_time"] * min(tm, 25.0))


DEFAULT_PARAMS = {
    "horizon": 18,
    "poss_base": 0.12,
    "poss_prog": 0.45,
    "loose_adv": 0.35,
    "loose_prog": 0.15,
    "loose_time": 0.008,
    "stick_bonus": 0.01,
    "tempo": 0.01,
    "fetch_margin": -3.0,
    "shot_checks": 700,
    "goal_value": 1.0,
    "time_decay": 0.004,
    "robust_lambda": 0.3,
    "screen_models": 2,
    "screen_keep": 8,
}


def rollout(sim0, me, first_action, opp_model, params, ctx=None):
    sim = sim0.clone()
    opp = 1 - me
    horizon = params["horizon"]
    model = MODEL_FUNCS[opp_model]
    mine = mine_policy
    if ctx is not None:
        ctx = dict(ctx, model=opp_model)
    base_s0, base_s1 = sim.s0, sim.s1
    tempo = params["tempo"]
    acc = 0.0
    oa = model(sim, opp)
    if me == 0:
        sim.step(first_action, oa)
    else:
        sim.step(oa, first_action)
    if sim.poss == me:
        acc += tempo
    elif sim.poss == opp:
        acc -= tempo
    t = 1
    while t < horizon:
        if sim.s0 != base_s0 or sim.s1 != base_s1 or sim.done:
            break
        a_me = mine(sim, me, model, ctx) if ctx is not None else mine(sim, me)
        a_op = model(sim, opp)
        if me == 0:
            sim.step(a_me, a_op)
        else:
            sim.step(a_op, a_me)
        t += 1
        if sim.poss == me:
            acc += tempo
        elif sim.poss == opp:
            acc -= tempo
    my_goals = (sim.s0 - base_s0) if me == 0 else (sim.s1 - base_s1)
    opp_goals = (sim.s1 - base_s1) if me == 0 else (sim.s0 - base_s0)
    if my_goals != opp_goals:
        sign = 1.0 if my_goals > opp_goals else -1.0
        return sign * (params["goal_value"] - params["time_decay"] * t)
    return leaf_value(sim, me, params) + acc


def candidate_actions(sim, me):
    c = sim.c
    mx, my, ox, oy = _pos(sim, me)
    cands = []
    for m in range(9):
        if m == 0:
            cands.append((0, 0, 0))
            continue
        u = UNIT[m]
        if sim.valid(mx + u[0] * c.ps, my + u[1] * c.ps):
            cands.append((m, 0, 0))
    if sim.poss == me:
        for d in range(1, 9):
            for p in (1, 2, 3):
                cands.append((0, d, p))
                u = UNIT[d]
                if sim.valid(mx + u[0] * c.ps, my + u[1] * c.ps):
                    cands.append((d, d, p))
    elif sim.poss == 1 - me and sim.ps >= 2 and _hypot(ox - mx, oy - my) < 17.0:
        # a tackle followed by an immediate kick is legal in the engine
        toward = [m for m in range(1, 9) if sim.valid(mx + UNIT[m][0] * c.ps, my + UNIT[m][1] * c.ps)]
        toward.sort(key=lambda m: _hypot(ox - (mx + UNIT[m][0] * c.ps), oy - (my + UNIT[m][1] * c.ps)))
        attack = UP if me == 0 else DOWN
        for m in toward[:2]:
            for d in (attack, _COMBINE[(attack, LEFT)], _COMBINE[(attack, RIGHT)]):
                for p in (2, 3):
                    cands.append((m, d, p))
    return cands


class Planner:
    def __init__(self, params=None):
        self.params = dict(DEFAULT_PARAMS)
        if params:
            self.params.update(params)
        self.reset()

    def reset(self):
        self.scores = [0.0] * len(MODEL_FUNCS)
        self.prev_sim = None
        self.last_info = {}
        self.prev_action = None

    # --- online opponent modelling: which scripted style explains the opponent's moves?
    def observe(self, sim, me):
        prev = self.prev_sim
        if prev is not None and prev.it + 1 == sim.it and prev.s0 + prev.s1 == sim.s0 + sim.s1:
            opp = 1 - me
            ox0, oy0 = (prev.x0, prev.y0) if opp == 0 else (prev.x1, prev.y1)
            ox1, oy1 = (sim.x0, sim.y0) if opp == 0 else (sim.x1, sim.y1)
            dx, dy = ox1 - ox0, oy1 - oy0
            if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                actual = 0
            else:
                best, bd = 0, -2.0
                n = _hypot(dx, dy)
                for m in range(1, 9):
                    dot = (UNIT[m][0] * dx + UNIT[m][1] * dy) / n
                    if dot > bd:
                        bd, best = dot, m
                actual = best if bd > 0.9 else -1
            if actual >= 0:
                for k, fn in enumerate(MODEL_FUNCS):
                    pred = fn(prev, opp)[0]
                    self.scores[k] = 0.93 * self.scores[k] + (1.0 if pred == actual else 0.0)

    def weights(self):
        beta = 0.6
        m = max(self.scores)
        ex = [math.exp(beta * (s - m)) for s in self.scores]
        z = sum(ex)
        w = [max(0.08, e / z) for e in ex]
        wz = sum(w)
        return [x / wz for x in w]

    @staticmethod
    def _boomerang(sim, me, action):
        s = sim.clone()
        stay = (0, 0, 0)
        ox, oy = sim.bx, sim.by
        if me == 0:
            s.step(action, stay)
        else:
            s.step(stay, action)
        for _ in range(3):
            if s.poss == me:
                return _hypot(s.bx - ox, s.by - oy) < 15.0
            if s.poss >= 0 or s.done:
                return False
            s.step(stay, stay)
        return s.poss == me and _hypot(s.bx - ox, s.by - oy) < 15.0

    def plan(self, sim, me, budget=0.9):
        t0 = time.perf_counter()
        self.observe(sim, me)
        self.prev_sim = sim
        params = self.params
        if sim.poss < 0 and sim.rem <= 0 and params["fetch_margin"] is not None:
            # loose, stationary ball and I am clearly closer: just go and get it (avoids a horizon artefact
            # where waiting looks better than fetching because the rollout ends before the payoff)
            mx, my, ox, oy = _pos(sim, me)
            c = sim.c
            tm = max(0.0, _hypot(sim.bx - mx, sim.by - my) - c.prad) / c.ps
            to = max(0.0, _hypot(sim.bx - ox, sim.by - oy) - c.prad) / c.ps
            if to - tm >= params["fetch_margin"]:
                m = move_toward(sim, me, sim.bx, sim.by)
                if m:
                    self.prev_action = (m, 0, 0)
                    return self.prev_action
        ctx = {"cache": {}, "count": [0], "limit": params["shot_checks"], "model": 0}
        weights = self.weights()
        cands = candidate_actions(sim, me)
        if sim.poss == me:
            # drop "boomerang" kicks: the ball bounces straight back to me, resetting possession
            # (the engine's possession timeout never fires) and wasting the clock
            kept = [a for a in cands if not (a[1] and self._boomerang(sim, me, a))]
            if kept:
                cands = kept
        lam = params["robust_lambda"]
        nm = len(MODEL_FUNCS)
        order = sorted(range(nm), key=lambda k: -weights[k])
        screen_models = order[: params["screen_models"]]
        keep = params["screen_keep"]
        # stage 1: cheap screen of every candidate against the most likely model(s)
        if len(cands) > keep:
            scored = []
            for cand in cands:
                v = sum(rollout(sim, me, cand, k, params, ctx) for k in screen_models) / len(screen_models)
                scored.append((v, cand))
                if time.perf_counter() - t0 > budget * 0.5:
                    break
            scored.sort(key=lambda item: -item[0])
            cands = [cand for _, cand in scored[:keep]]
        # stage 2: full evaluation of the survivors against every model
        best = None
        best_val = -1e9
        for cand in cands:
            vals = [rollout(sim, me, cand, k, params, ctx) for k in range(nm)]
            mean = sum(w * v for w, v in zip(weights, vals))
            val = (1 - lam) * mean + lam * min(vals)
            if cand == self.prev_action:
                val += params["stick_bonus"]  # hysteresis: break ties without dithering
            if val > best_val:
                best_val = val
                best = cand
            if time.perf_counter() - t0 > budget:
                break
        self.last_info = {"value": best_val, "weights": weights, "n": len(cands)}
        self.prev_action = best
        return best if best is not None else STAY_ACTION
