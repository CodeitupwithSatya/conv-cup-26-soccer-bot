"""Lean forward model of the AI Soccer engine (a line-by-line port of SoccerEnv.step).

It needs only the standard library, so it can ship inside the submission.
Players are indexed 0 (player_1: defends bottom, attacks UP) and 1 (player_2: attacks DOWN).
An action is a tuple (move, kick_dir, kick_power): move is 0..8 (STAY, UP, UP_RIGHT, ...),
kick_dir 0 means "no kick", otherwise 1..8 (same index table), kick_power is 1..3.
"""
from __future__ import annotations

import math

MOVE_NAMES = ("STAY", "UP", "UP_RIGHT", "RIGHT", "DOWN_RIGHT", "DOWN", "DOWN_LEFT", "LEFT", "UP_LEFT")
RAW = ((0, 0), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1))
UNIT = tuple(
    (0.0, 0.0) if (vx == 0 and vy == 0) else (vx / math.hypot(vx, vy), vy / math.hypot(vx, vy))
    for vx, vy in RAW
)
NAME_TO_IDX = {name: i for i, name in enumerate(MOVE_NAMES)}
_hypot = math.hypot
_ceil = math.ceil


class Cfg:
    __slots__ = ("W", "H", "GW", "pr", "ps", "br", "bs", "prad", "kd", "plim", "llim", "maxit", "maxg",
                 "gl", "gr", "start0", "start1")

    def __init__(self, W=100.0, H=140.0, GW=36.0, pr=3.0, ps=4.0, br=1.5, bs=8.0, prad=5.0,
                 kd=(32.0, 64.0, 96.0), plim=10, llim=20, maxit=400, maxg=7):
        self.W, self.H, self.GW, self.pr, self.ps = float(W), float(H), float(GW), float(pr), float(ps)
        self.br, self.bs, self.prad = float(br), float(bs), float(prad)
        self.kd = tuple(float(k) for k in kd)
        self.plim, self.llim, self.maxit, self.maxg = int(plim), int(llim), int(maxit), int(maxg)
        self.gl = (self.W - self.GW) / 2
        self.gr = self.gl + self.GW
        self.start0 = (self.W / 2, self.H * 0.25)
        self.start1 = (self.W / 2, self.H * 0.75)

    @classmethod
    def from_observation(cls, obs, overrides=None):
        field = obs["state"]["field"]
        kwargs = dict(
            W=field["width"], H=field["height"], GW=field["goal_width"],
            pr=field.get("player_radius", 3.0), ps=field.get("player_speed", 4.0),
            maxit=obs["state"].get("maximum_iterations", 400),
        )
        for key, value in (overrides or {}).items():
            kwargs[key] = value
        return cls(**kwargs)


class Sim:
    __slots__ = ("c", "obs", "x0", "y0", "x1", "y1", "bx", "by", "vx", "vy", "rem", "poss", "ps",
                 "ls", "lb", "s0", "s1", "it", "done")

    @classmethod
    def from_observation(cls, observation, cfg):
        state = observation["state"]
        sim = cls.__new__(cls)
        sim.c = cfg
        sim.obs = tuple((o["x"], o["y"], o["x"] + o["width"], o["y"] + o["height"]) for o in state.get("obstacles", []))
        p = state["players"]
        sim.x0, sim.y0 = p["player_1"]["x"], p["player_1"]["y"]
        sim.x1, sim.y1 = p["player_2"]["x"], p["player_2"]["y"]
        b = state["ball"]
        sim.bx, sim.by = b["x"], b["y"]
        v = b.get("velocity") or {}
        sim.vx, sim.vy = float(v.get("x", 0.0)), float(v.get("y", 0.0))
        sim.rem = float(b.get("remaining_kick_distance", 0.0))
        poss = b.get("possession")
        sim.poss = -1 if poss is None else (0 if poss == "player_1" else 1)
        sim.ps = int(b.get("possession_steps", 0))
        sim.ls = int(b.get("loose_ball_steps", 0))
        sim.lb = -1.0
        if sim.ls > 0 and sim.poss < 0 and sim.rem <= 0:
            sim.lb = min(_hypot(sim.x0 - sim.bx, sim.y0 - sim.by), _hypot(sim.x1 - sim.bx, sim.y1 - sim.by))
        sim.s0, sim.s1 = state["score"]["player_1"], state["score"]["player_2"]
        sim.it = int(state["iteration"])
        sim.done = bool(state.get("done", False))
        return sim

    def clone(self):
        n = Sim.__new__(Sim)
        n.c = self.c
        n.obs = self.obs
        n.x0, n.y0, n.x1, n.y1 = self.x0, self.y0, self.x1, self.y1
        n.bx, n.by, n.vx, n.vy, n.rem = self.bx, self.by, self.vx, self.vy, self.rem
        n.poss, n.ps, n.ls, n.lb = self.poss, self.ps, self.ls, self.lb
        n.s0, n.s1, n.it, n.done = self.s0, self.s1, self.it, self.done
        return n

    # ------------------------------------------------------------------ helpers
    def valid(self, x, y):
        c = self.c
        r = c.pr
        if x - r < 0 or x + r > c.W or y - r < 0 or y + r > c.H:
            return False
        rr = r * r
        for ax, ay, bx, by in self.obs:
            cx = ax if x < ax else (bx if x > bx else x)
            cy = ay if y < ay else (by if y > by else y)
            dx = x - cx
            dy = y - cy
            if dx * dx + dy * dy < rr:
                return False
        return True

    def _restart(self, possessor):
        c = self.c
        self.x0, self.y0 = c.start0
        self.x1, self.y1 = c.start1
        self.poss = possessor
        self.ps = 0
        self.ls = 0
        self.lb = -1.0
        if possessor == 0:
            self.bx, self.by = self.x0, self.y0
        else:
            self.bx, self.by = self.x1, self.y1
        self.vx = self.vy = 0.0
        self.rem = 0.0

    # ------------------------------------------------------------------ player movement
    def _move_players(self, m0, m1):
        sp = self.c.ps
        ox0, oy0, ox1, oy1 = self.x0, self.y0, self.x1, self.y1
        u = UNIT[m0]
        c0x, c0y = ox0 + u[0] * sp, oy0 + u[1] * sp
        if m0 and self.valid(c0x, c0y):
            p0x, p0y = c0x, c0y
        else:
            p0x, p0y = ox0, oy0
        u = UNIT[m1]
        c1x, c1y = ox1 + u[0] * sp, oy1 + u[1] * sp
        if m1 and self.valid(c1x, c1y):
            p1x, p1y = c1x, c1y
        else:
            p1x, p1y = ox1, oy1
        md = 2 * self.c.pr
        if _hypot(p0x - p1x, p0y - p1y) < md:
            p0x, p0y, p1x, p1y = self._resolve_contact(ox0, oy0, ox1, oy1, p0x, p0y, p1x, p1y, m0, m1, md)
        self.x0, self.y0, self.x1, self.y1 = p0x, p0y, p1x, p1y

    def _resolve_contact(self, ox0, oy0, ox1, oy1, p0x, p0y, p1x, p1y, m0, m1, md):
        r = self.c.pr
        sp = self.c.ps
        midx, midy = (p0x + p1x) / 2, (p0y + p1y) / 2
        sx, sy = ox0 - ox1, oy0 - oy1
        ln = _hypot(sx, sy)
        if ln == 0:
            sx, sy = 1.0, 0.0
        else:
            sx, sy = sx / ln, sy / ln
        r0 = (midx + sx * r, midy + sy * r)
        r1 = (midx - sx * r, midy - sy * r)
        moved = _hypot(ox0 - r0[0], oy0 - r0[1]) + _hypot(ox1 - r1[0], oy1 - r1[1])
        if moved > 0.1 and self.valid(*r0) and self.valid(*r1):
            return r0[0], r0[1], r1[0], r1[1]
        bx, by = self.bx, self.by
        olds = ((ox0, oy0), (ox1, oy1))
        props = ((p0x, p0y), (p1x, p1y))
        solo = []
        for i in (0, 1):
            other = olds[1 - i]
            pr_ = props[i]
            if self.valid(*pr_) and _hypot(pr_[0] - other[0], pr_[1] - other[1]) >= md:
                prog = _hypot(olds[i][0] - bx, olds[i][1] - by) - _hypot(pr_[0] - bx, pr_[1] - by)
                solo.append((prog, i))
        if solo:
            _, mover = max(solo)
            mv = m0 if mover == 0 else m1
            u = UNIT[mv]
            cand = (olds[mover][0] + u[0] * sp, olds[mover][1] + u[1] * sp)
            if not self.valid(*cand):
                return ox0, oy0, ox1, oy1
            if mover == 0:
                return cand[0], cand[1], ox1, oy1
            return ox0, oy0, cand[0], cand[1]
        alts = []
        for i in (0, 1):
            other = olds[1 - i]
            for d in range(1, 9):
                u = UNIT[d]
                cand = (olds[i][0] + u[0] * sp, olds[i][1] + u[1] * sp)
                if self.valid(*cand) and _hypot(cand[0] - other[0], cand[1] - other[1]) >= md:
                    prog = _hypot(olds[i][0] - bx, olds[i][1] - by) - _hypot(cand[0] - bx, cand[1] - by)
                    alts.append((prog, i, cand))
        if alts:
            _, mover, cand = max(alts)
            if mover == 0:
                return cand[0], cand[1], ox1, oy1
            return ox0, oy0, cand[0], cand[1]
        return ox0, oy0, ox1, oy1

    # ------------------------------------------------------------------ ball
    def _kick(self, player, d, power):
        c = self.c
        u = UNIT[d]
        clearance = c.pr + c.br + 0.05
        ox, oy = (self.x0, self.y0) if player == 0 else (self.x1, self.y1)
        self.bx, self.by = ox + u[0] * clearance, oy + u[1] * clearance
        self.vx, self.vy = u[0] * c.bs, u[1] * c.bs
        self.rem = c.kd[power - 1]
        self.poss = -1
        self.ps = 0
        self.ls = 0
        self.lb = -1.0

    def _move_ball(self):
        c = self.c
        if self.rem <= 0 or (self.vx == 0.0 and self.vy == 0.0):
            return -1
        br = c.br
        travel = min(c.bs, self.rem)
        limit = max(0.25, br * 0.45)
        n = max(1, _ceil(travel / limit))
        sd = travel / n
        W, H = c.W, c.H
        gl, gr = c.gl, c.gr
        reach = br + c.pr
        obs = self.obs
        bx, by, vx, vy = self.bx, self.by, self.vx, self.vy
        rem = self.rem
        for _ in range(n):
            vl = _hypot(vx, vy)
            if vl == 0:
                ux = uy = 0.0
            else:
                ux, uy = vx / vl, vy / vl
            px, py = bx, by
            cx, cy = px + ux * sd, py + uy * sd
            if gl <= cx <= gr:
                if cy + br >= H:
                    self.bx, self.by = cx, cy
                    self.rem = 0.0
                    self.vx = self.vy = 0.0
                    return 0
                if cy - br <= 0:
                    self.bx, self.by = cx, cy
                    self.rem = 0.0
                    self.vx = self.vy = 0.0
                    return 1
            # walls
            if cx - br < 0 or cx + br > W:
                vx = -vx
                cx = min(max(cx, br), W - br)
            if cy - br < 0 or cy + br > H:
                vy = -vy
                cy = min(max(cy, br), H - br)
            # obstacles (first hit only)
            for ax, ay, bx2, by2 in obs:
                if cx < ax - br or cx > bx2 + br or cy < ay - br or cy > by2 + br:
                    continue
                qx = ax if cx < ax else (bx2 if cx > bx2 else cx)
                qy = ay if cy < ay else (by2 if cy > by2 else cy)
                dx = cx - qx
                dy = cy - qy
                if dx * dx + dy * dy < br * br:
                    crossed_x = px <= ax - br or px >= bx2 + br
                    crossed_y = py <= ay - br or py >= by2 + br
                    if crossed_x:
                        vx = -vx
                    if crossed_y:
                        vy = -vy
                    if not crossed_x and not crossed_y:
                        vx, vy = -vx, -vy
                    vl = _hypot(vx, vy)
                    if vl == 0:
                        ux = uy = 0.0
                    else:
                        ux, uy = vx / vl, vy / vl
                    nudge = min(0.05, br / 10)
                    cx, cy = px + ux * nudge, py + uy * nudge
                    break
            bx, by = cx, cy
            rem = max(0.0, rem - sd)
            if _hypot(bx - self.x0, by - self.y0) <= reach:
                self._intercept(0)
                return -1
            if _hypot(bx - self.x1, by - self.y1) <= reach:
                self._intercept(1)
                return -1
        self.bx, self.by, self.vx, self.vy = bx, by, vx, vy
        self.rem = rem
        if rem <= 1e-9:
            self.rem = 0.0
            self.vx = self.vy = 0.0
        return -1

    def _intercept(self, player):
        self.poss = player
        self.ps = 0
        self.ls = 0
        self.lb = -1.0
        if player == 0:
            self.bx, self.by = self.x0, self.y0
        else:
            self.bx, self.by = self.x1, self.y1
        self.vx = self.vy = 0.0
        self.rem = 0.0

    def _claim(self):
        if self.poss >= 0 or self.rem > 0:
            return
        prad = self.c.prad
        d0 = _hypot(self.x0 - self.bx, self.y0 - self.by)
        d1 = _hypot(self.x1 - self.bx, self.y1 - self.by)
        winner = -1
        if d0 <= prad and (d1 > prad or d0 <= d1):
            winner = 0
        elif d1 <= prad:
            winner = 1
        if winner >= 0:
            self.poss = winner
            self.ps = 0
            self.ls = 0
            self.lb = -1.0
            if winner == 0:
                self.bx, self.by = self.x0, self.y0
            else:
                self.bx, self.by = self.x1, self.y1

    def _stalled(self):
        if self.poss >= 0 or self.rem > 0:
            self.ls = 0
            self.lb = -1.0
            return
        closest = min(_hypot(self.x0 - self.bx, self.y0 - self.by), _hypot(self.x1 - self.bx, self.y1 - self.by))
        if self.lb < 0 or closest < self.lb - 0.25:
            self.lb = closest
            self.ls = 0
            return
        self.ls += 1
        if self.ls < self.c.llim:
            return
        self.bx, self.by = self.c.W / 2, self.c.H / 2
        self.vx = self.vy = 0.0
        self.rem = 0.0
        self.poss = -1
        self.ps = 0
        self.ls = 0
        self.lb = -1.0

    # ------------------------------------------------------------------ main step
    def step(self, a0, a1):
        """Advance one iteration. Returns the scorer (0/1) or -1."""
        c = self.c
        self._move_players(a0[0], a1[0])
        if self.poss >= 0:
            owner = self.poss
            if self.ps >= 3:
                ao = a0 if owner == 0 else a1
                ac = a1 if owner == 0 else a0
                if not ao[1] and ac[0] != 0:
                    if _hypot(self.x0 - self.x1, self.y0 - self.y1) <= 2 * c.pr + 0.15:
                        ch = 1 - owner
                        self.poss = ch
                        self.ps = 0
        if self.poss >= 0:
            self.ps += 1
            p = self.poss
            if p == 0:
                self.bx, self.by = self.x0, self.y0
                act = a0
            else:
                self.bx, self.by = self.x1, self.y1
                act = a1
            if act[1]:
                self._kick(p, act[1], act[2])
            elif self.ps >= c.plim:
                self._kick(p, 1 if p == 0 else 5, 1)
        scorer = -1
        if self.poss < 0:
            scorer = self._move_ball()
        if scorer >= 0:
            self.ls = 0
            self.lb = -1.0
            if scorer == 0:
                self.s0 += 1
            else:
                self.s1 += 1
        else:
            self._claim()
            self._stalled()
        self.it += 1
        if self.s0 + self.s1 >= c.maxg or self.it >= c.maxit:
            self.done = True
        if scorer >= 0 and not self.done:
            self._restart(1 - scorer)
        return scorer
