"""Validation of the base's drives (scenario side, on the truth; scripts/dev/nav_sim.py
and nav_monitor_node): the true distance from the chassis to each person's legs, people
inside the protective field of the actual motion, contacts, and the motion, a base
version of motion_check.py: the detour (distance driven over the plan's) and the extra
turning (the true heading's travel beyond the plan's: back and forth), wiggles (a turn
reversed and reversed again within WIGGLE_S: one swerve), oscillations (wiggles in a
row), turn-backs (turning in place one way, then back) and, apart, realignments (the
same while aligning on a dock's axis). The field is the safety layer's specification
(nav/safety.py). With body(), also the robot's upper structure (the arm and a held box)
against the people's whole bodies (body_clearance.py), apart when the robot closes on
them.
"""
import numpy as np

from nav import safety

LEG_Y, LEG_R = 0.10, 0.06  # the person model's legs (room_scene.xml), at the chassis's height
FIELD_GRACE_S = 0.3  # a person stepping into the field: the robot has this long to have stopped
TURN_W = 0.2  # rad/s: a turn, for the motion checks
WIGGLE_S = 2.0
TURN_BACK_RAD = 0.2
MOVING_V, MOVING_W = 0.05, 0.1
CLOSING_MS = 0.05
DETOUR_FLAG = 1.25  # as motion_check.py
BODY_FLAG_M = 0.10
EXTRA_TURN_FLAG_DEG = 90.0


def rect_distance(pose, points):
    """Distance from points to the chassis rectangle at pose (same frame); 0 inside. Also
    the points in the chassis's frame."""
    c, s = np.cos(pose[2]), np.sin(pose[2])
    d = np.atleast_2d(points) - pose[:2]
    lx, ly = c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]
    ex = np.maximum(np.abs(lx) - safety.HALF_LENGTH, 0.0)
    ey = np.maximum(np.abs(ly) - safety.HALF_WIDTH, 0.0)
    return np.hypot(ex, ey), np.column_stack([lx, ly])


def legs(people):
    """Both legs' centres of each person (x, y, yaw): (2n, 2)."""
    return np.array([[x - np.sin(yaw) * LEG_Y * sgn, y + np.cos(yaw) * LEG_Y * sgn]
                     for x, y, yaw in people for sgn in (1.0, -1.0)]).reshape(-1, 2)


class DriveScore:
    def __init__(self, goal, t0):
        self.r = dict(goal=goal, t0=t0, min_clear=np.inf, min_clear_any=np.inf, intrusions=0, contacts=0, stops=0,
                      warns=0, reversals=0, wiggles=0, oscillations=0, turn_backs=0, realigns=0, wiggle_deg=0.0,
                      dist=0.0, turned=0.0, turned_by={}, events={}, planned=None, closest=None,
                      body_closing=(np.inf, ""), body_any=(np.inf, ""))
        self.in_field = {}
        self.prev_people = None
        self.prev_pose = None
        self.prev_t = None
        self.w_sign, self.last_rev_t, self.in_place, self.swing, self.after_wiggle = 0, -np.inf, 0.0, 0.0, False

    def truth(self, t, pose, v_body, w_body, people, docking=False, safety_state="", nav_state=""):
        """One tick of the truth: the base's pose (x, y, yaw) and body speeds, the people
        [(x, y, yaw)], all in one frame."""
        r = self.r
        dt = 0.0 if self.prev_t is None else t - self.prev_t
        if self.prev_pose is not None:
            r["dist"] += float(np.hypot(*(pose[:2] - self.prev_pose[:2])))
            turn = abs(float((pose[2] - self.prev_pose[2] + np.pi) % (2 * np.pi) - np.pi))
            r["turned"] += turn
            key = nav_state
            if nav_state == "route":
                key = "route/" + self._near_label(pose, people, dt)
            r["turned_by"][key] = r["turned_by"].get(key, 0.0) + turn
        self.prev_pose, self.prev_t = np.asarray(pose, dtype=float), t
        if not len(people):
            return
        leg_dist, leg_local = rect_distance(pose, legs(people))
        nearer = np.argmin(leg_dist.reshape(-1, 2), axis=1) + 2 * np.arange(len(people))
        dist, local = leg_dist[nearer] - LEG_R, leg_local[nearer]
        # How fast the robot closes on each person: its speed toward them, plus half its corners' speed turning.
        toward = v_body * local[:, 0] / np.maximum(np.hypot(*local.T), 1e-9) + 0.5 * abs(w_body) * safety.SWEEP_R
        moving = abs(v_body) > MOVING_V or abs(w_body) > MOVING_W
        closing = toward > CLOSING_MS
        here = np.array([p[:2] for p in people])
        if moving:
            r["min_clear_any"] = min(r["min_clear_any"], float(dist.min()))
        if moving and closing.any():
            j = int(np.argmin(np.where(closing, dist, np.inf)))
            if dist[j] < r["min_clear"]:
                pv = np.hypot(*((here[j] - self.prev_people[j]) / dt)) if self.prev_people is not None and dt > 0 else 0.0
                r["closest"] = (round(t - r["t0"], 1), round(v_body, 2), round(float(w_body), 2), round(float(pv), 2),
                                tuple(np.round(local[j], 2)), safety_state, nav_state)
            r["min_clear"] = min(r["min_clear"], float(dist[j]))
            hit = safety.protective(leg_local, v_body, w_body, docking, grow=LEG_R).reshape(-1, 2).any(axis=1)
            for j in range(len(people)):
                if hit[j]:
                    self.in_field.setdefault(j, t)
                    if t - self.in_field[j] > FIELD_GRACE_S:
                        r["intrusions"] += 1
                else:
                    self.in_field.pop(j, None)
        else:
            self.in_field.clear()
        r["contacts"] += int(((dist < 0) & (toward > CLOSING_MS)).any() and moving)
        self.prev_people = here

    def _near_label(self, pose, people, dt):
        """The nearest person within 1.5 m of the chassis's centre: standing or walking; else far."""
        if not len(people):
            return "far"
        here = np.array([p[:2] for p in people])
        d = np.hypot(*(here - np.asarray(pose[:2])).T)
        j = int(np.argmin(d))
        if d[j] > 1.5:
            return "far"
        if self.prev_people is None or dt <= 0 or len(self.prev_people) != len(here):
            return "near"
        return "standing" if np.hypot(*(here[j] - self.prev_people[j])) / dt < 0.2 else "walking"

    def command(self, t, v, w, dt, safety_state="", nav_state=""):
        """One command to the drives (after the safety layer), dt apart."""
        r = self.r
        r["stops"] += safety_state == "stop"
        r["warns"] += safety_state == "warn"
        sign = int(np.sign(w)) if abs(w) > TURN_W else 0
        self.swing += w * dt
        if sign and self.w_sign and sign != self.w_sign:
            r["reversals"] += 1
            wiggle = t - self.last_rev_t < WIGGLE_S
            if wiggle:
                r["wiggles"] += 1
                r["oscillations"] += self.after_wiggle
                r["wiggle_deg"] = max(r["wiggle_deg"], float(np.degrees(abs(self.swing))))
            back = self.in_place > TURN_BACK_RAD and abs(v) < MOVING_V
            r["realigns" if nav_state == "align" else "turn_backs"] += back
            self.last_rev_t, self.in_place, self.swing, self.after_wiggle = t, 0.0, 0.0, wiggle
        if abs(v) >= MOVING_V:
            self.in_place = 0.0
        elif sign:
            self.in_place += abs(w) * dt
        self.w_sign = sign or self.w_sign

    def body(self, dist, toward, moving, what):
        """One measurement of the upper structure's clearance to the nearest person: the
        distance, the closing speed of the robot's nearest point, whether the base moves,
        and which parts (text)."""
        if not moving or not np.isfinite(dist):
            return
        if dist < self.r["body_any"][0]:
            self.r["body_any"] = (dist, what)
        if toward > CLOSING_MS and dist < self.r["body_closing"][0]:
            self.r["body_closing"] = (dist, what)

    def events(self, counts):
        """The navigator's counts for this drive (aligns, replans, step-backs)."""
        self.r["events"] = dict(counts)

    def plan(self, length, turn):
        """The navigation's plan for this drive: its length (m) and heading travel (rad)."""
        self.r["planned"] = (float(length), float(turn))

    def detour(self):
        """(distance driven over planned, heading travel beyond the plan's in degrees), or NaN."""
        if self.r["planned"] is None:
            return np.nan, np.nan
        length, turn = self.r["planned"]
        return self.r["dist"] / max(length, 1e-6), float(np.degrees(self.r["turned"] - turn))

    def line(self):
        r = self.r
        detour, extra = self.detour()
        return (f"detour {detour:.2f}, extra turning {extra:+.0f} deg; closest person approached "
                f"{1e3 * r['min_clear']:5.0f} mm, field intrusions {r['intrusions']}, contacts {r['contacts']}, "
                f"stops {r['stops']}, warns {r['warns']}; turn reversals {r['reversals']}, wiggles "
                f"{r['wiggles']} (swing <= {r['wiggle_deg']:.0f} deg), oscillations {r['oscillations']}, turn-backs "
                f"{r['turn_backs']}, realignments {r['realigns']}"
                + (f"; upper body closing {1e3 * r['body_closing'][0]:.0f} mm ({r['body_closing'][1]}), any "
                   f"{1e3 * r['body_any'][0]:.0f} mm" if np.isfinite(r['body_any'][0]) else ""))


def summary(scores):
    """The run's totals over DriveScore records (.r)."""
    rs = [s.r for s in scores]
    det = np.array([s.detour() for s in scores]).reshape(-1, 2)
    flagged = [i + 1 for i, (dr, ex) in enumerate(det) if dr > DETOUR_FLAG or ex > EXTRA_TURN_FLAG_DEG]
    return (f"detour max {np.nanmax(det[:, 0]):.2f}, extra turning max {np.nanmax(det[:, 1]):+.0f} deg (drives over "
            f"{DETOUR_FLAG} or {EXTRA_TURN_FLAG_DEG:.0f} deg: {flagged}); closest person while moving "
            f"{1e3 * min(r['min_clear'] for r in rs):.0f} mm (any person, also one walking "
            f"up: {1e3 * min(r['min_clear_any'] for r in rs):.0f} mm); field intrusions {sum(r['intrusions'] for r in rs)}; "
            f"contacts {sum(r['contacts'] for r in rs)}; wiggles {sum(r['wiggles'] for r in rs)} (heading swing <= "
            f"{max(r['wiggle_deg'] for r in rs):.0f} deg), oscillations {sum(r['oscillations'] for r in rs)}, turn-backs "
            f"{sum(r['turn_backs'] for r in rs)}, realignments at a dock {sum(r['realigns'] for r in rs)} (turn reversals "
            f"{sum(r['reversals'] for r in rs)})"
            + _body_summary(rs))


def turning_summary(scores):
    """Where the heading travel went: extra turning per drive (median, max), the turning
    by navigator state (deg, all drives), and the navigator's aligns, replans and
    step-backs."""
    rs = [s.r for s in scores]
    extra = np.array([s.detour()[1] for s in scores], dtype=float)
    by = {}
    for r in rs:
        for k, v in r["turned_by"].items():
            by[k] = by.get(k, 0.0) + v
    ev = {k: sum(r["events"].get(k, 0) for r in rs) for k in ("aligns", "rejoins", "replans", "step_backs")}
    return (f"extra turning per drive median {np.nanmedian(extra):+.0f}, max {np.nanmax(extra):+.0f} deg; turning by "
            f"state " + ", ".join(f"{k or '-'} {np.degrees(v):.0f}" for k, v in sorted(by.items(), key=lambda kv: -kv[1]))
            + f" deg; turn-straight-turn aligns {ev['aligns']}, curves onto the line {ev['rejoins']}, replans "
              f"{ev['replans']}, step-backs {ev['step_backs']} in {len(rs)} drives")


def _body_summary(rs):
    closing = min((r["body_closing"] for r in rs), key=lambda v: v[0])
    any_ = min((r["body_any"] for r in rs), key=lambda v: v[0])
    if not np.isfinite(any_[0]):
        return ""
    under = sum(r["body_closing"][0] < BODY_FLAG_M for r in rs)
    return (f"; upper body (arm or box to a person): closest while closing on them "
            f"{1e3 * closing[0]:.0f} mm ({closing[1]}), {under} drives under {1e3 * BODY_FLAG_M:.0f} mm; "
            f"any {1e3 * any_[0]:.0f} mm ({any_[1]})")
