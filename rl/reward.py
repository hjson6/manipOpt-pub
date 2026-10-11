"""The reward of a drive for layer 1 of the learned policies (handover_notes/rl,
rl_crowd_nav_plan.md section 5), per plant tick of a NavWorld: on the truth, so for
training and tuning only (a deployed policy never sees it). The tuned baseline's
objective is the same, so an RL gain cannot be put down to a different goal.
"""
import numpy as np

from pick_place_common.nav_world import DT

W_TIME = 1.0  # per s
W_STOP = 2.0  # per protective stop's onset
W_WARN = 0.5  # per s under the safety layer's warning cap
W_TURN = 0.2  # per rad turned off the reference
W_SWERVE = 1.0  # per wiggle (nav_scoring's)
W_CLOSE, CLOSE_M = 1.0, 0.6  # per s x (CLOSE_M - d)^2 / CLOSE_M^2, people the moving chassis closes on
W_HIT = 50.0  # a protective-field intrusion or a contact: the drive ends
W_TIMEOUT = 20.0  # timed out or failed


class DriveReward:
    """start(world) with the drive, tick(world) after each world.tick(), finish(world) at
    its end; terms: the parts, summed over the drive."""

    def start(self, world):
        self.terms = dict(time=0.0, stops=0.0, warn=0.0, turn=0.0, swerves=0.0, close=0.0, hit=0.0, timeout=0.0)
        r = world.score.r
        self.prev = dict(state="clear", wiggles=r["wiggles"], hits=r["intrusions"] + r["contacts"])

    def tick(self, world):
        """This tick's reward; .hit says the drive should end (training)."""
        r, t, nav = world.score.r, self.terms, world.nav
        before = sum(t.values())
        t["time"] -= W_TIME * DT
        if world.last_state == "stop" and self.prev["state"] != "stop":
            t["stops"] -= W_STOP
        if world.last_state == "warn":
            t["warn"] -= W_WARN * DT
        if nav.last is not None and nav.state not in ("docked", "parked", "failed", "idle"):
            t["turn"] -= W_TURN * abs(world.w_body - nav.last[2][0, 4]) * DT
        t["swerves"] -= W_SWERVE * (r["wiggles"] - self.prev["wiggles"])
        near = np.maximum(0.0, CLOSE_M - world.score.closing)
        t["close"] -= W_CLOSE * float(np.sum(near ** 2)) / CLOSE_M ** 2 * DT
        hits = r["intrusions"] + r["contacts"]
        self.hit = hits > self.prev["hits"]
        if self.hit:
            t["hit"] -= W_HIT
        self.prev.update(state=world.last_state, wiggles=r["wiggles"], hits=hits)
        return sum(t.values()) - before

    def finish(self, score):
        if score.r["state"] not in ("docked", "parked") and not self.terms["hit"]:
            self.terms["timeout"] -= W_TIMEOUT
        return sum(self.terms.values())
