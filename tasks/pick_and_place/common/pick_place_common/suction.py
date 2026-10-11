"""The vacuum suction cup (grip:=suction): a rubber lip ring at the tool's end, a lumped
pneumatic model of a venturi ejector (the vacuum builds and leaks with the lip's seal,
blow-off on release), a vacuum sensor, a sprung bellows, and the hold: while the lip seals
on a box, a weld carries it from the lip, and breaks when the load exceeds what the vacuum
holds (pulled off, slipped, peeled or twisted off). A model, not air flow.

See docs/implementation_notes.md#suctionpy.
"""
import numpy as np
import mujoco

CUP_DIAMETER_M = 0.040
VACUUM_MAX_KPA = 60.0  # the ejector's vacuum at a sealed cup (gauge)
CUP_AREA_M2 = np.pi * (CUP_DIAMETER_M / 2) ** 2
HOLD_FORCE_MAX_N = VACUUM_MAX_KPA * 1e3 * CUP_AREA_M2  # 75 N
LIP_POINTS = 8
LIP_RADIUS_M = 0.003
LIP_RING_M = CUP_DIAMETER_M / 2 - LIP_RADIUS_M
LIP_END_M = 0.097  # below the wrist body, the bellows extended: where the weld's tool ended, 3 mm above the TCP
LIP_FRICTION = 0.8  # rubber on cardboard
SEAL_GAP_M = 0.0015  # the lip conforms: a bellows cup takes about 5 deg of tilt
LIP_SENSE_M = 0.002  # lip contacts are reported (without force) this far out
SEAL_COS = np.cos(np.radians(10.0))  # a lip contact on a face, not an edge
HOLD_SEAL = 0.5  # the hold starts with this share of the lip sealed
BELLOWS_STROKE_M = 0.005  # the lip stands this far out from the tool's face; a push compresses it
BELLOWS_K = 1500.0  # N/m
BELLOWS_PRELOAD_N = 2.0  # holds it extended
BELLOWS_DAMPING = 30.0  # N s/m
CUP_MASS_KG = 0.02
HOLD_SOLREF = (0.005, 1.0)  # the lip on the face: stiff, the bellows gives
LEAK_K = 8.0  # ejector flow vs a fully open lip's leak: one of 8 lips open holds ~40 %
EVACUATE_TAU_S = 0.15
LEAK_TAU_S = 0.08
BLOWOFF_TAU_S = 0.02
HOLD_MIN_KPA = 1.0  # below this nothing is held
SUPPORTED_N = 1.0  # the box bears this on something else: it cannot drop, no break
VACUUM_SIGMA_KPA = 0.3
CUP_BODY = "suction_cup"
CUP_HOLD = "cup_hold"
CUP_BELLOWS = "cup_bellows"


def add_cup(spec):
    """The cup on the wrist body of a robot spec: the lip ring on a sprung bellows standing
    out of the tool's end face (shortened by the stroke), compressed by a push, at its stop
    while carrying; and the hold (an inactive weld, pointed at a box when it grips)."""
    tool = spec.geom("tool")
    tool.size[1] = (LIP_END_M - BELLOWS_STROKE_M) / 2
    tool.pos[2] = (LIP_END_M - BELLOWS_STROKE_M) / 2
    cup = spec.body("attachment").add_body(name=CUP_BODY, mass=CUP_MASS_KG, ipos=[0, 0, LIP_END_M],
                                           inertia=[1e-6, 1e-6, 1e-6])
    cup.add_joint(name=CUP_BELLOWS, type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1],
                  range=[-BELLOWS_STROKE_M, 0.0], limited=mujoco.mjtLimited.mjLIMITED_TRUE,
                  stiffness=BELLOWS_K, springref=BELLOWS_PRELOAD_N / BELLOWS_K, damping=BELLOWS_DAMPING,
                  solref_limit=list(HOLD_SOLREF))
    for i in range(LIP_POINTS):
        a = 2 * np.pi * i / LIP_POINTS
        cup.add_geom(name=f"cup_lip_{i}", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[LIP_RADIUS_M, 0, 0],
                     pos=[LIP_RING_M * np.cos(a), LIP_RING_M * np.sin(a), LIP_END_M - LIP_RADIUS_M],
                     mass=0.0, condim=3, friction=[LIP_FRICTION, 0.005, 0.0001], gap=LIP_SENSE_M, group=3,
                     rgba=[0.1, 0.1, 0.1, 1])
    spec.add_equality(name=CUP_HOLD, type=mujoco.mjtEq.mjEQ_WELD, name1=CUP_BODY, name2="world",
                      objtype=mujoco.mjtObj.mjOBJ_BODY, active=False, solref=list(HOLD_SOLREF))


def vacuum_reached(sealed):
    """Steady vacuum (share of the maximum) with `sealed` of the lip on a face: the ejector's
    flow (falling linearly to the maximum) against the leak (open lip x sqrt(vacuum))."""
    b = LEAK_K * (1.0 - sealed)
    root = (-b + np.sqrt(b * b + 4.0)) / 2.0
    return root * root


def hold_margin(vacuum_kpa, pull, shear, bend, twist):
    """What breaks first and by how much: (worst load / its limit, name). pull: the cup
    pulling the box toward itself (N); shear (N); bend, twist: moments at the lip (Nm). Peel:
    the cup tips about the far side of its lip ring; slip and twist: friction on the lip's
    net press."""
    held = vacuum_kpa * 1e3 * CUP_AREA_M2
    press = held - pull
    if press <= 0.0:
        return np.inf, "pulled off"
    ratios = {"pulled off": pull / held, "slipped": shear / (LIP_FRICTION * press),
              "peeled": bend / (press * LIP_RING_M), "twisted": twist / (LIP_FRICTION * press * LIP_RING_M)}
    worst = max(ratios, key=ratios.get)
    return ratios[worst], worst


class SuctionCup:
    def __init__(self, model, rng=None, holdable=None):
        """holdable: body ids the cup may hold (the boxes); others it only touches."""
        self.lips = {model.geom(f"cup_lip_{i}").id for i in range(LIP_POINTS)}
        self.body = model.body(CUP_BODY).id
        self.hand = {self.body, int(model.body_parentid[self.body])}  # the lip and the tool's face
        self.eq = model.equality(CUP_HOLD).id
        self.holdable = set(holdable) if holdable is not None else None
        self.rng = rng
        self.valve = False
        self.vacuum_kpa = 0.0
        self.sealed = 0.0
        self.on_body = -1  # the body most of the lip touches
        self.held = -1
        self.held_seal = 0.0  # the lip stays on the face while the bellows gives
        self.load = (0.0, 0.0, 0.0, 0.0)  # pull, shear, bend, twist
        self.support = 0.0  # the held box's normal load on things other than the cup (N)
        self.broke = None  # (body, why, load) at the last break, until read

    def contacts(self, model, data):
        """(lip share sealed on a face, the body most lips touch or -1)."""
        axis = data.xmat[self.body].reshape(3, 3)[:, 2]
        sealed, touched = {}, {}
        for i in range(data.ncon):
            c = data.contact[i]
            lip = c.geom1 if c.geom1 in self.lips else c.geom2 if c.geom2 in self.lips else -1
            if lip < 0:
                continue
            other = model.geom_bodyid[c.geom2 if lip == c.geom1 else c.geom1]
            touched.setdefault(other, set()).add(lip)
            if c.dist < SEAL_GAP_M and abs(c.frame[:3] @ axis) > SEAL_COS:
                sealed.setdefault(other, set()).add(lip)
        if not touched:
            return 0.0, -1
        body = max(touched, key=lambda b: len(touched[b]))
        return len(sealed.get(body, ())) / LIP_POINTS, int(body)

    def cup_load(self, model, data, body):
        """The wrench the cup puts on the held box (its weld, lip and the tool's face), from the
        box's motion less gravity and its other contacts: (pull, shear, bend, twist) at the
        lip's centre."""
        mujoco.mj_rnePostConstraint(model, data)
        acc = np.zeros(6)
        mujoco.mj_objectAcceleration(model, data, mujoco.mjtObj.mjOBJ_BODY, body, acc, 0)  # gravity included
        rot = data.xmat[body].reshape(3, 3)
        com = data.xipos[body]
        inertia = rot @ np.diag(model.body_inertia[body]) @ rot.T
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body, vel, 0)
        force = model.body_mass[body] * acc[3:]
        torque = inertia @ acc[:3] + np.cross(vel[:3], inertia @ vel[:3])
        f6 = np.zeros(6)
        self.support = 0.0
        for i in range(data.ncon):
            c = data.contact[i]
            b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
            if body not in (b1, b2) or {b1, b2} & self.hand:
                continue
            mujoco.mj_contactForce(model, data, i, f6)
            f = c.frame.reshape(3, 3).T @ f6[:3] * (1.0 if b2 == body else -1.0)  # on the box
            force -= f
            torque -= np.cross(c.pos - com, f)
            self.support += max(0.0, f6[0])
        cup = data.xmat[self.body].reshape(3, 3)
        axis = cup[:, 2]  # out of the cup, toward the box
        face = data.xpos[self.body] + LIP_END_M * axis
        moment = torque + np.cross(com - face, force)
        pull = float(-force @ axis)
        twist = float(moment @ axis)
        return (pull, float(np.linalg.norm(force + pull * axis)),
                float(np.linalg.norm(moment - twist * axis)), abs(twist))

    def _grip(self, model, data, body):
        neg_p, neg_q, rel_p, rel_q = np.zeros(3), np.zeros(4), np.zeros(3), np.zeros(4)
        mujoco.mju_negPose(neg_p, neg_q, data.xpos[self.body], data.xquat[self.body])
        mujoco.mju_mulPose(rel_p, rel_q, neg_p, neg_q, data.xpos[body], data.xquat[body])
        model.eq_obj2id[self.eq] = body
        model.eq_data[self.eq, 0:3] = 0.0
        model.eq_data[self.eq, 3:6] = rel_p
        model.eq_data[self.eq, 6:10] = rel_q
        data.eq_active[self.eq] = True
        self.held = body
        self.held_seal = self.sealed

    def _let_go(self, data):
        data.eq_active[self.eq] = False
        self.held = -1

    def step(self, model, data, dt):
        """Advance the vacuum by dt from the seal now; start, keep or break the hold."""
        self.sealed, self.on_body = self.contacts(model, data)
        if self.held >= 0:
            self.sealed, self.on_body = self.held_seal, self.held
            self.load = self.cup_load(model, data, self.held)
        if self.valve:
            target = VACUUM_MAX_KPA * vacuum_reached(self.sealed)
            tau = EVACUATE_TAU_S if target > self.vacuum_kpa else LEAK_TAU_S
        else:
            target, tau = 0.0, BLOWOFF_TAU_S
        self.vacuum_kpa += (target - self.vacuum_kpa) * (1.0 - np.exp(-dt / tau))
        if self.held >= 0:
            ratio, why = hold_margin(self.vacuum_kpa, *self.load)
            if self.vacuum_kpa < HOLD_MIN_KPA or (ratio > 1.0 and self.support <= SUPPORTED_N):
                self.broke = (self.held, why if self.valve else "released", self.load)
                self._let_go(data)
        elif (self.valve and self.vacuum_kpa >= HOLD_MIN_KPA and self.sealed >= HOLD_SEAL
              and (self.holdable is None or self.on_body in self.holdable)):
            self._grip(model, data, self.on_body)

    def reading(self):
        """The vacuum switch's analog output (kPa)."""
        noise = self.rng.normal(0.0, VACUUM_SIGMA_KPA) if self.rng is not None else 0.0
        return max(0.0, self.vacuum_kpa + noise)
