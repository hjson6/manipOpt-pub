"""Plant side of the mobile base: the wheel drives as a motor controller runs them
(a speed loop every physics substep, a standstill hold, a command watchdog), the
braked wheels' static friction, and the wheel encoders and the IMU as their
drivers give them. The base's pose is
never published to the method; the method integrates these
(core/base_odometry.py).

See docs/implementation_notes.md#base_drivepy.
"""
import mujoco
import numpy as np

from pick_place_common import plant_conditions

WHEEL_JOINTS = ("wheel_left", "wheel_right")
WHEEL_MOTORS = ("wheel_left_motor", "wheel_right_motor")
BASE_BODY = "base_link"
IMU_SITE = "imu"
COMMAND_TIMEOUT_S = 0.2  # watchdog: no wheel command for this long, stop
HOLD_AFTER_S = 0.2  # commanded zero this long: brake the wheels where they are
BRAKE_K = 10000.0  # Nm/rad at the wheel, the brake's stiffness through the gearbox
BRAKE_D = 40.0  # Nm per (rad/s)
SPEED_KV = 40.0  # Nm per (rad/s), the speed loop
SPEED_MAX = 20.0  # rad/s at the wheel
ENCODER_COUNTS_PER_REV = 16384  # at the wheel
GYRO_SIGMA = 0.002  # rad/s per sample
GYRO_BIAS_MAX = 0.005  # rad/s, drawn once per run
ACCEL_SIGMA = 0.03  # m/s^2 per axis
STICK_FRICTION = 1.0  # tyre on floor, as the wheel-floor contact pairs
STICK_SPEED = 0.002  # m/s: a slipping wheel sticks again below this


def has_base(model):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, BASE_BODY) >= 0


class BaseDrive:
    def __init__(self, model, data, rng, noise=True, conditions=()):
        """conditions: plant_conditions.CONDITIONS that act here (spill, gyro_drift)."""
        self.model = model
        self.qpos_adr = [model.joint(n).qposadr[0] for n in WHEEL_JOINTS]
        self.dof_adr = [model.joint(n).dofadr[0] for n in WHEEL_JOINTS]
        self.motor_ids = [model.actuator(n).id for n in WHEEL_MOTORS]
        self.torque_max = model.actuator_ctrlrange[self.motor_ids, 1].copy()
        self.body = model.body(BASE_BODY).id
        self.imu_site = model.site(IMU_SITE).id
        self.accel_adr = model.sensor_adr[model.sensor("imu_accel").id]
        self.rng = rng
        self.noise = noise
        self.gyro_bias = rng.uniform(-GYRO_BIAS_MAX, GYRO_BIAS_MAX, 3) if noise else np.zeros(3)
        self.command = np.zeros(2)
        self.command_t = -np.inf
        self.zero_since = 0.0
        self.hold = None
        self.speed_ref = np.zeros(2)
        self.prev_rot = data.site_xmat[self.imu_site].reshape(3, 3).copy()
        self.wheel_bodies = [model.jnt_bodyid[model.joint(n).id] for n in WHEEL_JOINTS]
        self.wheel_geoms = [model.body_geomadr[b] for b in self.wheel_bodies]
        self.stick_eqs = [model.equality(f"{n}_stick").id for n in WHEEL_JOINTS]
        self.stuck = False
        self.slips = 0
        self.spill = "spill" in conditions
        self.gyro_drift = "gyro_drift" in conditions and noise
        self.floor_pairs = [int(np.flatnonzero((model.pair_geom1 == g) | (model.pair_geom2 == g))[0])
                            for g in self.wheel_geoms]
        self.friction = np.full(2, STICK_FRICTION)

    def set_command(self, speeds, t):
        self.command = np.clip(np.asarray(speeds, dtype=float)[:2], -SPEED_MAX, SPEED_MAX)
        self.command_t = t

    def update(self, data):
        """The speed setpoints for the next tick (sim time data.time), or the brake after
        HOLD_AFTER_S at zero; returns the setpoints."""
        t = data.time
        cmd = self.command if t - self.command_t <= COMMAND_TIMEOUT_S else np.zeros(2)
        if np.any(cmd != 0.0):
            self.zero_since, self.hold = t, None
        elif t - self.zero_since >= HOLD_AFTER_S and self.hold is None:
            self.hold = data.qpos[self.qpos_adr].copy()
        self.speed_ref = cmd
        if self.spill:
            for i, (body, pair) in enumerate(zip(self.wheel_bodies, self.floor_pairs)):
                mu = plant_conditions.SPILL_FRICTION if plant_conditions.on_spill(data.xpos[body]) else STICK_FRICTION
                self.model.pair_friction[pair, 0:2] = mu
                self.friction[i] = mu
        self._stiction(data, self.hold is not None)
        return cmd

    def _stiction(self, data, braked):
        """A braked wheel's contact point sticks to the floor (MuJoCo's soft friction only
        creeps): pinned while braked and still, released when its horizontal force would
        exceed friction (it slides, and sticks again once it has stopped) or when the
        drive moves."""
        if self.stuck and (not braked or self._breaks_away(data)):
            data.eq_active[self.stick_eqs] = False
            self.stuck = False
            self.slips += braked
            return
        if braked and not self.stuck and self._wheels_still(data):
            for body, geom, eq in zip(self.wheel_bodies, self.wheel_geoms, self.stick_eqs):
                rot = data.xmat[body].reshape(3, 3)
                point = data.xpos[body] - (0.0, 0.0, self.model.geom_size[geom][0])
                self.model.eq_data[eq, 0:3] = rot.T @ (point - data.xpos[body])
                self.model.eq_data[eq, 3:6] = point
            data.eq_active[self.stick_eqs] = True
            self.stuck = True

    def _wheels_still(self, data):
        vel = np.zeros(6)
        for body in self.wheel_bodies:
            mujoco.mj_objectVelocity(self.model, data, mujoco.mjtObj.mjOBJ_BODY, body, vel, 0)
            if np.hypot(vel[3], vel[4]) > STICK_SPEED:
                return False
        return True

    def _breaks_away(self, data):
        n = data.nefc
        rows = data.efc_type[:n] == mujoco.mjtConstraint.mjCNSTR_EQUALITY
        f6 = np.zeros(6)
        for eq, geom, mu in zip(self.stick_eqs, self.wheel_geoms, self.friction):
            f = data.efc_force[:n][rows & (data.efc_id[:n] == eq)]
            if len(f) != 3:
                continue
            normal = f[2]
            for i in np.flatnonzero((data.contact.geom1 == geom) | (data.contact.geom2 == geom)):
                mujoco.mj_contactForce(self.model, data, i, f6)
                normal += data.contact.frame[i][2] * f6[0]
            if np.hypot(f[0], f[1]) > mu * max(normal, 0.0):
                return True
        return False

    def torque(self, data):
        """Before each physics substep: the speed loop, or the brake holding the wheels'
        angles; wheel torques into data.ctrl."""
        qd = data.qvel[self.dof_adr]
        tau = (BRAKE_K * (self.hold - data.qpos[self.qpos_adr]) - BRAKE_D * qd if self.hold is not None
               else SPEED_KV * (self.speed_ref - qd))
        data.ctrl[self.motor_ids] = np.clip(tau, -self.torque_max, self.torque_max)

    def step(self, model, data, n):
        """n physics substeps with the speed loop before each."""
        for _ in range(n):
            self.torque(data)
            mujoco.mj_step(model, data)

    def encoders(self, data):
        """Wheel angles (rad) at the encoder's resolution."""
        step = 2 * np.pi / ENCODER_COUNTS_PER_REV
        return np.round(data.qpos[self.qpos_adr] / step) * step

    def imu(self, data, dt):
        """(gyro, accel) in the IMU frame: the gyro as the mean rate over the last dt (a
        delta-angle output), with noise and a bias; the accelerometer now, with noise."""
        rot = data.site_xmat[self.imu_site].reshape(3, 3).copy()
        d = self.prev_rot.T @ rot
        self.prev_rot = rot
        angle = np.arccos(np.clip((np.trace(d) - 1) / 2, -1.0, 1.0))
        axis = np.array([d[2, 1] - d[1, 2], d[0, 2] - d[2, 0], d[1, 0] - d[0, 1]])
        gyro = axis * (angle / (2 * np.sin(angle)) if angle > 1e-9 else 0.5) / dt
        accel = data.sensordata[self.accel_adr:self.accel_adr + 3].copy()
        if self.gyro_drift:
            a = np.exp(-dt / plant_conditions.GYRO_DRIFT_TAU_S)
            self.gyro_bias = a * self.gyro_bias + plant_conditions.GYRO_DRIFT_SIGMA * np.sqrt(1 - a * a) * self.rng.normal(size=3)
        if self.noise:
            gyro = gyro + self.gyro_bias + self.rng.normal(0.0, GYRO_SIGMA, 3)
            accel = accel + self.rng.normal(0.0, ACCEL_SIGMA, 3)
        return gyro, accel

    def truth(self, data):
        """base_link's true pose in the room: [x, y, z, qw, qx, qy, qz]. Validation only."""
        return np.concatenate([data.xpos[self.body], data.xquat[self.body]])
