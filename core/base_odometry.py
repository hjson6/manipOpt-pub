"""Odometry of the differential-drive base: base_link's pose in the odom frame
from the wheel encoders (distance) and the IMU's yaw rate (heading), with the
gyro's bias learnt whenever the wheels stand still. Taught: the nominal wheel
radius and track. Framework-free.

See docs/implementation_notes.md#base_odometrypy.
"""
import numpy as np

BIAS_SAMPLES_MAX = 500  # running mean over the last ~10 s of standstill
STILL_AFTER_S = 0.3  # the wheels unmoved this long: standing still


class BaseOdometry:
    def __init__(self, wheel_radius, track, use_gyro=True):
        self.r = wheel_radius
        self.track = track
        self.use_gyro = use_gyro
        self.pose = np.zeros(3)  # x, y, yaw in odom
        self.twist = np.zeros(2)  # v, omega
        self.prev = None
        self.bias = 0.0
        self.bias_n = 0
        self.still_s = 0.0
        self.distance = 0.0

    def update(self, wheels, dt, gyro_z=None):
        """One sample: wheel angles [left, right] (rad), time since the last sample, and
        the mean yaw rate over it. Returns the pose."""
        wheels = np.asarray(wheels, dtype=float)
        if self.prev is None or dt <= 0.0:
            self.prev = wheels
            return self.pose
        dl, dr = (wheels - self.prev) * self.r
        self.prev = wheels
        ds = 0.5 * (dl + dr)
        dth = (dr - dl) / self.track
        self.still_s = self.still_s + dt if dl == 0.0 and dr == 0.0 else 0.0
        if self.use_gyro and gyro_z is not None:
            if self.still_s >= STILL_AFTER_S:
                self.bias_n = min(self.bias_n + 1, BIAS_SAMPLES_MAX)
                self.bias += (gyro_z - self.bias) / self.bias_n
            dth = (gyro_z - self.bias) * dt
        yaw_mid = self.pose[2] + 0.5 * dth
        self.pose = self.pose + (ds * np.cos(yaw_mid), ds * np.sin(yaw_mid), dth)
        self.pose[2] = (self.pose[2] + np.pi) % (2 * np.pi) - np.pi
        self.twist = np.array([ds / dt, dth / dt])
        self.distance += abs(ds)
        return self.pose


def wheel_speeds(v, omega, wheel_radius, track, wheel_speed_max):
    """Wheel speeds [left, right] (rad/s) for a body twist, scaled down together (same
    curvature) if one would exceed wheel_speed_max."""
    w = np.array([v - 0.5 * omega * track, v + 0.5 * omega * track]) / wheel_radius
    peak = np.abs(w).max()
    return w * (wheel_speed_max / peak) if peak > wheel_speed_max else w
