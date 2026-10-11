"""Depth-camera noise for the simulated cameras: per-pixel noise growing with
depth, a per-frame offset, and invalid pixels (0) along depth edges and at
random, as structured-light and stereo cameras give. See
docs/implementation_notes.md#depth_noisepy.
"""
import numpy as np

SIGMA_0_M = 0.0005  # per-pixel noise at zero depth
SIGMA_Z2_PER_M = 0.0005  # plus this times depth squared (~0.7 mm at 0.65 m, ~6.6 mm at 3.5 m)
FRAME_OFFSET_PER_M = 0.001  # per-frame offset, std, per metre of depth
EDGE_JUMP_M = 0.02  # a neighbour this much nearer or farther marks a depth edge
EDGE_DROP_P = 0.5
RANDOM_DROP_P = 0.002


def add_depth_noise(depth, rng):
    """A noisy copy of a metric depth image; invalid pixels are 0."""
    d = np.asarray(depth, dtype=np.float64)
    noisy = d + rng.normal(0.0, 1.0, d.shape) * (SIGMA_0_M + SIGMA_Z2_PER_M * d * d)
    noisy += rng.normal(0.0, FRAME_OFFSET_PER_M * float(np.median(d)))
    edge = np.zeros(d.shape, dtype=bool)
    dy = np.abs(np.diff(d, axis=0)) > EDGE_JUMP_M
    dx = np.abs(np.diff(d, axis=1)) > EDGE_JUMP_M
    edge[:-1] |= dy
    edge[1:] |= dy
    edge[:, :-1] |= dx
    edge[:, 1:] |= dx
    drop = (edge & (rng.random(d.shape) < EDGE_DROP_P)) | (rng.random(d.shape) < RANDOM_DROP_P)
    noisy[drop] = 0.0
    return noisy.astype(np.asarray(depth).dtype)
