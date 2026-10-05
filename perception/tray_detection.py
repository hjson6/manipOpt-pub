"""Find a tray (floor and four walls) in a heightmap of the place zone, and
refine its inner wall faces from the depth pixels. Plain numpy, no ROS or MuJoCo.

See docs/implementation_notes.md#tray_detectionpy.
"""
from dataclasses import dataclass

import numpy as np

MIN_WALL_HEIGHT_M = 0.05  # a rim this far above the floor
RIM_TOL_M = 0.02
WALL_FILL_FRAC = 0.5  # a wall's column (or row) holds at least this share of the fullest one's rim cells
REFINE_BAND_M = 0.03  # depth pixels this far round a coarse wall face refine it
CORNER_SKIP_M = 0.03
WALL_BIN_M = 0.005  # bins along a wall for its inner face


@dataclass
class Tray:
    x_min: float  # inner wall faces
    x_max: float
    y_min: float
    y_max: float
    floor_z: float  # floor top
    wall_top: float

    @property
    def bounds(self):
        return (self.x_min, self.x_max), (self.y_min, self.y_max)

    @property
    def center(self):
        return np.array([(self.x_min + self.x_max) / 2.0, (self.y_min + self.y_max) / 2.0,
                         (self.floor_z + self.wall_top) / 2.0])

    def hull_radius(self, pad):
        """Bounding sphere radius round center, plus pad."""
        return float(np.linalg.norm([(self.x_max - self.x_min) / 2.0, (self.y_max - self.y_min) / 2.0,
                                     (self.wall_top - self.floor_z) / 2.0])) + pad


def _world_points(frames):
    """World x, y, z, validity and pixel footprint (m) of every pixel (pixel centres)
    of every frame, flattened. A frame is (depth, cam_pos, cam_mat, fovy_deg,
    img_width, img_height)."""
    out = []
    for depth, cam_pos, cam_mat, fovy_deg, img_width, img_height in frames:
        f = (img_height / 2.0) / np.tan(np.deg2rad(fovy_deg) / 2.0)
        vs, us = np.mgrid[0:img_height, 0:img_width] + 0.5
        lx = (us - img_width / 2.0) * depth / f
        ly = -(vs - img_height / 2.0) * depth / f
        world = np.asarray(cam_pos) + np.stack([lx, ly, -depth], -1) @ np.asarray(cam_mat).T
        out.append(np.column_stack([world.reshape(-1, 3), (depth > 0).ravel(), (depth / f).ravel()]))
    a = np.vstack(out)
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3] > 0, a[:, 4]


def _inner_edges(coords, lo, hi, res):
    """Inner faces (low wall's high edge, high wall's low edge) from rim point
    coordinates along one axis, or None."""
    counts, edges = np.histogram(coords, bins=max(int(round((hi - lo) / res)), 1), range=(lo, hi))
    counts = np.convolve(counts, np.ones(3), mode="same")  # a thin wall's points split over two bins
    if counts.max() == 0:
        return None
    full = np.nonzero(counts >= WALL_FILL_FRAC * counts.max())[0]
    if len(full) < 2:
        return None
    mid = (full[0] + full[-1]) / 2.0
    low, high = full[full < mid], full[full > mid]
    if not len(low) or not len(high):
        return None
    lw = coords[(coords >= edges[low.min()]) & (coords < edges[low.max() + 1])]
    hw = coords[(coords >= edges[high.min()]) & (coords < edges[high.max() + 1])]
    return float(lw.max()), float(hw.min())


def detect_tray(frames, x_bounds, y_bounds, res, floor_z):
    """Coarse tray from the depth pixels of one or more frames in the place zone, or
    None: the rim is the highest surface; its densest x and y bands are the walls."""
    wx, wy, wz, ok, _ = _world_points(frames)
    ok &= (wx > x_bounds[0]) & (wx < x_bounds[1]) & (wy > y_bounds[0]) & (wy < y_bounds[1])
    high = ok & (wz > floor_z + MIN_WALL_HEIGHT_M)
    if high.sum() < 50:
        return None
    wall_top = float(np.percentile(wz[high], 90))
    rim = high & (np.abs(wz - wall_top) < RIM_TOL_M)
    xs = _inner_edges(wx[rim], *x_bounds, res)
    ys = _inner_edges(wy[rim], *y_bounds, res)
    if xs is None or ys is None:
        return None
    (x0, x1), (y0, y1) = xs, ys
    m = CORNER_SKIP_M
    inside = ok & (wx > x0 + m) & (wx < x1 - m) & (wy > y0 + m) & (wy < y1 - m)
    if not inside.any():
        return None
    return Tray(x0, x1, y0, y1, float(np.median(wz[inside])), float(np.median(wz[rim])))


def _wall_faces(tray, frames):
    """Inner faces (x_min, x_max, y_min, y_max; None where too few rim pixels), the wall
    top and half a pixel's footprint, from the rim pixels near the tray's walls."""
    wx, wy, wz, ok, px = _world_points(frames)
    rim = ok & (np.abs(wz - tray.wall_top) < RIM_TOL_M / 2.0)
    if not rim.any():
        return [None] * 4, tray.wall_top, 0.0
    top = float(np.median(wz[rim]))
    b, s = REFINE_BAND_M, CORNER_SKIP_M
    along_y = (wy > tray.y_min + s) & (wy < tray.y_max - s)
    along_x = (wx > tray.x_min + s) & (wx < tray.x_max - s)
    faces = []
    for sel, coord, along, pick in (
            (rim & along_y & (np.abs(wx - tray.x_min) < b), wx, wy, np.max),
            (rim & along_y & (np.abs(wx - tray.x_max) < b), wx, wy, np.min),
            (rim & along_x & (np.abs(wy - tray.y_min) < b), wy, wx, np.max),
            (rim & along_x & (np.abs(wy - tray.y_max) < b), wy, wx, np.min)):
        c, bins = coord[sel], np.floor(along[sel] / WALL_BIN_M).astype(int)
        per_bin = [pick(c[bins == k]) for k in np.unique(bins)]
        faces.append(float(np.median(per_bin)) if len(per_bin) >= 5 else None)
    return faces, top, 0.5 * float(np.median(px[rim]))


def refine_walls(tray, frames):
    """The tray with its inner faces and wall top from the depth pixels on the rim:
    per 5 mm along each wall the rim pixel nearest the inside, median over the
    wall, plus half a pixel."""
    faces, top, half_px = _wall_faces(tray, frames)
    if all(f is None for f in faces):
        return tray
    x0, x1, y0, y1 = (v + half_px * sgn if v is not None else d for v, d, sgn in zip(
        faces, (tray.x_min, tray.x_max, tray.y_min, tray.y_max), (1, -1, 1, -1)))
    return Tray(x0, x1, y0, y1, tray.floor_z, top)


def reanchor(tray, frames, max_shift=REFINE_BAND_M - 0.005, max_resize=0.01):
    """The known tray found again in a new scan (the robot docked a little differently):
    (the tray moved by what the walls seen show, or None, why). Its size is known, so
    one wall per axis is enough (a box held over the tray hides some); where both are
    seen, the size must not have changed; the move must be under max_shift."""
    faces, top, half_px = _wall_faces(tray, frames)
    found = [None if f is None else f + half_px * sgn for f, sgn in zip(faces, (1, -1, 1, -1))]
    old = (tray.x_min, tray.x_max, tray.y_min, tray.y_max)
    shift = []
    for axis, (lo, hi) in (("x", (0, 1)), ("y", (2, 3))):
        seen = [found[i] - old[i] for i in (lo, hi) if found[i] is not None]
        if not seen:
            return None, f"no {axis} wall seen"
        if len(seen) == 2 and abs(seen[1] - seen[0]) >= max_resize:
            return None, f"{axis} size changed {1e3 * abs(seen[1] - seen[0]):.0f} mm"
        shift.append(float(np.mean(seen)))
    dx, dy = shift
    if np.hypot(dx, dy) >= max_shift:
        return None, f"moved {1e3 * np.hypot(dx, dy):.0f} mm"
    return Tray(tray.x_min + dx, tray.x_max + dx, tray.y_min + dy, tray.y_max + dy, tray.floor_z, top), ""
