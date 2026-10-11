"""Occupancy grid (log-odds, free space traced along each ray from its scanner), with
each cell's mean hit position (a wall is not quantized to the cells), and a signed
distance field. Saved and loaded in the ROS map_server format (PGM + YAML), the hit
means alongside (.npz). Framework-free.
"""
from pathlib import Path

import numpy as np
from scipy import ndimage

from slam.scan import transform

L_OCC, L_FREE, L_MIN, L_MAX = 0.85, -0.4, -2.0, 3.5
OCC_THRESHOLD, FREE_THRESHOLD = 0.6, -0.6  # log-odds


class OccupancyGrid:
    def __init__(self, origin, shape, resolution=0.05):
        self.origin = np.array(origin, dtype=float)  # map-frame position of cell (0, 0)'s corner
        self.res = float(resolution)
        self.log_odds = np.zeros(shape)  # [row = y, col = x]
        self.hit_sum = np.zeros(shape + (2,))
        self.hit_n = np.zeros(shape)
        self._df = None

    @classmethod
    def around(cls, points, resolution=0.05, margin=1.0):
        lo, hi = points.min(axis=0) - margin, points.max(axis=0) + margin
        shape = tuple(np.ceil((hi - lo) / resolution).astype(int)[::-1])
        return cls(lo, shape, resolution)

    def cells(self, xy):
        return np.floor((xy - self.origin) / self.res).astype(int)

    def integrate(self, pose, points, origins):
        """One scan at pose: points and their scanners' positions in base_link."""
        p = transform(pose, points)
        o = transform(pose, origins)
        length = np.hypot(*(p - o).T)
        step = self.res * 0.5
        n = np.maximum(np.ceil(length / step).astype(int) - 1, 0)
        ray = np.repeat(np.arange(len(p)), n)
        k = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
        frac = (k * step / np.maximum(length[ray], 1e-9))[:, None]
        free = self.cells(o[ray] + (p[ray] - o[ray]) * frac)
        hit = self.cells(p)
        h, w = self.log_odds.shape
        free = free[(free[:, 0] >= 0) & (free[:, 0] < w) & (free[:, 1] >= 0) & (free[:, 1] < h)]
        inside = (hit[:, 0] >= 0) & (hit[:, 0] < w) & (hit[:, 1] >= 0) & (hit[:, 1] < h)
        hit, p = hit[inside], p[inside]
        np.add.at(self.hit_sum, (hit[:, 1], hit[:, 0]), p)
        np.add.at(self.hit_n, (hit[:, 1], hit[:, 0]), 1)
        free_idx = np.unique(free[:, 1] * w + free[:, 0])
        hit_idx = np.unique(hit[:, 1] * w + hit[:, 0])
        free_idx = np.setdiff1d(free_idx, hit_idx)
        flat = self.log_odds.reshape(-1)
        flat[free_idx] += L_FREE
        flat[hit_idx] += L_OCC
        np.clip(self.log_odds, L_MIN, L_MAX, out=self.log_odds)
        self._df = None

    @property
    def occupied(self):
        return self.log_odds > OCC_THRESHOLD

    @property
    def free(self):
        return self.log_odds < FREE_THRESHOLD

    def occupied_points(self):
        """Per occupied cell, the mean of the hits in it (the cell's centre if none)."""
        r, c = np.nonzero(self.occupied)
        centre = self.origin + (np.column_stack([c, r]) + 0.5) * self.res
        n = self.hit_n[r, c]
        return np.where(n[:, None] > 0, self.hit_sum[r, c] / np.maximum(n, 1)[:, None], centre)

    def distance_field(self):
        """(signed distance in m, d/dx, d/dy) per cell: from cell centres, positive in
        free space, negative inside occupied cells, so that its zero lies on the faces
        between occupied and free cells, where the surfaces are, not on the occupied
        cells' centres half a cell behind them."""
        if self._df is None:
            occ = self.occupied
            d = (ndimage.distance_transform_edt(~occ) - ndimage.distance_transform_edt(occ)) * self.res
            gy, gx = np.gradient(d, self.res)
            self._df = (d, gx, gy)
        return self._df

    def sample(self, xy):
        """Signed distance and its gradient at map points, bilinear (inf outside)."""
        d, gx, gy = self.distance_field()
        u = (xy - self.origin) / self.res - 0.5
        h, w = d.shape
        x0 = np.clip(np.floor(u[:, 0]).astype(int), 0, w - 2)
        y0 = np.clip(np.floor(u[:, 1]).astype(int), 0, h - 2)
        fx = np.clip(u[:, 0] - x0, 0.0, 1.0)
        fy = np.clip(u[:, 1] - y0, 0.0, 1.0)

        def bil(a):
            return ((1 - fx) * (1 - fy) * a[y0, x0] + fx * (1 - fy) * a[y0, x0 + 1]
                    + (1 - fx) * fy * a[y0 + 1, x0] + fx * fy * a[y0 + 1, x0 + 1])
        inside = (u[:, 0] >= 0) & (u[:, 0] <= w - 1) & (u[:, 1] >= 0) & (u[:, 1] <= h - 1)
        return np.where(inside, bil(d), np.inf), bil(gx), bil(gy)

    def save(self, path):
        """ROS map_server format: <path>.pgm (0 occupied, 254 free, 205 unknown) and .yaml."""
        path = Path(path)
        img = np.full(self.log_odds.shape, 205, np.uint8)
        img[self.free] = 254
        img[self.occupied] = 0
        img = img[::-1]  # PGM rows go from the top (+y) down
        with open(path.with_suffix(".pgm"), "wb") as f:
            f.write(f"P5\n{img.shape[1]} {img.shape[0]}\n255\n".encode())
            f.write(img.tobytes())
        np.savez_compressed(path.with_suffix(".npz"), hit_sum=self.hit_sum, hit_n=self.hit_n)
        path.with_suffix(".yaml").write_text(
            f"image: {path.with_suffix('.pgm').name}\nmode: trinary\nresolution: {self.res}\n"
            f"origin: [{self.origin[0]:.4f}, {self.origin[1]:.4f}, 0.0]\nnegate: 0\n"
            f"occupied_thresh: 0.65\nfree_thresh: 0.25\n")

    @classmethod
    def load(cls, yaml_path):
        """A map_server map (trinary) as a grid; log-odds at the thresholds' extremes."""
        yaml_path = Path(yaml_path)
        meta = dict(line.split(": ", 1) for line in yaml_path.read_text().splitlines() if ": " in line)
        res = float(meta["resolution"])
        origin = [float(v) for v in meta["origin"].strip("[]").split(",")[:2]]
        raw = (yaml_path.parent / meta["image"]).read_bytes()
        header = raw.split(b"\n", 3)
        w, h = (int(v) for v in header[1].split())
        img = np.frombuffer(header[3][:w * h], np.uint8).reshape(h, w)[::-1]
        grid = cls(origin, (h, w), res)
        occ_thresh = float(meta.get("occupied_thresh", 0.65))
        free_thresh = float(meta.get("free_thresh", 0.25))
        p = (255 - img.astype(float)) / 255.0
        grid.log_odds[p > occ_thresh] = L_MAX
        grid.log_odds[p < free_thresh] = L_MIN
        hits = yaml_path.with_suffix(".npz")
        if hits.exists():
            with np.load(hits) as f:
                grid.hit_sum, grid.hit_n = f["hit_sum"], f["hit_n"]
        return grid
