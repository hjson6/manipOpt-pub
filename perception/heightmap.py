"""Depth image -> heightmap, and the searches on it: box segmentation for
picking, flat-spot search for placing. Plain numpy, no ROS or MuJoCo.

See docs/implementation_notes.md#heightmappy.
"""
import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def project_world_point(cam_pos, cam_mat, fovy_deg, img_width, img_height, world_pt):
    """World point -> (u, v) pixel, MuJoCo pinhole convention (camera looks along
    local -z, local +y up; cam_mat columns are the camera axes in world, as
    data.cam_xmat). None if the point is behind the camera.
    """
    rel = np.asarray(world_pt) - np.asarray(cam_pos)
    local = np.asarray(cam_mat).T @ rel
    x, y, z = local
    if z >= 0:
        return None
    f = (img_height / 2.0) / np.tan(np.deg2rad(fovy_deg) / 2.0)
    u = img_width / 2.0 + f * (x / -z)
    v = img_height / 2.0 - f * (y / -z)
    return u, v


def build_column_pixel_map(cam_pos, cam_mat, fovy_deg, img_width, img_height,
                            columns_xy, ref_z):
    """One (row, col) pixel per (x, y) column, projected at the fixed height ref_z."""
    pixel_map = []
    for x, y in columns_xy:
        uv = project_world_point(cam_pos, cam_mat, fovy_deg, img_width, img_height,
                                  (x, y, ref_z))
        if uv is None:
            pixel_map.append(None)
            continue
        u, v = uv
        row, col = int(round(v)), int(round(u))
        if 0 <= row < img_height and 0 <= col < img_width:
            pixel_map.append((row, col))
        else:
            pixel_map.append(None)
    return pixel_map


def infer_column_heights(depth, pixel_map, cam_z, floor_z, empty_tol=0.02):
    """Height per column from a metric depth image (m, along the optical axis;
    the camera looks straight down, so height = cam_z - depth). floor_z where
    nothing is above floor_z + empty_tol or the pixel is off the image.
    """
    heights = np.full(len(pixel_map), floor_z, dtype=float)
    for i, rc in enumerate(pixel_map):
        if rc is None:
            continue
        row, col = rc
        z = cam_z - float(depth[row, col])
        heights[i] = z if z > floor_z + empty_tol else floor_z
    return heights


def fill_invalid(depth):
    """Depth image with invalid pixels (<= 0) set from the nearest valid pixel."""
    bad = ~(np.asarray(depth) > 0)
    if not bad.any():
        return depth
    idx = ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True)
    return np.asarray(depth)[tuple(idx)]


def infer_heights_parallax_corrected(depth, cam_pos, cam_mat, fovy_deg,
                                      img_width, img_height, columns_xy, ref_z,
                                      floor_z, empty_tol=0.02, passes=3):
    """Height per column, each re-projected at its own measured height (passes
    times, on the same frame). Returns (heights, pixel_map). Cells that do not
    read back their own height (unseen, e.g. shadowed floor) are reported as
    floor.
    """
    depth = fill_invalid(depth)
    xy = np.asarray(columns_xy, dtype=float)  # (N, 2)
    cam_pos = np.asarray(cam_pos, dtype=float)
    rot_t = np.asarray(cam_mat, dtype=float).T  # world -> camera
    f = (img_height / 2.0) / np.tan(np.deg2rad(fovy_deg) / 2.0)
    heights = np.full(len(columns_xy), float(ref_z), dtype=float)
    rows = np.full(len(columns_xy), -1, dtype=int)
    cols = np.full(len(columns_xy), -1, dtype=int)

    for _ in range(passes):
        world = np.column_stack([xy, heights])  # (N, 3)
        rel = world - cam_pos
        local = rel @ rot_t.T
        lx, ly, lz = local[:, 0], local[:, 1], local[:, 2]
        behind = lz >= 0
        # Behind-camera points are masked below; avoid dividing by their ~0 z.
        safe_lz = np.where(behind, -1.0, lz)
        u = img_width / 2.0 + f * (lx / -safe_lz)
        v = img_height / 2.0 - f * (ly / -safe_lz)
        r = np.round(v).astype(int)
        c = np.round(u).astype(int)
        valid = (~behind) & (r >= 0) & (r < img_height) & (c >= 0) & (c < img_width)
        rows = np.where(valid, r, -1)
        cols = np.where(valid, c, -1)
        # Clipped only to keep the index in range; invalid reads are discarded.
        depth_vals = depth[np.clip(rows, 0, img_height - 1), np.clip(cols, 0, img_width - 1)]
        z_world = float(cam_pos[2]) - depth_vals
        heights = np.where(valid, z_world, heights)

    valid = rows >= 0
    pixel_map = [(int(r), int(c)) if ok else None
                 for r, c, ok in zip(rows, cols, valid)]
    # Consistency check: a visible surface reads back its own height. An unseen
    # cell alternates between floor and box top, and without this check the last
    # pass decided which it reported.
    world = np.column_stack([xy, heights])
    local = (world - cam_pos) @ rot_t.T
    safe_lz = np.where(local[:, 2] >= 0, -1.0, local[:, 2])
    r = np.round(img_height / 2.0 - f * (local[:, 1] / -safe_lz)).astype(int)
    c = np.round(img_width / 2.0 + f * (local[:, 0] / -safe_lz)).astype(int)
    ok = valid & (r >= 0) & (r < img_height) & (c >= 0) & (c < img_width)
    reread = float(cam_pos[2]) - depth[np.clip(r, 0, img_height - 1), np.clip(c, 0, img_width - 1)]
    consistent = ok & (np.abs(reread - heights) <= empty_tol / 2.0)
    # Collapse to floor_z only now, so later passes are not re-projected from it.
    out = np.where(consistent & (heights > floor_z + empty_tol), heights, floor_z)
    return out, pixel_map


def build_dense_grid_xy(x_bounds, y_bounds, resolution):
    """Row-major (x, y) points spanning the bounds at `resolution`, and the
    (n_rows, n_cols) shape to reshape the heights with.
    """
    xs = np.arange(x_bounds[0], x_bounds[1] + resolution / 2, resolution)
    ys = np.arange(y_bounds[0], y_bounds[1] + resolution / 2, resolution)
    points = [(float(x), float(y)) for y in ys for x in xs]
    return points, (len(ys), len(xs))


def find_best_footprint(heightmap, footprint_cells, mode, flatness_tol,
                         floor_z=None, adjacency_cells=None, inlier_frac=1.0,
                         wall_bonus=1.0, neighbor_bonus=2.0,
                         neighbor_tol=None):
    """Best (row, col, height) of a flat footprint_cells = (fh, fw) window
    (top-left corner), or None.

    A window is a candidate if at least inlier_frac of its cells are within
    flatness_tol of its median; its height is the inliers' max. The window is
    a probe and may be smaller than the box, since edge cells read side walls.
    mode="highest": the highest candidate, centred on its connected plateau.
    mode="lowest": the lowest candidates (within neighbor_tol), scored by
    contact with walls and occupied cells round adjacency_cells (default:
    footprint_cells). floor_z is required for "lowest".
    """
    fh, fw = footprint_cells
    ah, aw = adjacency_cells if adjacency_cells is not None else footprint_cells
    n_rows, n_cols = heightmap.shape
    if neighbor_tol is None:
        neighbor_tol = flatness_tol
    if mode == "lowest" and floor_z is None:
        raise ValueError("floor_z is required for mode='lowest'")

    # All windows at once (a strided view); a per-window loop cost tens of ms.
    windows = np.lib.stride_tricks.sliding_window_view(heightmap, (fh, fw))
    medians = np.median(windows, axis=(2, 3))
    inlier_mask = np.abs(windows - medians[:, :, None, None]) <= flatness_tol
    valid = inlier_mask.mean(axis=(2, 3)) >= inlier_frac
    # Height from the inliers only: the outliers are the unreliable edge cells.
    heights_per_window = np.where(inlier_mask, windows, -np.inf).max(axis=(2, 3))

    valid_rows, valid_cols = np.nonzero(valid)
    if len(valid_rows) == 0:
        return None
    candidates = list(zip(
        valid_rows.tolist(), valid_cols.tolist(),
        heights_per_window[valid_rows, valid_cols].tolist()))

    if mode == "highest":
        best_height = max(c[2] for c in candidates)
        tied = {(r, c) for r, c, h in candidates if best_height - h <= flatness_tol}
        # Centre on the plateau connected to the first tied window: its top-left was
        # up to a probe width off (14 mm on a 60 mm box). Other boxes at the same
        # height are separate plateaus.
        seed = min(tied)
        component, frontier = {seed}, [seed]
        while frontier:
            r, c = frontier.pop()
            for nb in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if nb in tied and nb not in component:
                    component.add(nb)
                    frontier.append(nb)
        row = int(round(sum(r for r, _ in component) / len(component)))
        col = int(round(sum(c for _, c in component) / len(component)))
        return row, col, best_height

    if mode != "lowest":
        raise ValueError(f"unknown mode {mode!r}")

    def adjacency_score(row, col, height):
        score = 0.0
        # Rings round the true footprint (the probe is centred inside it).
        a_row = row - (ah - fh) // 2
        a_col = col - (aw - fw) // 2
        # One-cell rings; each side scores on its own, so a corner beats a wall.
        sides = {
            "west": (slice(max(a_row, 0), a_row + ah), a_col - 1),
            "east": (slice(max(a_row, 0), a_row + ah), a_col + aw),
            "north": (a_row - 1, slice(max(a_col, 0), a_col + aw)),
            "south": (a_row + ah, slice(max(a_col, 0), a_col + aw)),
        }
        for _name, (r_idx, c_idx) in sides.items():
            if isinstance(r_idx, slice):
                out_of_bounds = c_idx < 0 or c_idx >= n_cols
            else:
                out_of_bounds = r_idx < 0 or r_idx >= n_rows
            if out_of_bounds:
                score += wall_bonus
                continue
            ring = np.atleast_1d(heightmap[r_idx, c_idx])
            occupied_frac = float(np.mean(ring > floor_z + neighbor_tol))
            score += neighbor_bonus * occupied_frac
        return score

    min_height = min(c[2] for c in candidates)
    near_lowest = [c for c in candidates if c[2] - min_height <= neighbor_tol]
    row, col, height = max(near_lowest, key=lambda c: adjacency_score(*c))
    return row, col, height


def footprint_center_xy(row, col, footprint_cells, x_bounds, y_bounds, resolution):
    """World (x, y) of the centre of a window whose top-left is (row, col)."""
    fh, fw = footprint_cells
    x = x_bounds[0] + (col + (fw - 1) / 2.0) * resolution
    y = y_bounds[0] + (row + (fh - 1) / 2.0) * resolution
    return float(x), float(y)


def _label_by_height(heightmap, mask, step_tol):
    """Connected components of `mask` over 4-neighbours whose heights differ by at
    most step_tol. Returns (labels, n) like ndimage.label.
    """
    h, w = heightmap.shape
    idx = np.arange(h * w).reshape(h, w)
    rows, cols = [], []
    for a, b, ha, hb, ma, mb in (
            (idx[:, :-1], idx[:, 1:], heightmap[:, :-1], heightmap[:, 1:], mask[:, :-1], mask[:, 1:]),
            (idx[:-1, :], idx[1:, :], heightmap[:-1, :], heightmap[1:, :], mask[:-1, :], mask[1:, :])):
        keep = ma & mb & (np.abs(ha - hb) <= step_tol)
        rows.append(a[keep])
        cols.append(b[keep])
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(h * w, h * w))
    _, comp = connected_components(graph, directed=False)
    comp = comp.reshape(h, w)
    labels = np.zeros((h, w), dtype=int)
    ids = np.unique(comp[mask])
    for n, c in enumerate(ids, start=1):
        labels[(comp == c) & mask] = n
    return labels, len(ids)


def find_topmost_boxes(heightmap, floor_z, flatness_tol, min_footprint_cells,
                        inlier_frac=0.85, min_fill_frac=0.85):
    """Segment a heightmap into visible box tops: (row0, col0, row1, col1, height,
    area_cells) per box, bounding box end-exclusive, height the inliers' median.

    Neighbouring cells join if their heights differ by at most flatness_tol / 2,
    so a short box beside a tall one stays separate. A box that another box
    only partly covers still shows a top; pick_order puts it last. Regions smaller than
    min_footprint_cells, filling less than min_fill_frac of their bounding box,
    or with fewer than inlier_frac of cells near the median are dropped.
    """
    occupied = heightmap > floor_z + flatness_tol
    labeled, n_labels = _label_by_height(heightmap, occupied, flatness_tol / 2.0)
    records = []
    for label_id in range(1, n_labels + 1):
        region_mask = labeled == label_id
        area = int(region_mask.sum())
        if area < min_footprint_cells:
            continue
        rows, cols = np.nonzero(region_mask)
        row0, row1 = int(rows.min()), int(rows.max()) + 1
        col0, col1 = int(cols.min()), int(cols.max()) + 1
        bbox_area = (row1 - row0) * (col1 - col0)
        if area / bbox_area < min_fill_frac:
            continue
        region_heights = heightmap[region_mask]
        median = np.median(region_heights)
        inliers = np.abs(region_heights - median) <= flatness_tol
        if inliers.mean() < inlier_frac:
            continue
        height = float(np.median(region_heights[inliers]))  # the max is biased up by noise
        records.append((row0, col0, row1, col1, height, area))
    return records


TOP_EXTENT_BIAS_M = 0.0  # per side, fitted with scripts/dev/footprint_accuracy.py


def top_extent(depth, cam_pos, cam_mat, fovy_deg, img_width, img_height, x_bounds, y_bounds,
               top_z, tol, seed_xy=None):
    """(x_min, x_max, y_min, y_max) of a box top at height top_z inside the given
    world bounds, from the depth pixels on it (finer than the heightmap grid).
    Only pixels on a flat patch count (all four neighbours at that height too), so
    a wall or taller box's side face crossing that height is not taken for the top;
    that drops the edge pixels, so one and a half pixels are added back each side.
    With seed_xy, only the connected patch nearest that point counts (a neighbour of
    the same height in the bounds is left out). None if no pixel is on it.
    """
    f = (img_height / 2.0) / np.tan(np.deg2rad(fovy_deg) / 2.0)
    vs, us = np.mgrid[0:img_height, 0:img_width] + 0.5  # pixel centres
    lx = (us - img_width / 2.0) * depth / f
    ly = -(vs - img_height / 2.0) * depth / f
    world = np.asarray(cam_pos) + np.stack([lx, ly, -depth], -1) @ np.asarray(cam_mat).T
    flat = ndimage.binary_erosion(np.abs(world[..., 2] - top_z) < tol)
    sel = (flat & (world[..., 0] > x_bounds[0]) & (world[..., 0] < x_bounds[1])
           & (world[..., 1] > y_bounds[0]) & (world[..., 1] < y_bounds[1]))
    if not sel.any():
        return None
    if seed_xy is not None:
        labels, n = ndimage.label(sel)
        if n > 1:
            d2 = (world[..., 0] - seed_xy[0]) ** 2 + (world[..., 1] - seed_xy[1]) ** 2
            sel = labels == labels[sel][np.argmin(d2[sel])]
    pad = 1.5 * float(np.median(depth[sel])) / f + TOP_EXTENT_BIAS_M
    wx, wy = world[..., 0][sel], world[..., 1][sel]
    return (float(wx.min() - pad), float(wx.max() + pad), float(wy.min() - pad), float(wy.max() + pad))


def touches_higher(heightmap, records, flatness_tol):
    """Per record from find_topmost_boxes, whether its top touches anything
    higher in the heightmap: a higher box may rest on it, and part of its top may
    be hidden, so its measured size is not the box's. Any higher cells count, not
    only detected tops (noise can make a top fail detection).
    """
    def top_mask(rec):
        row0, col0, row1, col1, height, _ = rec
        mask = np.zeros(heightmap.shape, dtype=bool)
        mask[row0:row1, col0:col1] = np.abs(heightmap[row0:row1, col0:col1] - height) <= flatness_tol
        return mask

    flags = []
    for rec in records:
        mask = top_mask(rec)
        ring = ndimage.binary_dilation(mask) & ~mask
        flags.append(bool((heightmap[ring] > rec[4] + flatness_tol).any()))
    return flags


def pick_order(heightmap, records, flatness_tol):
    """Records from find_topmost_boxes in pick order: largest area first (then
    tallest), but a top that touches a higher detected top goes after every
    top that does not (known_issues H5).
    """
    flags = touches_higher(heightmap, records, flatness_tol)
    return [r for _, r in sorted(zip(flags, records), key=lambda br: (br[0], -br[1][5], -br[1][4]))]
