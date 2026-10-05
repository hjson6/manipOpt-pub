"""Obstacle detection from the overhead depth camera (numpy/scipy only).

A pixel is foreground if it unprojects above the floor plane; the pile and
tray are masked out by region, and the robot by the caller's robot_mask.
Output is raw blobs in the base frame; tracking and classification are the
supervisor's job. See docs/design_notes.md, "Workspace obstacle sensing".

Conventions as heightmap.project_world_point; depth is along the optical
axis, not the ray.
"""
import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError


def unproject_depth(depth, cam_pos, cam_mat, fovy_deg):
    """(H, W) metric depth -> (H, W, 3) world points."""
    h, w = depth.shape
    f = (h / 2.0) / np.tan(np.deg2rad(fovy_deg) / 2.0)
    u = np.arange(w)[None, :]
    v = np.arange(h)[:, None]
    # Integer (u, v) are pixel indices, as in project_world_point.
    x = ((u - w / 2.0) / f * depth).astype(np.float32)
    y = (-(v - h / 2.0) / f * depth).astype(np.float32)
    # Elementwise, not a matmul: next to the solver, the BLAS call was ~10x slower.
    r = np.asarray(cam_mat, dtype=np.float32)
    pts = (x[..., None] * r[:, 0] + y[..., None] * r[:, 1]
           - depth.astype(np.float32)[..., None] * r[:, 2])
    return pts + np.asarray(cam_pos, dtype=np.float32)


def project_points(points, cam_pos, cam_mat, fovy_deg, img_width, img_height):
    """(n, 3) world points -> (n, 2) pixels (u, v), as unproject_depth inverts; points
    behind the camera are dropped."""
    local = (np.asarray(points, dtype=float) - np.asarray(cam_pos)) @ np.asarray(cam_mat)
    local = local[local[:, 2] < 0]
    f = (img_height / 2.0) / np.tan(np.deg2rad(fovy_deg) / 2.0)
    return np.column_stack([img_width / 2.0 + f * local[:, 0] / -local[:, 2],
                            img_height / 2.0 - f * local[:, 1] / -local[:, 2]])


def convex_silhouette(point_sets, cam_pos, cam_mat, fovy_deg, img_width, img_height):
    """Pixels (bool image) inside the convex hull of each set of world points (a
    robot link's collision hull, a held box's corners), projected into the camera."""
    mask = np.zeros((img_height, img_width), dtype=bool)
    for pts in point_sets:
        uv = project_points(pts, cam_pos, cam_mat, fovy_deg, img_width, img_height)
        if len(uv) < 3:
            continue
        try:
            hull = ConvexHull(uv)
        except QhullError:
            continue
        u0, v0 = np.maximum(np.floor(uv.min(axis=0)).astype(int), 0)
        u1 = min(int(np.ceil(uv[:, 0].max())), img_width - 1)
        v1 = min(int(np.ceil(uv[:, 1].max())), img_height - 1)
        if u1 < u0 or v1 < v0:
            continue
        vs, us = np.mgrid[v0:v1 + 1, u0:u1 + 1]
        inside = np.ones(us.shape, dtype=bool)
        for a, b, c in hull.equations:  # a u + b v + c <= 0 inside
            inside &= a * us + b * vs + c <= 1e-9
        mask[v0:v1 + 1, u0:u1 + 1] |= inside
    return mask


def region_mask(points, boxes, margin):
    """True where a point is inside any region, grown by margin in XY. A region is
    ((x0, x1), (y0, y1)), any height, or ((x0, x1), (y0, y1), z_max), only
    below z_max (so a person leaning over it is not masked).
    """
    mask = np.zeros(points.shape[:-1], dtype=bool)
    for box in boxes:
        (x0, x1), (y0, y1) = box[0], box[1]
        inside = ((points[..., 0] >= x0 - margin) & (points[..., 0] <= x1 + margin)
                  & (points[..., 1] >= y0 - margin) & (points[..., 1] <= y1 + margin))
        if len(box) > 2:
            inside &= points[..., 2] <= box[2]
        mask |= inside
    return mask


def detect_blobs(depth, cam_pos, cam_mat, fovy_deg, floor_z, min_height,
                 robot_mask=None, robot_dilate_px=2, mask_boxes=(), mask_margin=0.0,
                 min_blob_px=25, merge_px=3, return_masks=False):
    """Foreground blobs in the world frame, one row each: [x, y, z, radius, n_px,
    z_max].

    The camera sees only upper surfaces, so treat a blob as a column from the
    floor to z_max. [x, y, z] is a sphere centre one radius below z_max, for
    consumers that take spheres (the OCP slots).
    """
    points = unproject_depth(depth, cam_pos, cam_mat, fovy_deg)
    above = (points[..., 2] > floor_z + min_height) & (np.asarray(depth) > 0)  # 0: no reading
    fg = above.copy()
    robot = np.zeros_like(fg)
    if robot_mask is not None and robot_mask.any():
        robot = ndimage.binary_dilation(robot_mask, iterations=robot_dilate_px)
        fg &= ~robot
    ignored = region_mask(points, mask_boxes, mask_margin) if mask_boxes else np.zeros_like(fg)
    fg &= ~ignored

    # Pixels within merge_px join one blob: a person's thin arms otherwise came
    # out as separate blobs.
    grouped = ndimage.binary_dilation(fg, iterations=merge_px) if merge_px > 0 else fg
    labels, n = ndimage.label(grouped, structure=np.ones((3, 3)))
    labels[~fg] = 0
    blobs = []
    for i in range(1, n + 1):
        sel = labels == i
        n_px = int(sel.sum())
        if n_px < min_blob_px:
            continue
        pts = points[sel]
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        cx, cy = (lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0
        # Enclosing radius; half the bounding box cut corners (0.23 m vs a true 0.29 m).
        r = float(np.max(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)))
        blobs.append([cx, cy, hi[2] - r, r, n_px, hi[2]])
    blobs = np.asarray(blobs, dtype=float).reshape(-1, 6)
    if return_masks:
        # For mujoco_sim_node's obstacle view.
        return blobs, {"points": points, "above": above, "robot": robot,
                       "ignored": ignored, "foreground": fg}
    return blobs
