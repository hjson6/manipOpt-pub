import numpy as np

from perception import obstacle_detection as od

CAM_POS = np.array([0.0, 0.0, 2.0])
CAM_MAT = np.eye(3)  # local x/y on world x/y, looking down -z (MuJoCo convention)
FOVY = 70.0
H, W = 240, 320


def _render_floor_and_sphere(center, r):
    """Analytic depth of a floor plane z=0 plus one sphere, for a camera
    looking straight down -- the same pinhole convention the detector
    inverts."""
    f = (H / 2.0) / np.tan(np.deg2rad(FOVY) / 2.0)
    u = np.arange(W)[None, :]
    v = np.arange(H)[:, None]
    dx, dy = (u - W / 2.0) / f, -(v - H / 2.0) / f  # ray = (dx, dy, -1) * depth
    depth = np.full((H, W), CAM_POS[2])  # floor: depth == camera height
    # Ray/sphere intersection along direction d=(dx,dy,-1), origin CAM_POS.
    d = np.stack([dx + 0 * dy, dy + 0 * dx, -np.ones((H, W))], axis=-1)
    oc = CAM_POS - np.asarray(center)
    b = (d @ oc)
    a = (d * d).sum(-1)
    disc = b * b - a * (oc @ oc - r * r)
    hit = disc > 0
    t = np.where(hit, (-b - np.sqrt(np.where(hit, disc, 0))) / a, np.inf)
    return np.where(hit, np.minimum(depth, t), depth)  # t is in units of depth (d_z == -1)


def test_empty_floor_has_no_blobs():
    depth = np.full((H, W), CAM_POS[2])
    assert od.detect_blobs(depth, CAM_POS, CAM_MAT, FOVY, 0.0, 0.05).shape == (0, 6)


def test_sphere_recovered_in_world_frame():
    center, r = (0.3, -0.2, 0.55), 0.12
    depth = _render_floor_and_sphere(center, r)
    blobs = od.detect_blobs(depth, CAM_POS, CAM_MAT, FOVY, 0.0, 0.05)
    assert len(blobs) == 1
    x, y, z, rad, n_px, z_max = blobs[0]
    assert abs(x - 0.3) < 0.02 and abs(y + 0.2) < 0.02
    assert abs(rad - r) < 0.03
    assert abs(z - 0.55) < 0.05  # visible-cap estimate of the centre height


def test_robot_mask_and_region_mask_remove_blobs():
    depth = _render_floor_and_sphere((0.3, -0.2, 0.55), 0.12)
    everything = np.ones((H, W), dtype=bool)
    assert len(od.detect_blobs(depth, CAM_POS, CAM_MAT, FOVY, 0.0, 0.05,
                               robot_mask=everything)) == 0
    box = ((0.1, 0.5), (-0.4, 0.0))
    assert len(od.detect_blobs(depth, CAM_POS, CAM_MAT, FOVY, 0.0, 0.05,
                               mask_boxes=(box,))) == 0


def test_nearby_pieces_merge_into_one_object():
    # two small spheres 0.22 m apart centre to centre (a ~2 cm gap): one object
    d1 = _render_floor_and_sphere((0.30, 0.0, 1.3), 0.10)
    d2 = _render_floor_and_sphere((0.30, 0.22, 1.3), 0.10)
    blobs = od.detect_blobs(np.minimum(d1, d2), CAM_POS, CAM_MAT, FOVY, 0.0, 0.05)
    assert len(blobs) == 1
    assert abs(blobs[0][3] - 0.21) < 0.03  # radius spans both pieces


def test_two_separate_objects_give_two_blobs():
    d1 = _render_floor_and_sphere((0.5, 0.3, 0.5), 0.1)
    d2 = _render_floor_and_sphere((-0.5, -0.3, 0.5), 0.1)
    blobs = od.detect_blobs(np.minimum(d1, d2), CAM_POS, CAM_MAT, FOVY, 0.0, 0.05)
    assert len(blobs) == 2


def test_height_limited_mask_keeps_what_is_above_the_region():
    depth = _render_floor_and_sphere((0.3, -0.2, 0.55), 0.12)
    low = ((0.1, 0.5), (-0.4, 0.0), 0.3)   # region masked only below 0.3 m
    high = ((0.1, 0.5), (-0.4, 0.0), 1.0)
    assert len(od.detect_blobs(depth, CAM_POS, CAM_MAT, FOVY, 0.0, 0.05, mask_boxes=(low,))) == 1
    assert len(od.detect_blobs(depth, CAM_POS, CAM_MAT, FOVY, 0.0, 0.05, mask_boxes=(high,))) == 0
