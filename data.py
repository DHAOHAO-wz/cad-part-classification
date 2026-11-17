"""Synthetic CAD-like point cloud generator (prototype placeholder).

In the final project this module will be replaced by loading real meshes /
point clouds from a public CAD dataset (e.g. Fusion 360 Gallery, ABC subset).
Here four simple part categories are synthesized so the feature + classifier
pipeline can be exercised end-to-end without external data.
"""

import numpy as np

CATEGORIES = ["block", "bracket_L", "cylinder", "plate_holes"]


def _box(n, lo, hi, rng):
    return rng.uniform(lo, hi, size=(n, 3))


def _cylinder(n, radius, height, rng):
    theta = rng.uniform(0, 2 * np.pi, n)
    z = rng.uniform(0, height, n)
    # mix side surface and two end caps
    side = rng.random(n) < 0.7
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    pts = np.stack([x, y, z], axis=1)
    cap = ~side
    if cap.any():
        r = radius * np.sqrt(rng.random(cap.sum()))
        a = rng.uniform(0, 2 * np.pi, cap.sum())
        pts[cap] = np.stack([r * np.cos(a), r * np.sin(a),
                             height * (rng.random(cap.sum()) < 0.5)], axis=1)
    return pts


def _bracket(n, rng):
    # L-shape: vertical plate + horizontal plate, union sampling
    n1 = n // 2
    v = np.column_stack([
        rng.uniform(0, 0.4, n1),
        rng.uniform(0, 2.0, n1),
        rng.uniform(0, 2.0, n1),
    ])
    h = np.column_stack([
        rng.uniform(0, 2.0, n - n1),
        rng.uniform(0, 0.4, n - n1),
        rng.uniform(0, 0.6, n - n1),
    ])
    return np.vstack([v, h])


def _plate_holes(n, rng):
    # thin plate with a few cylindrical holes approximated by rejection
    pts = []
    holes = [(0.7, 0.7, 0.18), (1.5, 1.5, 0.18), (1.5, 0.7, 0.14)]
    while len(pts) < n:
        p = rng.uniform([0, 0, 0], [2.2, 2.2, 0.25])
        inside = False
        for hx, hy, hr in holes:
            if (p[0] - hx) ** 2 + (p[1] - hy) ** 2 < hr ** 2:
                inside = True
                break
        if not inside:
            pts.append(p)
    return np.array(pts[:n])


def make_dataset(n_per_class=120, n_points=2048, seed=42):
    """Return point clouds array (M, n_points, 3) and integer labels (M,)."""
    rng = np.random.default_rng(seed)
    clouds, labels = [], []
    for ci, cat in enumerate(CATEGORIES):
        for _ in range(n_per_class):
            if cat == "block":
                lo = rng.uniform(-0.2, 0.2, 3)
                hi = lo + rng.uniform(1.2, 2.2, 3)
                pc = _box(n_points, lo, hi, rng)
            elif cat == "bracket_L":
                pc = _bracket(n_points, rng)
            elif cat == "cylinder":
                pc = _cylinder(n_points,
                               radius=rng.uniform(0.6, 0.9),
                               height=rng.uniform(1.2, 2.2), rng=rng)
            else:
                pc = _plate_holes(n_points, rng)
            # small rigid jitter to simulate scanned noise
            pc = pc + rng.normal(0, 0.01, pc.shape)
            clouds.append(pc)
            labels.append(ci)
    return np.array(clouds), np.array(labels)
