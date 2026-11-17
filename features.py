"""Interpretable geometric features extracted from a point cloud.

These hand-engineered features mirror what a CAD part's overall geometry
typically reveals: proportions, anisotropy, planarity and surface curvature.
In the final project B-rep features (face/edge counts, face-type ratios,
surface area, volume) will be added via PythonOCC where available.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors


def _pca_eigen(pts):
    centered = pts - pts.mean(axis=0)
    cov = np.cov(centered, rowvar=False)
    vals = np.linalg.eigvalsh(cov)
    vals = np.sort(np.clip(vals, 0, None))[::-1]
    return vals


def local_planarity(pts, k=16, sample=512, seed=0):
    """Ratio of planar local patches using the smallest PCA eigenvalue share."""
    rng = np.random.default_rng(seed)
    m = len(pts)
    idx = rng.choice(m, size=min(sample, m), replace=False)
    sub = pts[idx]
    nn = NearestNeighbors(n_neighbors=k).fit(pts)
    _, neigh = nn.kneighbors(sub)
    lambda3_share = []
    for ring in neigh:
        vals = _pca_eigen(pts[ring])
        tot = vals.sum() + 1e-9
        lambda3_share.append(vals[2] / tot)
    lambda3_share = np.array(lambda3_share)
    planar = (lambda3_share < 0.05).mean()
    return lambda3_share.mean(), lambda3_share.std(), planar


def extract_features(pc):
    """pc: (N, 3) -> fixed-length feature vector."""
    pc = pc - pc.mean(axis=0)
    mins, maxs = pc.min(axis=0), pc.max(axis=0)
    size = maxs - mins
    dx, dy, dz = size
    longest, shortest = size.max(), size.min()

    dist = np.linalg.norm(pc - pc.mean(axis=0), axis=1)
    vals = _pca_eigen(pc)
    v1, v2, v3 = vals + 1e-9
    mean_l3, std_l3, planar = local_planarity(pc)

    feat = [
        dx, dy, dz,
        longest / (shortest + 1e-9),          # aspect ratio
        dx / (longest + 1e-9),
        dy / (longest + 1e-9),
        dz / (longest + 1e-9),
        dist.mean(), dist.std(), dist.max(), dist.min(),
        v1 / vals.sum(),                      # anisotropy
        (v1 - v2) / v1,                       # linearity
        (v2 - v3) / v1,                       # planarity (global)
        mean_l3, std_l3, planar,              # local planarity stats
    ]
    return np.array(feat, dtype=float)


FEATURE_NAMES = [
    "size_x", "size_y", "size_z", "aspect_ratio",
    "share_x", "share_y", "share_z",
    "dist_mean", "dist_std", "dist_max", "dist_min",
    "anisotropy", "linearity", "planarity_global",
    "local_l3_mean", "local_l3_std", "planar_ratio",
]
