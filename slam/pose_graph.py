"""SE(2) pose graph: keyframe poses linked by relative-pose measurements (consecutive
scan matches and loop closures), optimized by Gauss-Newton on sparse matrices with
the first pose fixed. Framework-free (numpy, scipy).
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def relative(a, b):
    """b in a's frame, poses (x, y, yaw)."""
    c, s = np.cos(a[2]), np.sin(a[2])
    dx, dy = b[0] - a[0], b[1] - a[1]
    return np.array([c * dx + s * dy, -s * dx + c * dy, wrap(b[2] - a[2])])


def compose(a, b):
    c, s = np.cos(a[2]), np.sin(a[2])
    return np.array([a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], wrap(a[2] + b[2])])


class PoseGraph:
    def __init__(self):
        self.nodes = []
        self.edges = []  # (i, j, z (3,), info (3, 3), kind)

    def add_node(self, pose):
        self.nodes.append(np.array(pose, dtype=float))
        return len(self.nodes) - 1

    def add_edge(self, i, j, z, info, kind="odom"):
        self.edges.append((i, j, np.array(z, dtype=float), np.array(info, dtype=float), kind))

    def errors(self, x=None):
        x = np.array(self.nodes) if x is None else x
        return np.array([self._error(x[i], x[j], z) for i, j, z, _, _ in self.edges])

    @staticmethod
    def _error(xi, xj, z):
        return relative(z, relative(xi, xj)) * [1, 1, 1]

    def optimize(self, iterations=10, tol=1e-6):
        """Gauss-Newton; returns the final chi2."""
        x = np.array(self.nodes)
        n = len(x)
        if n < 2 or not self.edges:
            return 0.0
        chi2 = np.inf
        for _ in range(iterations):
            rows, cols, vals = [], [], []
            b = np.zeros(3 * n)
            chi2 = 0.0
            for i, j, z, info, _ in self.edges:
                xi, xj = x[i], x[j]
                ci, si = np.cos(xi[2]), np.sin(xi[2])
                cz, sz = np.cos(z[2]), np.sin(z[2])
                rit = np.array([[ci, si], [-si, ci]])
                rzt = np.array([[cz, sz], [-sz, cz]])
                drit = np.array([[-si, ci], [-ci, -si]])
                dt = xj[:2] - xi[:2]
                e = np.r_[rzt @ (rit @ dt - z[:2]), wrap(xj[2] - xi[2] - z[2])]
                a = np.zeros((3, 3))
                a[:2, :2] = -rzt @ rit
                a[:2, 2] = rzt @ drit @ dt
                a[2, 2] = -1.0
                bm = np.zeros((3, 3))
                bm[:2, :2] = rzt @ rit
                bm[2, 2] = 1.0
                chi2 += float(e @ info @ e)
                for (p, jp), (q, jq) in (((i, a), (i, a)), ((i, a), (j, bm)), ((j, bm), (i, a)), ((j, bm), (j, bm))):
                    blk = jp.T @ info @ jq
                    r, c = np.meshgrid(np.arange(3 * p, 3 * p + 3), np.arange(3 * q, 3 * q + 3), indexing="ij")
                    rows.append(r.ravel())
                    cols.append(c.ravel())
                    vals.append(blk.ravel())
                b[3 * i:3 * i + 3] += a.T @ info @ e
                b[3 * j:3 * j + 3] += bm.T @ info @ e
            h = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(3 * n, 3 * n))
            h = h + sp.diags(np.r_[np.full(3, 1e12), np.full(3 * n - 3, 1e-9)])  # node 0 fixed
            dx = spla.spsolve(h.tocsc(), -b).reshape(n, 3)
            x = x + dx
            x[:, 2] = wrap(x[:, 2])
            if np.abs(dx).max() < tol:
                break
        self.nodes = list(x)
        return chi2
