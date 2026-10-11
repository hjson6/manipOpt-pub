"""A learned policy at runtime, numpy only (float64): an MLP exported from training as an
.npz beside its manifest (.json). The .npz holds the layers (tanh between them, the last
linear), the observation normalization frozen at export (VecNormalize's mean, variance,
epsilon and clip) and the feature version; the manifest how it was made and the .npz's
sha256, which must match, so a run named after a model is that model. Deterministic: the
mean action, clipped to [-1, 1]. Shared by the learned policies' layers (handover_notes/rl).
"""
import hashlib
import json
from pathlib import Path

import numpy as np


class PolicyFileError(RuntimeError):
    pass


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(npz_path, layers, obs_mean, obs_var, obs_eps, obs_clip, feature_version, manifest):
    """Writes a policy: layers [(W (out, in), b)], the normalization, the feature version;
    the manifest (a dict) gets the .npz's sha256 and goes beside it."""
    npz_path = Path(npz_path)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f"w{i}": np.asarray(w, dtype=float) for i, (w, _) in enumerate(layers)}
    arrays.update({f"b{i}": np.asarray(b, dtype=float) for i, (_, b) in enumerate(layers)})
    np.savez(npz_path, n_layers=len(layers), obs_mean=np.asarray(obs_mean, dtype=float),
             obs_var=np.asarray(obs_var, dtype=float), obs_eps=float(obs_eps), obs_clip=float(obs_clip),
             feature_version=int(feature_version), **arrays)
    npz_path.with_suffix(".json").write_text(json.dumps({**manifest, "sha256": sha256(npz_path)}, indent=1) + "\n")


class LearnedPolicy:
    def __init__(self, npz_path, feature_version=None):
        """Loads policies/<layer>/<name>.npz and its manifest; refuses (PolicyFileError) a
        missing file, a sha256 that does not match, or another feature version."""
        npz_path = Path(npz_path)
        manifest_path = npz_path.with_suffix(".json")
        for p in (npz_path, manifest_path):
            if not p.exists():
                raise PolicyFileError(f"policy {npz_path.stem}: {p} not found")
        self.manifest = json.loads(manifest_path.read_text())
        self.sha256 = sha256(npz_path)
        if self.manifest.get("sha256") != self.sha256:
            raise PolicyFileError(f"policy {npz_path.stem}: the .npz's sha256 {self.sha256[:8]} is not its "
                                  f"manifest's {str(self.manifest.get('sha256'))[:8]}")
        with np.load(npz_path) as z:
            self.mean, self.var = z["obs_mean"], z["obs_var"]
            self.eps, self.clip = float(z["obs_eps"]), float(z["obs_clip"])
            self.layers = [(z[f"w{i}"], z[f"b{i}"]) for i in range(int(z["n_layers"]))]
            self.feature_version = int(z["feature_version"])
        if feature_version is not None and self.feature_version != feature_version:
            raise PolicyFileError(f"policy {npz_path.stem}: features v{self.feature_version}, this code builds "
                                  f"v{feature_version}")
        self.name = npz_path.stem
        self.obs_dim, self.act_dim = len(self.mean), len(self.layers[-1][1])

    def __call__(self, obs):
        x = np.clip((np.asarray(obs, dtype=float) - self.mean) / np.sqrt(self.var + self.eps), -self.clip, self.clip)
        for w, b in self.layers[:-1]:
            x = np.tanh(w @ x + b)
        w, b = self.layers[-1]
        return np.clip(w @ x + b, -1.0, 1.0)
