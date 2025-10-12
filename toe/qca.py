from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def _coin(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],
                     [s,  c]], dtype=complex)


def norm(psi: np.ndarray) -> float:
    """L2 norm of a state vector."""
    return float(np.vdot(psi, psi).real)


@dataclass
class SplitStepQCA1D:
    """
    1D split-step QCA on a ring of n_sites qubits:
      U = S_down * C(theta2) * S_up * C(theta1)
    Acts on a 2^n complex vector in the computational basis.
    """
    n_sites: int
    d:       int
    seed:    int | None = None
    theta1:  float | None = None
    theta2:  float | None = None

    def __post_init__(self):
        if self.n_sites <= 0:
            raise ValueError("n_sites must be positive.")
        if self.d != 2:
            raise ValueError("This toy QCA supports qubits (d=2) only.")
        rng = np.random.default_rng(self.seed)
        if self.theta1 is None:
            self.theta1 = float(rng.uniform(0.1, 0.4))
        if self.theta2 is None:
            self.theta2 = float(rng.uniform(0.2, 0.5))
        self.C1 = _coin(self.theta1)
        self.C2 = _coin(self.theta2)

    def _apply_coin_layer(self, psi: np.ndarray, C: np.ndarray) -> np.ndarray:
        n = self.n_sites
        out = psi.reshape([2] * n)
        for site in range(n):
            out = np.moveaxis(out, site, 0)
            out = (C @ out.reshape(2, -1)).reshape(2, *out.shape[1:])
            out = np.moveaxis(out, 0, site)
        return out.reshape(-1)

    def _shift_up_right(self, psi: np.ndarray) -> np.ndarray:
        """
        Shift the |1> component of each site one position to the right (periodic).
        Careful with axes: after fixing axis i to 1, remaining tensor has n-1 axes;
        the neighbor axis index becomes (i if i+1 < n else 0).
        """
        n = self.n_sites
        T = psi.reshape([2] * n)
        out = T.copy()
        for i in range(n):
            sl1 = [slice(None)] * n
            sl1[i] = 1
            block1 = out[tuple(sl1)]
            axis_neighbor = i if (i + 1) < n else 0
            block1 = np.roll(block1, shift=+1, axis=axis_neighbor)
            out[tuple(sl1)] = block1
        return out.reshape(-1)

    def _shift_down_left(self, psi: np.ndarray) -> np.ndarray:
        """
        Shift the |0> component of each site one position to the left (periodic).
        After fixing axis i to 0, neighbor axis index is (i if i+1 < n else 0).
        """
        n = self.n_sites
        T = psi.reshape([2] * n)
        out = T.copy()
        for i in range(n):
            sl0 = [slice(None)] * n
            sl0[i] = 0
            block0 = out[tuple(sl0)]
            axis_neighbor = i if (i + 1) < n else 0
            block0 = np.roll(block0, shift=-1, axis=axis_neighbor)
            out[tuple(sl0)] = block0
        return out.reshape(-1)

    def step(self, psi: np.ndarray) -> np.ndarray:
        if psi.ndim != 1 or psi.size != 2 ** self.n_sites:
            raise ValueError(f"psi must be a 1D vector of length 2^{self.n_sites}")
        x = psi
        x = self._apply_coin_layer(x, self.C1)
        x = self._shift_up_right(x)
        x = self._apply_coin_layer(x, self.C2)
        x = self._shift_down_left(x)
        return x

    # For the meta-axiom test: nearest-neighbor QCA → LR speed ≤ 1 site/step.
    def lightcone_radius(self, steps: int) -> int:
        return int(steps)


def energy_conservation_proxy(arg1, arg2, steps: int = 100):
    """
    Backward-compatible API:

    - energy_conservation_proxy(psi0, qca, steps=...) → returns array of norms over time
    - energy_conservation_proxy(qca, psi0, steps=...) → returns scalar max norm drift
    """
    # Detect call order
    if isinstance(arg1, SplitStepQCA1D):
        qca: SplitStepQCA1D = arg1
        psi0: np.ndarray = arg2
        n0 = norm(psi0)
        psi = psi0.copy()
        max_drift = 0.0
        for _ in range(int(steps)):
            psi = qca.step(psi)
            drift = abs(norm(psi) - n0)
            if drift > max_drift:
                max_drift = drift
        return float(max_drift)

    # psi-first path: return norms array
    psi0: np.ndarray = arg1
    qca: SplitStepQCA1D = arg2
    norms = [norm(psi0)]
    psi = psi0.copy()
    for _ in range(int(steps)):
        psi = qca.step(psi)
        norms.append(norm(psi))
    return np.asarray(norms, dtype=float)
