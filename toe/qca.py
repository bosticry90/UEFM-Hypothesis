from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


def norm(psi: np.ndarray) -> float:
    """L2 norm of a state vector."""
    return float(np.vdot(psi, psi).real)


def _hadamard() -> np.ndarray:
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2.0)
    return H


def _kron_pow(op: np.ndarray, n: int) -> np.ndarray:
    out = np.array([[1.0 + 0j]])
    for _ in range(n):
        out = np.kron(out, op)
    return out


def _apply_sitewise_hadamard(psi: np.ndarray, n_sites: int) -> np.ndarray:
    Hn = _kron_pow(_hadamard(), n_sites)  # 2^N x 2^N
    return Hn @ psi


def _apply_swap_pair(psi: np.ndarray, n_sites: int, i: int, j: int) -> np.ndarray:
    """
    Apply SWAP between qubits i and j (0-based, little-endian indexing in axes).
    Implemented by reshaping to tensor (2,)*N and transposing axes i<->j.
    """
    if i == j:
        return psi
    tens = psi.reshape((2,) * n_sites)
    axes = list(range(n_sites))
    axes[i], axes[j] = axes[j], axes[i]
    tens_swapped = np.transpose(tens, axes=axes)
    return tens_swapped.reshape(-1)


def _apply_swap_layer(psi: np.ndarray, n_sites: int, pairs: List[Tuple[int, int]]) -> np.ndarray:
    out = psi
    for (i, j) in pairs:
        out = _apply_swap_pair(out, n_sites, i, j)
    return out


@dataclass
class SplitStepQCA1D:
    """
    A simple unitary, locality-preserving brickwork QCA over N qubits:
      Step = [H on all sites] -> [SWAP on even bonds] -> [SWAP on odd bonds]
    This preserves ||psi|| exactly up to floating-point error.
    """
    n_sites: int
    d: int = 2
    seed: int | None = None
    theta1: float | None = None  # kept for compatibility (unused)
    theta2: float | None = None  # kept for compatibility (unused)

    def __post_init__(self):
        if self.d != 2:
            raise ValueError("This QCA scaffold currently supports qubits (d=2) only.")
        if self.n_sites <= 0:
            raise ValueError("n_sites must be positive.")

        N = self.n_sites
        self.pairs_even = [(i, i + 1) for i in range(0, N - 1, 2)]
        self.pairs_odd = [(i, i + 1) for i in range(1, N - 1, 2)]

        if self.seed is not None and (self.theta1 is None or self.theta2 is None):
            rng = np.random.default_rng(self.seed)
            self.theta1 = float(rng.uniform(0, np.pi / 4))
            self.theta2 = float(rng.uniform(0, np.pi / 4))

    def step(self, psi: np.ndarray) -> np.ndarray:
        """Apply one QCA step to a 2^N state vector."""
        if psi.ndim != 1 or psi.shape[0] != (2 ** self.n_sites):
            raise ValueError(f"psi must be a 1D vector of length 2^{self.n_sites}")
        out = _apply_sitewise_hadamard(psi, self.n_sites)
        out = _apply_swap_layer(out, self.n_sites, self.pairs_even)
        out = _apply_swap_layer(out, self.n_sites, self.pairs_odd)
        return out


def energy_conservation_proxy(psi0: np.ndarray, qca: SplitStepQCA1D, steps: int = 100) -> np.ndarray:
    """
    Return the sequence of norms over time: [||psi0||, ||psi1||, ..., ||psi_steps||].
    Tests compare that all entries equal the initial value (unitarity).
    """
    norms = [norm(psi0)]
    psi = psi0.copy()
    for _ in range(int(steps)):
        psi = qca.step(psi)
        norms.append(norm(psi))
    return np.array(norms, dtype=float)
