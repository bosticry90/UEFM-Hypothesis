# toe/noise.py
from __future__ import annotations
import numpy as np
from typing import Literal

Prng = np.random.Generator | None

def bit_flip(psi: np.ndarray, p: float, rng: Prng = None) -> np.ndarray:
    """
    Apply independent X flips to computational-basis qubits of a 2^n statevector
    by Monte-Carlo sampling a single trajectory (efficient, no Kraus matmul).
    """
    if p <= 0:  # fast path
        return psi
    if p >= 1:
        # flip every qubit once is equivalent to bitwise NOT: reverse all indices
        return psi[::-1].copy()

    rng = rng or np.random.default_rng()
    psi_out = psi.copy()
    # For each qubit, with prob p, swap amplitude pairs differing on that bit.
    # State vector length must be power of two.
    N = int(np.log2(psi_out.size))
    if 2**N != psi_out.size:
        raise ValueError("State vector length is not a power of two.")
    for qb in range(N):
        if rng.random() < p:
            stride = 1 << qb
            block = stride << 1
            # swap [k : k+stride] with [k+stride : k+2*stride] for all blocks
            for start in range(0, psi_out.size, block):
                a = slice(start, start + stride)
                b = slice(start + stride, start + 2*stride)
                psi_out[a], psi_out[b] = psi_out[b].copy(), psi_out[a].copy()
    return psi_out

def phase_flip(psi: np.ndarray, p: float, rng: Prng = None) -> np.ndarray:
    """
    Independent Z flips: multiply amplitudes where bit=1 by -1 when the qubit flips.
    """
    if p <= 0:
        return psi
    rng = rng or np.random.default_rng()
    psi_out = psi.copy()
    N = int(np.log2(psi_out.size))
    if 2**N != psi_out.size:
        raise ValueError("State vector length is not a power of two.")

    # Precompute Gray-code masks for speed
    idx = np.arange(psi_out.size, dtype=np.uint64)
    for qb in range(N):
        if rng.random() < p:
            mask = (idx >> qb) & 1
            psi_out[mask == 1] *= -1.0
    return psi_out

def depolarizing(psi: np.ndarray, p: float, rng: Prng = None) -> np.ndarray:
    """
    Single-trajectory channel: with prob p on each qubit, pick uniformly at random
    one of {X,Y,Z} and apply it (X and Z as above; Y = iXZ).
    """
    if p <= 0:
        return psi
    rng = rng or np.random.default_rng()
    psi_out = psi.copy()
    N = int(np.log2(psi_out.size))
    if 2**N != psi_out.size:
        raise ValueError("State vector length is not a power of two.")
    idx = np.arange(psi_out.size, dtype=np.uint64)

    def _x_flip_in_place(qb: int):
        stride = 1 << qb
        block = stride << 1
        for start in range(0, psi_out.size, block):
            a = slice(start, start + stride)
            b = slice(start + stride, start + 2*stride)
            psi_out[a], psi_out[b] = psi_out[b].copy(), psi_out[a].copy()

    def _z_flip_in_place(qb: int):
        mask = (idx >> qb) & 1
        psi_out[mask == 1] *= -1.0

    for qb in range(N):
        if rng.random() < p:
            gate = rng.integers(0, 3)  # 0->X, 1->Y, 2->Z
            if gate == 0:
                _x_flip_in_place(qb)
            elif gate == 2:
                _z_flip_in_place(qb)
            else:
                # Y = i XZ → apply Z then X, and multiply amplitudes on flipped branch by i
                # Implement as Z then X, then phase i to the swapped half.
                _z_flip_in_place(qb)
                # capture pre-swap indices to place i correctly
                stride = 1 << qb
                block = stride << 1
                for start in range(0, psi_out.size, block):
                    a = slice(start, start + stride)
                    b = slice(start + stride, start + 2*stride)
                    # swap with phase i on new 'b' (which came from 'a')
                    a_block = psi_out[a].copy()
                    b_block = psi_out[b].copy()
                    psi_out[a] = b_block
                    psi_out[b] = (1j) * a_block
    return psi_out

Channel = Literal["bitflip", "phaseflip", "depolarizing"]

def apply_channel(psi: np.ndarray, p: float, channel: Channel = "bitflip", rng: Prng = None) -> np.ndarray:
    if channel == "bitflip":
        return bit_flip(psi, p, rng)
    if channel == "phaseflip":
        return phase_flip(psi, p, rng)
    if channel == "depolarizing":
        return depolarizing(psi, p, rng)
    raise ValueError(f"Unknown channel: {channel}")
