from __future__ import annotations
import itertools
from typing import Iterable, Tuple, Any, Dict, List
import numpy as np


def random_isometry(
    d_logical: int | None = None,
    d_physical: int | None = None,
    seed: int | None = None,
    *,
    d_in: int | None = None,
    d_out: int | None = None,
) -> np.ndarray:
    """
    Test-friendly signature:
      random_isometry(d_logical=..., d_physical=..., seed=...)
    Back-compat aliases:
      d_in = d_logical, d_out = d_physical
    Returns V shape (d_physical, d_logical) with V^† V = I.
    """
    if d_in is None and d_logical is None:
        raise ValueError("Provide d_logical (or d_in).")
    if d_out is None and d_physical is None:
        raise ValueError("Provide d_physical (or d_out).")
    d_in = int(d_logical if d_logical is not None else d_in)  # columns
    d_out = int(d_physical if d_physical is not None else d_out)  # rows
    if d_out < d_in:
        raise ValueError("Need d_physical >= d_logical.")
    rng = np.random.default_rng(seed)

    # Haar unitary via QR of Ginibre
    Z = (rng.normal(size=(d_out, d_out)) + 1j * rng.normal(size=(d_out, d_out))) / np.sqrt(2.0)
    Q, R = np.linalg.qr(Z)
    diag = np.diag(R)
    phases = np.ones_like(diag)
    nz = np.abs(diag) > 0
    phases[nz] = diag[nz] / np.abs(diag[nz])
    Q = Q * phases
    return Q[:, :d_in]


def erasure_errors(
    n_physical: int | None = None,
    *,
    n: int | None = None,
    k: int = 1,
    local_dim: int = 2
) -> List[Tuple[np.ndarray, Tuple[int, ...]]]:
    """
    Test-friendly signature:
      erasure_errors(n_physical=3, k=1, local_dim=2)
    Returns list of (E_S, S) Kraus ops projecting erased sites to |0><0|.
    """
    n = int(n_physical if n_physical is not None else n)
    if k < 0 or k > n:
        raise ValueError("k must be between 0 and n.")
    d_tot = local_dim ** n

    def proj_zero(d: int) -> np.ndarray:
        P = np.zeros((d, d), dtype=complex)
        P[0, 0] = 1.0
        return P

    I1 = np.eye(local_dim, dtype=complex)
    P0 = proj_zero(local_dim)

    def kron_all(ops):
        out = np.array([[1.0+0j]])
        for op in ops:
            out = np.kron(out, op)
        return out

    kraus = []
    for r in range(0, k + 1):
        for S in itertools.combinations(range(n), r):
            ops = []
            for i in range(n):
                ops.append(P0 if i in S else I1)
            E = kron_all(ops)
            assert E.shape == (d_tot, d_tot)
            kraus.append((E, S))
    return kraus


def knill_laflamme_checks(
    V: np.ndarray,
    errors: Iterable[np.ndarray] | Iterable[Tuple[np.ndarray, Any]],
    atol: float = 1e-8
) -> tuple[bool, np.ndarray]:
    """
    KL condition:  V^† E_a^† E_b V = c_ab I  for all a,b.
    Returns:
      ok : bool
      C  : complex matrix (len(E) x len(E)) of the scalars c_ab
    """
    # normalize ops to a list of pure operators
    ops: list[np.ndarray] = []
    for item in errors:
        E = item[0] if isinstance(item, tuple) else item
        ops.append(np.asarray(E, dtype=complex))

    d_out, d_in = V.shape
    # orthonormalize columns if needed
    Q, _ = np.linalg.qr(V)
    V = Q[:, :d_in]

    m = len(ops)
    C = np.zeros((m, m), dtype=complex)
    ok = True

    I_in = np.eye(d_in, dtype=complex)
    for a, Ea in enumerate(ops):
        for b, Eb in enumerate(ops):
            M = V.conj().T @ Ea.conj().T @ Eb @ V  # d_in x d_in
            # best-fit scalar (in least-squares sense) is alpha = Tr(M)/d_in
            alpha = np.trace(M) / d_in
            C[a, b] = alpha
            # check proportional-to-identity
            R = M - alpha * I_in
            offdiag = np.max(np.abs(R - np.diag(np.diag(R))))
            diag_std = np.std(np.real(np.diag(M)))
            if offdiag > atol or diag_std > atol:
                ok = False

    return ok, C
