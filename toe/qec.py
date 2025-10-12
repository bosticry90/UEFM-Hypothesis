from __future__ import annotations
import itertools
from typing import Iterable, Tuple, Any, List
import numpy as np


def random_isometry(
    d_in: int | None = None,
    d_out: int | None = None,
    seed: int | None = None,
    **kwargs
) -> np.ndarray:
    """
    Haar-random isometry V : C^{d_in} -> C^{d_out}.
    Accepts aliases: d_logical -> d_in, d_physical -> d_out.
    Returns array (d_out, d_in) with V^† V = I_{d_in}.
    """
    if d_in is None:
        d_in = kwargs.get("d_logical")
    if d_out is None:
        d_out = kwargs.get("d_physical")
    if d_in is None or d_out is None:
        raise TypeError("random_isometry requires (d_in, d_out) or (d_logical, d_physical).")
    if d_out < d_in:
        raise ValueError("Need d_out >= d_in to embed an isometry.")

    rng = np.random.default_rng(seed)
    Z = (rng.normal(size=(d_out, d_out)) + 1j * rng.normal(size=(d_out, d_out))) / np.sqrt(2.0)
    Q, R = np.linalg.qr(Z)
    diag = np.diag(R)
    phases = np.ones_like(diag, dtype=complex)
    nz = np.abs(diag) > 0
    phases[nz] = diag[nz] / np.abs(diag[nz])
    Q = Q * phases
    return Q[:, :d_in]


def _kron_n(ops: Iterable[np.ndarray]) -> np.ndarray:
    out = np.array([[1.0 + 0j]])
    for op in ops:
        out = np.kron(out, op)
    return out


def _proj_zero(d: int) -> np.ndarray:
    P = np.zeros((d, d), dtype=complex)
    P[0, 0] = 1.0
    return P


def identity(d: int) -> np.ndarray:
    return np.eye(d, dtype=complex)


def erasure_errors(n_physical: int, k: int = 1, local_dim: int = 2) -> List[Tuple[np.ndarray, Tuple[int, ...]]]:
    """
    Toy erasure Kraus set for up to k sites among n_physical.
    Returns list of (E_S, S) with E_S = (⊗_{i∈S} |0><0|) ⊗ (⊗_{j∉S} I).
    """
    if k < 0 or k > n_physical:
        raise ValueError("k must be between 0 and n_physical.")
    d_tot = local_dim ** n_physical
    P0 = _proj_zero(local_dim)
    I1 = identity(local_dim)

    kraus = []
    for r in range(k + 1):
        for S in itertools.combinations(range(n_physical), r):
            blocks = [P0 if i in S else I1 for i in range(n_physical)]
            E = _kron_n(blocks)
            assert E.shape == (d_tot, d_tot)
            kraus.append((E, S))
    return kraus


def _proportional_to_identity(M: np.ndarray, atol: float) -> tuple[bool, float, float]:
    d = M.shape[0]
    I = np.eye(d, dtype=complex)
    alpha = np.trace(M) / d
    R = M - alpha * I
    max_offdiag = np.max(np.abs(R - np.diag(np.diag(R))))
    diag_var = np.std(np.real(np.diag(M)))
    ok = (max_offdiag <= atol) and (diag_var <= atol)
    return ok, float(max_offdiag), float(diag_var)


def _match_dim(E: np.ndarray, d_target: int) -> np.ndarray:
    """
    Match operator to physical dim d_target by padding with identity (if smaller)
    or taking the top-left block (if larger).
    """
    dE = E.shape[0]
    if dE == d_target:
        return E
    if dE < d_target:
        pad = np.eye(d_target - dE, dtype=complex)
        return np.block([
            [E, np.zeros((dE, d_target - dE), dtype=complex)],
            [np.zeros((d_target - dE, dE), dtype=complex), pad],
        ])
    return E[:d_target, :d_target]


def knill_laflamme_checks(
    V: np.ndarray,
    errors: Iterable[np.ndarray] | Iterable[Tuple[np.ndarray, Any]],
    atol: float = 1e-8
) -> tuple[bool, np.ndarray]:
    """
    Check KL: V^† E_a^† E_b V = c_ab I  for all a,b.
    Returns (ok, C) where C[a,b] = (1/d_in) Tr[V^† E_a^† E_b V].
    """
    ops: list[np.ndarray] = []
    for item in errors:
        E = item[0] if isinstance(item, tuple) else item
        ops.append(np.asarray(E, dtype=complex))

    d_out, d_in = V.shape
    Q, _ = np.linalg.qr(V)
    V = Q[:, :d_in]

    ops = [_match_dim(E, d_out) for E in ops]

    m = len(ops)
    C = np.zeros((m, m), dtype=complex)
    ok = True

    for a, Ea in enumerate(ops):
        for b, Eb in enumerate(ops):
            M = V.conj().T @ Ea.conj().T @ Eb @ V
            good, offd, dstd = _proportional_to_identity(M, atol=atol)
            ok = ok and good
            C[a, b] = np.trace(M) / d_in

    # Ensure Python bool (not numpy.bool_) for tests like isinstance(ok, bool)
    return bool(ok), C
