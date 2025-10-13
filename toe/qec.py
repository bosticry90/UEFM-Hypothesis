from __future__ import annotations
import itertools
from typing import Iterable, Tuple, Any, List
import numpy as np


# ----------------------------
# Random isometry (Haar via QR)
# ----------------------------
def random_isometry(
    d_in: int | None = None,
    d_out: int | None = None,
    seed: int | None = None,
    **kwargs,
) -> np.ndarray:
    """
    Sample a Haar-random isometry V : C^{d_in} -> C^{d_out}.
    Returns array of shape (d_out, d_in) with V^† V = I_{d_in}.

    Accepts legacy aliases from tests:
      - d_logical -> d_in
      - d_physical -> d_out
    """
    # Accept aliases used by tests
    if "d_logical" in kwargs and d_in is None:
        d_in = kwargs.pop("d_logical")
    if "d_physical" in kwargs and d_out is None:
        d_out = kwargs.pop("d_physical")
    if kwargs:
        # Any other unexpected kwargs should raise clearly
        raise TypeError(f"Unexpected keyword arguments: {list(kwargs.keys())}")

    if d_in is None or d_out is None:
        raise TypeError("random_isometry requires d_in/d_out (or d_logical/d_physical)")

    if d_out < d_in:
        raise ValueError("Need d_out >= d_in to embed an isometry.")

    rng = np.random.default_rng(seed)
    # Ginibre -> QR -> Haar
    Z = (rng.normal(size=(d_out, d_out)) + 1j * rng.normal(size=(d_out, d_out))) / np.sqrt(2.0)
    Q, R = np.linalg.qr(Z)
    # Fix random phases to make Q Haar
    diag = np.diag(R)
    phases = np.ones_like(diag)
    nz = np.abs(diag) > 0
    phases[nz] = diag[nz] / np.abs(diag[nz])
    Q = Q * phases
    return Q[:, :d_in]


# -------------------------------------------
# Toy erasure error set (projector-based, up to k sites)
# -------------------------------------------
def erasure_errors(
    n_physical: int,
    k: int = 1,
    local_dim: int = 2,
) -> List[Tuple[np.ndarray, Tuple[int, ...]]]:
    """
    Build a compact Kraus set that models erasures on up to k sites in an n-site register.
    For each subset S (|S|<=k), create a single Kraus operator:
      E_S = ( ⊗_{i in S} |0><0| ) ⊗ ( ⊗_{j notin S} I )
    Returns a list of (E_S, support_tuple).

    NOTE: This is a toy model (not CPTP-complete for all channels).
    """
    if k < 0 or k > n_physical:
        raise ValueError("k must be between 0 and n_physical.")

    d_tot = local_dim ** n_physical
    # |0><0|
    P0 = np.zeros((local_dim, local_dim), dtype=complex)
    P0[0, 0] = 1.0
    I1 = np.eye(local_dim, dtype=complex)

    def _kron_n(ops: Iterable[np.ndarray]) -> np.ndarray:
        out = np.array([[1.0 + 0j]])
        for op in ops:
            out = np.kron(out, op)
        return out

    kraus: List[Tuple[np.ndarray, Tuple[int, ...]]] = []
    for r in range(0, k + 1):
        for S in itertools.combinations(range(n_physical), r):
            ops = []
            for i in range(n_physical):
                ops.append(P0 if i in S else I1)
            E = _kron_n(ops)
            assert E.shape == (d_tot, d_tot)
            kraus.append((E, S))
    return kraus


# ---------------------------------------------------
# Knill–Laflamme condition checks with shape adaptation
# ---------------------------------------------------
def _embed_to_dim(E: np.ndarray, target_dim: int) -> np.ndarray:
    """
    Embed or truncate a square operator E to a target_dim x target_dim matrix:
      - If E is smaller, pad as E ⊕ I to reach target_dim.
      - If E is larger, take the top-left target_dim x target_dim block.
      - If equal, return E.
    """
    d = E.shape[0]
    if d == target_dim:
        return E
    if d < target_dim:
        out = np.eye(target_dim, dtype=complex)
        out[:d, :d] = E
        return out
    # d > target_dim
    return E[:target_dim, :target_dim]


def knill_laflamme_checks(
    V: np.ndarray,
    errors: Iterable[np.ndarray] | Iterable[Tuple[np.ndarray, Any]],
    atol: float = 1e-8
) -> tuple[bool, np.ndarray]:
    """
    Verify KL conditions for an encoding isometry V and a collection of error operators {E_a}.
    Condition: V^† E_a^† E_b V = c_ab * I (proportional to identity on logical space).

    Returns
    -------
    ok : bool
        True if all pairs (a,b) satisfy KL within tolerance.
    C : np.ndarray
        Matrix of scalars c_ab (len(E) x len(E)).
    """
    # normalize ops to a list of pure operators
    ops: list[np.ndarray] = []
    for item in errors:
        E = item[0] if isinstance(item, tuple) else item
        E = np.asarray(E, dtype=complex)
        if E.ndim != 2 or E.shape[0] != E.shape[1]:
            raise ValueError("Each error operator must be square.")
        ops.append(E)

    d_out, d_in = V.shape
    # orthonormalize columns just in case
    Q, _ = np.linalg.qr(V)
    V = Q[:, :d_in]

    ops = [_embed_to_dim(E, d_out) for E in ops]

    m = len(ops)
    C = np.zeros((m, m), dtype=complex)
    ok_all = True
    I_in = np.eye(d_in, dtype=complex)

    def _prop_to_I(M: np.ndarray) -> tuple[bool, float, float]:
        alpha = np.trace(M) / d_in
        R = M - alpha * I_in
        offdiag = np.max(np.abs(R - np.diag(np.diag(R))))
        diag_std = float(np.std(np.real(np.diag(M))))
        ok = (offdiag <= atol) and (diag_std <= atol)
        return bool(ok), float(offdiag), diag_std

    for a, Ea in enumerate(ops):
        for b, Eb in enumerate(ops):
            M = V.conj().T @ Ea.conj().T @ Eb @ V  # d_in x d_in
            ok_pair, _, _ = _prop_to_I(M)
            ok_all = ok_all and ok_pair
            C[a, b] = np.trace(M) / d_in

    return bool(ok_all), C


# ---------------------------------------------------
# Naive decoder baseline for erasure sets
# ---------------------------------------------------
def naive_erasure_decoder(V: np.ndarray, errors: list[np.ndarray] | list[tuple[np.ndarray, Any]], atol=1e-6) -> bool:
    """
    Naive (pseudoinverse-based) decoder that attempts to recover the code subspace
    after applying each error E. We test whether the recovery map approximately
    fixes the isometric image:  R_E V ≈ V.

    R_E := V (V^† E^† E V)^+ V^† E^† E

    where (.)^+ is Moore–Penrose pseudoinverse.
    """
    d_out, d_in = V.shape
    Q, _ = np.linalg.qr(V)
    V = Q[:, :d_in]

    raw_ops: list[np.ndarray] = []
    for item in errors:
        E = item[0] if isinstance(item, tuple) else item
        E = np.asarray(E, dtype=complex)
        E = _embed_to_dim(E, d_out)
        raw_ops.append(E)

    def op_norm(A: np.ndarray) -> float:
        return float(np.linalg.svd(A, compute_uv=False, hermitian=False)[0])

    worst = 0.0
    for E in raw_ops:
        G = V.conj().T @ E.conj().T @ E @ V  # d_in x d_in
        Gpinv = np.linalg.pinv(G)
        R = V @ Gpinv @ V.conj().T @ E.conj().T @ E
        dev = op_norm(R @ V - V)
        worst = max(worst, dev)
        if worst > atol:
            return False
    return True
