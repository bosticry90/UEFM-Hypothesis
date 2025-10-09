
import numpy as np

def random_isometry(d_logical: int, d_physical: int, seed: int | None = None):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(d_physical, d_logical)) + 1j*rng.normal(size=(d_physical, d_logical))
    Q, _ = np.linalg.qr(X)
    return Q[:, :d_logical]

def knill_laflamme_checks(V: np.ndarray, error_ops: list[np.ndarray], atol=1e-8):
    d_physical, d_logical = V.shape
    IdL = np.eye(d_logical, dtype=complex)
    C = np.zeros((len(error_ops), len(error_ops)), dtype=complex)
    ok = True
    for a, Ea in enumerate(error_ops):
        for b, Eb in enumerate(error_ops):
            M = V.conj().T @ Ea.conj().T @ Eb @ V
            lam = np.trace(M) / d_logical
            C[a,b] = lam
            if np.linalg.norm(M - lam*IdL) > atol:
                ok = False
    return ok, C

def erasure_errors(n_physical: int, dims: list[int] | None = None, erase_set: set[int] | None = None):
    if erase_set is None: erase_set = set()
    if dims is None: dims = [2]*n_physical
    D = int(np.prod(dims))
    Id = np.eye(D, dtype=complex)
    E = [Id]
    mask = np.ones(D, dtype=complex)
    mask[::2] = 0.0
    E.append(np.diag(mask))
    return E
