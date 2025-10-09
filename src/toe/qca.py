
import numpy as np

def local_unitary(dim: int, seed: int | None = None):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(dim, dim)) + 1j*rng.normal(size=(dim, dim))
    Q, R = np.linalg.qr(X)
    Q *= np.exp(-1j*np.angle(np.linalg.det(Q)) / dim)
    return Q

class SplitStepQCA1D:
    def __init__(self, n_sites: int, d: int = 2, seed: int | None = None):
        self.n = n_sites
        self.d = d
        self.seed = seed
        self.U_even = local_unitary(d*d, seed=seed)
        self.U_odd  = local_unitary(d*d, seed=None if seed is None else seed+1)

    def step(self, psi):
        psi = self._apply_layer(psi, parity=0)
        psi = self._apply_layer(psi, parity=1)
        return psi

    def _apply_layer(self, psi, parity: int):
        nblocks = (self.n - 1 - parity) // 2
        for b in range(nblocks+1):
            i = 2*b + parity
            if i+1 >= self.n: break
            psi = self._apply_two_site_gate(psi, i, i+1, self.U_even if parity==0 else self.U_odd)
        return psi

    def _apply_two_site_gate(self, psi, i, j, U):
        d = self.d; n = self.n
        psi = psi.reshape([d]*n)
        axes = [i, j] + [k for k in range(n) if k not in (i,j)]
        inv_axes = np.argsort(axes)
        psi2 = np.transpose(psi, axes).reshape(d*d, -1)
        psi2 = (U @ psi2).reshape([d, d] + [d]*(n-2))
        psi = np.transpose(psi2, inv_axes).reshape(d**n)
        return psi

def norm(psi):
    return float(np.vdot(psi, psi).real)

def energy_conservation_proxy(psi0, qca: SplitStepQCA1D, steps: int = 50):
    norms = []
    psi = psi0.copy()
    for _ in range(steps):
        psi = qca.step(psi)
        norms.append(norm(psi))
    return np.array(norms)
