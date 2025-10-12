import numpy as np
from toe import knill_laflamme_checks
from pathlib import Path

def kron(*ops):
    out = np.array([[1.0+0j]])
    for op in ops: out = np.kron(out, op)
    return out

def rep3_V():
    V = np.zeros((8,2), dtype=complex)
    V[0,0] = 1.0  # |000>
    V[7,1] = 1.0  # |111>
    return V

def pauli_errors(weight=1):
    X = np.array([[0,1],[1,0]], dtype=complex)
    Z = np.array([[1,0],[0,-1]], dtype=complex)
    I = np.eye(2, dtype=complex)
    Es = [kron(I,I,I)]
    if weight >= 1:
        Es += [kron(X,I,I), kron(I,X,I), kron(I,I,X)]
    if weight >= 2:
        Es += [kron(X,X,I), kron(X,I,X), kron(I,X,X)]
    if weight >= 3:
        Es += [kron(X,X,X)]
    # (You can add Z/Y similarly)
    return Es

def main():
    V = rep3_V()
    rows = []
    for w in (0,1,2,3):
        E = pauli_errors(weight=w)
        ok, C = knill_laflamme_checks(V, E, atol=1e-10)
        off = np.max(np.abs(C - np.diag(np.diag(C))))
        rows.append((w, int(ok), float(off)))
    arr = np.array(rows, dtype=float)
    Path("outputs").mkdir(exist_ok=True, parents=True)
    np.savetxt("outputs/kl_rep3_vs_weight.tsv", arr, fmt="%.6e", header="weight ok max_offdiag")
    print("Wrote outputs/kl_rep3_vs_weight.tsv")

if __name__ == "__main__":
    main()
