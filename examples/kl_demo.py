# examples/kl_demo.py
import numpy as np
from toe import knill_laflamme_checks

def basis_vec(dim, idx):
    v = np.zeros(dim, dtype=complex)
    v[idx] = 1.0
    return v

def kron(*ops):
    out = np.array([[1.0+0j]])
    for op in ops:
        out = np.kron(out, op)
    return out

def repetition_3qubit_isometry():
    """
    [[3,1,3]] repetition code (protects against single X):
      |0_L> = |000>, |1_L> = |111>
    Returns V of shape (8, 2) with V^† V = I_2.
    """
    d_out, d_in = 8, 2
    V = np.zeros((d_out, d_in), dtype=complex)
    # |000> is index 0, |111> is index 7 in computational ordering
    V[:, 0] = basis_vec(8, 0)  # |0_L>
    V[:, 1] = basis_vec(8, 7)  # |1_L>
    return V

def pauli_x_errors_1weight(n=3):
    """Return [I, X on qubit 0, X on qubit 1, X on qubit 2] as 8x8 matrices."""
    X = np.array([[0,1],[1,0]], dtype=complex)
    I = np.eye(2, dtype=complex)
    Es = []
    # Identity
    Es.append(kron(I, I, I))
    # Single-qubit X errors
    Es.append(kron(X, I, I))
    Es.append(kron(I, X, I))
    Es.append(kron(I, I, X))
    return Es

def main():
    V = repetition_3qubit_isometry()
    errors = pauli_x_errors_1weight()

    ok, C = knill_laflamme_checks(V, errors, atol=1e-10)
    max_off = np.max(np.abs(C - np.diag(np.diag(C))))
    max_diag_spread = np.max(np.abs(np.diag(C) - np.mean(np.diag(C))))

    print(f"KL holds for single-X errors? {ok}")
    print(f"max off-diagonal |C_ab|: {max_off:.3e}")
    print(f"diag proportionality spread: {max_diag_spread:.3e}")

if __name__ == "__main__":
    main()
