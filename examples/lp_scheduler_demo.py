# examples/lp_scheduler_demo.py
import numpy as np

def try_linprog():
    try:
        from scipy.optimize import linprog
    except Exception:
        print("SciPy not installed; skipping LP demo. (pip install scipy to enable)")
        return

    # Simple resource scheduling: allocate time across 3 tasks to maximize utility
    # max c^T x s.t. A x <= b, x >= 0
    c = np.array([4.0, 3.0, 2.0])            # utility
    A = np.array([[1.0, 1.0, 1.0],           # total time <= 8h
                  [2.0, 1.0, 0.5]])          # GPU quota <= 10
    b = np.array([8.0, 10.0])

    # linprog minimizes; use -c
    res = linprog(-c, A_ub=A, b_ub=b, bounds=[(0, None)]*3, method="highs")
    if res.success:
        x = res.x
        util = c @ x
        print("LP OK. Allocation:", x.round(3), "Total utility:", round(util, 3))
    else:
        print("LP failed:", res.message)

if __name__ == "__main__":
    try_linprog()
