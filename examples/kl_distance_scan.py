# examples/kl_distance_scan.py
import numpy as np
from pathlib import Path
from toe.qec import random_isometry, erasure_errors, knill_laflamme_checks

OUT = Path("outputs"); OUT.mkdir(exist_ok=True)

def main():
    rng = np.random.default_rng(123)
    d_log = 2
    # physical dims: 2^n where n in 3..7
    ns = [3,4,5,6,7]
    ks = [0,1,2,3]     # max erasures tolerated
    reps = 5

    lines = ["n_physical k_erasure rep ok max_offdiag max_diag_std"]
    for n in ns:
        d_phys = 2**n
        for k in ks:
            if k > n: continue
            for r in range(reps):
                V = random_isometry(d_in=d_log, d_out=d_phys, rng=rng)
                E = erasure_errors(n=n, k=k, local_dim=2)
                ok, stats = knill_laflamme_checks(V, [e for (e,_) in E], atol=1e-8)
                lines.append(f"{n} {k} {r} {int(ok)} {stats['max_offdiag']:.3e} {stats['max_diag_std']:.3e}")

    out = OUT / "kl_distance_scan.tsv"
    out.write_text("\n".join(lines))
    print(f"Wrote {out}")
if __name__ == "__main__":
    main()
