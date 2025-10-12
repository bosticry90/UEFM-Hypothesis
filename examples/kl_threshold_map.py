# examples/kl_threshold_map.py
from __future__ import annotations
import numpy as np
from pathlib import Path
from toe.qec import random_isometry, erasure_errors, knill_laflamme_checks

def main(d_logs=(2,3), d_phys_list=(4,5,6,8,12), weights=(1,2), trials=5, atol=1e-8):
    rows = []
    for d_in in d_logs:
        for d_out in d_phys_list:
            for w in weights:
                ok_count = 0
                for t in range(trials):
                    V = random_isometry(d_in=d_in, d_out=d_out, seed=1000 + 10*t + d_out + w)
                    n_physical = int(np.round(np.log(d_out)/np.log(2)))  # crude: treat as ~qubits if near power of 2
                    n_physical = max(n_physical, 2)
                    errs = erasure_errors(n_physical=n_physical, k=w, local_dim=2)
                    ok, C = knill_laflamme_checks(V, errs, atol=atol)
                    ok_count += 1 if ok else 0
                success_rate = ok_count / trials
                rows.append((d_in, d_out, w, trials, success_rate))
    out = Path("outputs/kl_threshold_map.tsv")
    np.savetxt(out, np.array(rows, float), fmt="%.6f", delimiter="\t",
               header="d_in\td_out\terasure_weight\ttrials\tsuccess_rate")
    print("Wrote", out)

if __name__ == "__main__":
    main()
