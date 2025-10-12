import numpy as np
from toe import Substrate
from pathlib import Path

def main(n=32, chi_list=(2,3,4,6,8)):
    Svals = []
    for chi in chi_list:
        S = []
        ring = Substrate.ring(n, d=chi)
        for m in range(1, n//2 + 1):  # contiguous region sizes
            A = list(range(m))
            S_bits = ring.entropy_of(A).s_bits
            S.append((m, S_bits))
        Svals.append((chi, np.array(S)))
    # save TSVs for plotting
    outdir = Path("outputs"); outdir.mkdir(exist_ok=True, parents=True)
    for chi, arr in Svals:
        np.savetxt(outdir / f"area_law_n{n}_chi{chi}.tsv", arr, fmt="%.6f", header="size\tS_bits")
    print("Wrote:", [f"outputs/area_law_n{n}_chi{chi}.tsv" for chi,_ in Svals])

if __name__ == "__main__":
    main()
