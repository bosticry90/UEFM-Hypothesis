# examples/wedge_phase_diagram.py
import numpy as np
from pathlib import Path
from toe.geometry import wedge_reconstructable_1d

OUT = Path("outputs"); OUT.mkdir(exist_ok=True)

def main():
    N = 256
    bndry_sizes = list(range(1, N//2+1, 4))
    erased_list = list(range(0, 9))          # how many physical qubits on boundary are erased
    code_d_edges = list(range(1, 9))         # code distance measured in edge units

    rows = ["boundary_size erased code_distance_edges reconstructable"]
    for B in bndry_sizes:
        for e in erased_list:
            for d in code_d_edges:
                rec = wedge_reconstructable_1d(boundary_size=B, erased=e, code_distance_edges=d)
                rows.append(f"{B} {e} {d} {int(rec)}")

    out = OUT / "wedge_phase.tsv"
    out.write_text("\n".join(rows))
    print(f"Wrote {out}")
if __name__ == "__main__":
    main()
