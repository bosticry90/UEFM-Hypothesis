# examples/make_figures.py
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob

OUT = Path("outputs")
FIGS = OUT / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

def _load_tsv(path):
    return np.loadtxt(path, delimiter="\t", comments="#")

def fig_entropy_path_ring():
    # Path
    for tsv in sorted(glob.glob(str(OUT / "entropy_path_N*_chi*.tsv"))):
        dat = _load_tsv(tsv)  # N, chi, m, s_bits
        m = dat[:,2]; s = dat[:,3]; chi = int(dat[0,1]); N = int(dat[0,0])
        plt.figure()
        plt.plot(m, s, marker="o", linestyle="-")
        plt.xlabel("Region size |A|")
        plt.ylabel("S(A) [bits]")
        plt.title(f"Path N={N}, bond dim χ={chi}")
        plt.tight_layout()
        base = Path(tsv).stem
        plt.savefig(FIGS / f"{base}.png", dpi=200)
        plt.savefig(FIGS / f"{base}.svg")
        plt.close()

    # Ring
    for tsv in sorted(glob.glob(str(OUT / "entropy_ring_N*_chi*.tsv"))):
        dat = _load_tsv(tsv)  # N, chi, m, s_bits
        m = dat[:,2]; s = dat[:,3]; chi = int(dat[0,1]); N = int(dat[0,0])
        plt.figure()
        plt.plot(m, s, marker="o", linestyle="-")
        plt.xlabel("Region size |A|")
        plt.ylabel("S(A) [bits]")
        plt.title(f"Ring N={N}, bond dim χ={chi}")
        plt.tight_layout()
        base = Path(tsv).stem
        plt.savefig(FIGS / f"{base}.png", dpi=200)
        plt.savefig(FIGS / f"{base}.svg")
        plt.close()

def fig_entropy_er_random():
    for tsv in sorted(glob.glob(str(OUT / "entropy_er_random_n*_chi*.tsv"))):
        dat = _load_tsv(tsv)  # n, chi, p, sample, m, s_bits
        n = int(dat[0,0]); chi = int(dat[0,1])
        # group by p
        ps = np.unique(dat[:,2])
        plt.figure()
        for p in ps:
            mask = dat[:,2] == p
            # average over samples per m
            Ms = np.unique(dat[mask][:,4])
            means = []
            for m in Ms:
                mm = dat[mask][:,4] == m
                means.append(dat[mask][mm][:,5].mean())
            plt.plot(Ms, means, marker="o", linestyle="-", label=f"p={p:.2f}")
        plt.xlabel("Region size |A|")
        plt.ylabel("⟨S(A)⟩ [bits]")
        plt.title(f"ER random (n={n}), bond dim χ={chi}")
        plt.legend()
        plt.tight_layout()
        base = Path(tsv).stem
        plt.savefig(FIGS / f"{base}.png", dpi=200)
        plt.savefig(FIGS / f"{base}.svg")
        plt.close()

def fig_lr_radius():
    tsv = OUT / "lr_radius.tsv"
    if tsv.exists():
        dat = _load_tsv(tsv)  # N, steps, radius, v_eff
        steps = dat[:,1]; radius = dat[:,2]; v = dat[:,3]
        plt.figure()
        plt.plot(steps, radius, marker="o", linestyle="-")
        plt.xlabel("Steps")
        plt.ylabel("Lightcone radius (proxy)")
        plt.title("Operator-spread Lieb–Robinson radius")
        plt.tight_layout()
        plt.savefig(FIGS / "lr_radius.png", dpi=200)
        plt.savefig(FIGS / "lr_radius.svg")
        plt.close()

        plt.figure()
        plt.plot(steps, v, marker="o", linestyle="-")
        plt.xlabel("Steps")
        plt.ylabel("v_eff = radius/steps")
        plt.title("Effective LR velocity")
        plt.tight_layout()
        plt.savefig(FIGS / "lr_velocity.png", dpi=200)
        plt.savefig(FIGS / "lr_velocity.svg")
        plt.close()

def fig_kl_threshold_map():
    tsv = OUT / "kl_threshold_map.tsv"
    if tsv.exists():
        dat = _load_tsv(tsv)  # d_in, d_out, w, trials, success_rate
        d_ins = np.unique(dat[:,0])
        for d_in in d_ins:
            mask = dat[:,0] == d_in
            douts = np.unique(dat[mask][:,1])
            ws = np.unique(dat[mask][:,2])
            for w in ws:
                m2 = mask & (dat[:,2] == w)
                plt.figure()
                plt.plot(dat[m2][:,1], dat[m2][:,4], marker="o", linestyle="-")
                plt.ylim(-0.05, 1.05)
                plt.xlabel("Physical dimension d_out")
                plt.ylabel("KL success rate")
                plt.title(f"KL feasibility (d_in={int(d_in)}, erasure weight={int(w)})")
                plt.tight_layout()
                fname = f"kl_success_din{int(d_in)}_w{int(w)}"
                plt.savefig(FIGS / f"{fname}.png", dpi=200)
                plt.savefig(FIGS / f"{fname}.svg")
                plt.close()

def main():
    fig_entropy_path_ring()
    fig_entropy_er_random()
    fig_lr_radius()
    fig_kl_threshold_map()
    print("Figures saved to", FIGS)

if __name__ == "__main__":
    main()
