# Theory of Everything (ToE) Research Framework

This repository contains the full implementation and validation suite for the **Unified Ether Field Model (UEFM)** — an emergent field theory built from a coherence-based scalar foundation and extended through tensor-train (TT) and quantum cellular automaton (QCA) analysis.

---

## 📘 Overview
- **Phase 1:** Foundational QCA + QEC coherence core  
- **Phase 2:** Noise–drift relations and stochastic stability  
- **Phase 3:** Tensor-train coherence and area law validation  
- **Phase 3.5:** Curvature, superfluid mapping, and stochastic extension

Each phase is fully reproducible using Python examples in `/examples` and verified through unit tests and predictive checks.

---

## ⚙️ Setup
```bash
git clone https://github.com/<your-user>/ToE.git
cd ToE
python -m venv .venv
.venv\Scripts\activate  # or source .venv/bin/activate on Linux
pip install -e .
pip install -r requirements.txt
pytest -q
