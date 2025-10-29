# tests/test_phase11_fss.py
import numpy as np
from examples.phase11_kz_fss import FSSParams, sweep_fss, summarize

def test_defects_increase_with_system_size():
    rows = sweep_fss(FSSParams(sizes=((24,16),(32,24),(48,32)), taus=(32.0,)))
    summ = summarize(rows)
    ok_all = all(summ["increase_with_size_by_tau"][str(32.0)] for _ in [0])
    assert ok_all

def test_defects_decrease_with_slower_quench_each_size():
    rows = sweep_fss(FSSParams(sizes=((32,24),(48,32)), taus=(16.0,32.0,64.0)))
    summ = summarize(rows)
    for shape in ("(32, 24)", "(48, 32)"):
        assert summ["decrease_with_tau_by_size"][shape]

def test_loglog_slope_vs_tau_is_negative():
    rows = sweep_fss(FSSParams(sizes=((32,24),), taus=(16.0, 32.0, 64.0)))
    summ = summarize(rows)
    slopes = [v["loglog_slope_tau"] for v in summ["by_size"].values()]
    assert all(s < 0.0 for s in slopes)
