"""Generate the per-target pipeline-runtime LaTeX table (tab:pipeline-cost).

Reads the runtime logs saved under each experiment directory and rebuilds
every cell from raw measurements — no hand-entered numbers:

  runtime.csv                 training / retrain / trace phases, appended by
                              scripts/train.py, rare_train_references.py,
                              score_lt_iqr.py and score_liu_mlp.py; RARE
                              scoring phases by scripts/measure_runtime.py
  logit_gen_runtime.csv       per-model logit-generation time by num_augs
  score_inference_runtime.csv cold per-target attack scoring (t_total)
  loss_forward_pass_runtime.csv  one A=1 forward pass over the train set
                              (these three: scripts/measure_runtime.py)

`--timings-dir runtime_measurements` builds the table from a fresh run of
scripts/measure_runtime.py instead of the timings shipped with the paper.

Cost model (per-target, A-to-Z, no amortization; A=2 logits are canonical
for LiRA, A=1 for RMIA / Attack-R / LOSS):

  Train            M x (baseline no-trace retrain).  Std of an M-model sum
                   is sqrt(M) x per-model std.
  Train artifacts  trajectory-recording overhead:
                     LT-IQR : target loss-trace overhead
                     LT-MLP : target + 1 ref loss-trace overhead (2x)
                     RARE   : target (loss,margin)-trace overhead
                              + reference trace overhead
                              (ref_retrain wall-clock minus baseline mean)
                   Overheads are paired per target id where both phases
                   logged the same id, else differenced against the mean.
  Inference        (M+1) x per-model logit generation at the attack's A,
                   plus the logged cold scoring time; trajectory methods
                   use their logged score time only.

Baseline source: `target_retrain_baseline` rows; experiments without them
(CIFAR-100) fall back to the `train` shadow-retrain rows.

Usage:
  python paper_tabs_figs/tab_pipeline_cost.py
  python paper_tabs_figs/tab_pipeline_cost.py --output figures/tab.tex
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_tabs_figs._tex import ADJUSTBOX_OPEN, dense_marks, mark_math  # noqa: E402

DEFAULT_BENCHMARKS = {
    "CIFAR10": ROOT / "experiments/cifar10/wrn28-2/2025-12-07_0140",
    "CIFAR100": ROOT / "experiments/cifar100/wrn28-2/2025-12-25_0121",
    "CINIC10": ROOT / "experiments/cinic10/wrn28-2/2026-05-11_0215",
}
DEFAULT_OUTPUT = ROOT / "paper_tabs_figs/tables/tab_pipeline_cost.tex"

_ID_RE = re.compile(r"(?:target|shadow)_(\d+)$")
_REF0_RE = re.compile(r"target_(\d+)_ref_0$")


def _load_runtime_rows(path: Path):
    """Parse runtime.csv tolerating the variable-width rows different
    scripts append (e.g. liu_mlp_score carries an extra R field)."""
    rows = []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if not row or row[0] == "timestamp" or len(row) < 4:
                continue
            try:
                sec = float(row[3])
            except ValueError:
                continue
            rows.append({"ts": row[0], "phase": row[1], "item": row[2],
                         "sec": sec, "extra": row[4:]})
    return rows


def _dedup_last(rows):
    """Keep the last logged measurement per item (reruns supersede).

    Used for scoring phases, where later rows reflect the current
    (optimized) implementation."""
    out = {}
    for r in rows:
        out[r["item"]] = r
    return list(out.values())


def _dedup_min(rows):
    """Keep the fastest complete measurement per item.

    Used for training/trace phases: re-runs of the same retrain landed on
    days with concurrent GPU load (e.g. the 2026-05-13 `train` batches run
    ~10-40% slower than the dedicated sweeps), so the minimum is the least
    contended estimate of the true cost. Rows under half the phase median
    are aborted partial runs (e.g. a 42 s ref retrain immediately re-run to
    completion at 145 s) and are dropped first."""
    if not rows:
        return []
    cutoff = 0.5 * float(np.median([r["sec"] for r in rows]))
    out = {}
    for r in rows:
        if r["sec"] < cutoff:
            continue
        prev = out.get(r["item"])
        if prev is None or r["sec"] < prev["sec"]:
            out[r["item"]] = r
    return list(out.values())


def _by_id(rows, id_re=_ID_RE):
    out = {}
    for r in rows:
        m = id_re.search(r["item"])
        if m:
            out[int(m.group(1))] = r["sec"]
    return out


class Stat:
    def __init__(self, values):
        v = np.asarray(list(values), dtype=float)
        if v.size == 0:
            raise ValueError("empty stat")
        self.mean = float(v.mean())
        self.std = float(v.std(ddof=1)) if v.size > 1 else 0.0
        self.n = int(v.size)
        self.values = v

    @classmethod
    def scaled(cls, base: "Stat", m: int) -> "Stat":
        """Sum of m i.i.d. draws: mean m*mu, std sqrt(m)*sigma."""
        s = cls.__new__(cls)
        s.mean = m * base.mean
        s.std = float(np.sqrt(m) * base.std)
        s.n = base.n
        s.values = None
        return s

    @classmethod
    def combine(cls, *stats: "Stat") -> "Stat":
        """Sum of independent components (quadrature std)."""
        s = cls.__new__(cls)
        s.mean = sum(x.mean for x in stats)
        s.std = float(np.sqrt(sum(x.std ** 2 for x in stats)))
        s.n = min(x.n for x in stats)
        s.values = None
        return s


def _overhead(trace_rows, base_rows, base_stat: Stat) -> Stat:
    """Trace-recording overhead: paired per target id when the baseline
    logged the same ids, else unpaired difference against the mean."""
    tr = _by_id(_dedup_min(trace_rows))
    ba = _by_id(_dedup_min(base_rows))
    common = sorted(set(tr) & set(ba))
    if len(common) >= 3:
        return Stat([tr[i] - ba[i] for i in common])
    t = Stat(tr.values())
    s = Stat.__new__(Stat)
    s.mean = t.mean - base_stat.mean
    s.std = float(np.sqrt(t.std ** 2 + base_stat.std ** 2))
    s.n = t.n
    s.values = None
    return s


# scripts/train.py logs target retrains as target_train_*; the paper's logs
# were written under the earlier target_retrain_* names. Both are accepted.
_PHASE_ALIAS = {
    "target_train_baseline": "target_retrain_baseline",
    "target_train_loss_trace": "target_retrain_loss_trace",
    "target_train_full_trace": "target_retrain_rare_trace",
}


# Inference-time scoring phases written by scripts/measure_runtime.py.
_SCORING_PHASES = {"anomaly_ltiqr", "rare_ltbtmr", "anomaly_rare"}


def collect(exp: Path, timings_dir: Path | None = None):
    """Timing statistics for one benchmark.

    Training-time phases come from `<exp>/runtime.csv`. The stand-alone
    inference timings (the three CSVs below and the RARE scoring phases) come
    from `timings_dir` when given -- a folder written by
    scripts/measure_runtime.py -- and otherwise from `exp` itself.
    """
    src = Path(timings_dir) if timings_dir is not None else exp
    rows = _load_runtime_rows(exp / "runtime.csv")
    if timings_dir is not None:
        rows = [r for r in rows if r["phase"] not in _SCORING_PHASES]
        rows += [r for r in _load_runtime_rows(src / "runtime.csv") if r["phase"] in _SCORING_PHASES]
    by_phase = {}
    for r in rows:
        by_phase.setdefault(_PHASE_ALIAS.get(r["phase"], r["phase"]), []).append(r)

    base_rows = by_phase.get("target_retrain_baseline") or by_phase["train"]
    base = Stat([r["sec"] for r in _dedup_min(base_rows)])

    loss_trace_rows = by_phase.get("target_retrain_loss_trace") \
        or by_phase["target_retrain_rare_trace"]
    rare_trace_rows = by_phase["target_retrain_rare_trace"]
    loss_oh = _overhead(loss_trace_rows, base_rows, base)
    rare_oh = _overhead(rare_trace_rows, base_rows, base)

    # RARE M=1 reference: full retrain wall-clock includes its trace.
    ref0 = Stat([r["sec"] for r in _dedup_min(
        [r for r in by_phase["ref_retrain"] if _REF0_RE.search(r["item"])])])
    ref0_oh = Stat.__new__(Stat)
    ref0_oh.mean = ref0.mean - base.mean
    ref0_oh.std = float(np.sqrt(ref0.std ** 2 + base.std ** 2))
    ref0_oh.n = ref0.n
    ref0_oh.values = None

    def phase_stat(name, predicate=None):
        sel = by_phase[name]
        if predicate:
            sel = [r for r in sel if predicate(r)]
        return Stat([r["sec"] for r in _dedup_last(sel)])

    lt_iqr_score = phase_stat("lt_iqr_score")
    # The 2026-07-24 liu_mlp_score batch is excluded: 21 rows tightly
    # clustered at 82-87 s against 5.7-6.8 s for identical work (same arch,
    # dataset, loading and n_data_points, R=1) on every benchmark. The
    # spread is too tight for GPU contention, so those runs measured a
    # different code path and are not comparable to the rest of the log.
    liu_score = phase_stat("liu_mlp_score",
                           lambda r: r["extra"] and r["extra"][0] == "1"
                           and not r["ts"].startswith("2026-07-24"))
    # RARE score = LT-IQR + standardized sum, plus LT-BTM-R when it was timed
    # separately (scripts/measure_runtime.py logs it as rare_ltbtmr).
    rare_parts = [phase_stat("anomaly_ltiqr"), phase_stat("anomaly_rare")]
    if "rare_ltbtmr" in by_phase:
        rare_parts.append(phase_stat("rare_ltbtmr"))
    rare_score = Stat.combine(*rare_parts)

    gen = {}
    with open(src / "logit_gen_runtime.csv", newline="") as f:
        for row in csv.DictReader(f):
            gen.setdefault(int(row["num_augs"]), []).append(float(row["seconds"]))
    gen = {a: Stat(v) for a, v in gen.items()}

    score = {}
    with open(src / "score_inference_runtime.csv", newline="") as f:
        for row in csv.DictReader(f):
            score.setdefault((row["method"], int(row["M"])), []).append(
                float(row["t_total"]))
    score = {k: Stat(v) for k, v in score.items()}

    loss_fwd = []
    with open(src / "loss_forward_pass_runtime.csv", newline="") as f:
        for row in csv.DictReader(f):
            loss_fwd.append(float(row["seconds"]))
    loss_fwd = Stat(loss_fwd)

    return dict(base=base, loss_oh=loss_oh, rare_oh=rare_oh, ref0_oh=ref0_oh,
                lt_iqr_score=lt_iqr_score, liu_score=liu_score,
                rare_score=rare_score, gen=gen, score=score, loss_fwd=loss_fwd)


def build_rows(c):
    """Return {method: (train, artifacts, inference, total)} of Stat|None."""
    base, gen, score = c["base"], c["gen"], c["score"]

    def shadow_attack(m, aug, method):
        train = Stat.scaled(base, m)
        inference = Stat.combine(Stat.scaled(gen[aug], m + 1), score[(method, m)])
        return train, None, inference, Stat.combine(train, inference)

    rows = {}
    rows["LiRA (M=256, gold)"] = shadow_attack(256, 2, "LiRA")
    rows["LiRA (M=8)"] = shadow_attack(8, 2, "LiRA")
    rows["RMIA (M=8)"] = shadow_attack(8, 1, "RMIA")
    rows["Attack-R (M=16)"] = shadow_attack(16, 1, "Attack-R")

    lt_art = Stat.combine(c["loss_oh"], c["loss_oh"])  # target + 1 ref trace
    rows["LT-MLP (M=1)"] = (base, lt_art, c["liu_score"],
                            Stat.combine(base, lt_art, c["liu_score"]))
    rows["LT-IQR (M=0)"] = (None, c["loss_oh"], c["lt_iqr_score"],
                            Stat.combine(c["loss_oh"], c["lt_iqr_score"]))
    loss_inf = Stat.combine(c["loss_fwd"], score[("LOSS", 0)])
    rows["LOSS (M=0)"] = (None, None, loss_inf, loss_inf)
    rare_art = Stat.combine(c["rare_oh"], c["ref0_oh"])
    rows["RARE (M=1)"] = (base, rare_art, c["rare_score"],
                          Stat.combine(base, rare_art, c["rare_score"]))
    return rows


def _fmt_num(v: float) -> str:
    if v >= 1000:
        return f"{v:,.0f}".replace(",", "{,}")
    if v >= 100:
        return f"{v:.0f}"
    if v >= 10:
        return f"{v:.1f}"
    if v >= 1:
        return f"{v:.2f}"
    return f"{v:.3f}"


def _fmt_std(v: float) -> str:
    """Two significant figures, keeping a trailing zero (6.0, 0.63, 71)."""
    if v >= 100:
        return f"{v:,.0f}".replace(",", "{,}")
    if v >= 10:
        return f"{v:.0f}"
    return f"{v:#.2g}"


def _fmt_cell(s) -> str:
    if s is None:
        return "---"
    return f"${_fmt_num(s.mean)}{{\\scriptstyle\\pm{_fmt_std(s.std)}}}$"


def _fmt_speedup(v: float, mark=None) -> str:
    """Speedup rounded to the nearest integer, with an optional best/second mark."""
    body = f"{v:,.0f}".replace(",", "{,}") + "\\times"
    return f"${mark_math(body, mark)}$"


METHOD_ORDER = [
    "LiRA (M=256, gold)", "LiRA (M=8)", "RMIA (M=8)", "Attack-R (M=16)",
    "LT-MLP (M=1)", "LT-IQR (M=0)", "LOSS (M=0)", "RARE (M=1)",
]
# Row labels as typeset in the paper.
TEX_LABEL = {
    "LiRA (M=256, gold)": "LiRA ($M{=}256$, gold)", "LiRA (M=8)": "LiRA ($M{=}8$)",
    "RMIA (M=8)": "RMIA ($M{=}8$)", "Attack-R (M=16)": "Attack-R ($M{=}16$)",
    "LT-MLP (M=1)": "LT-MLP ($M{=}1$)", "LT-IQR (M=0)": "LT-IQR ($M{=}0$)",
    "LOSS (M=0)": "LOSS ($M{=}0$)", "RARE (M=1)": "\\rare{} ($M{=}1$)",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--timings-dir", default=None,
                    help="Folder name under each experiment dir holding re-measured inference "
                         "timings from scripts/measure_runtime.py (e.g. runtime_measurements). "
                         "Default: the timings shipped in the experiment dirs (the paper's).")
    args = ap.parse_args()

    collected = {name: collect(exp, exp / args.timings_dir if args.timings_dir else None)
                 for name, exp in DEFAULT_BENCHMARKS.items()}
    per_ds = {name: build_rows(c) for name, c in collected.items()}
    gold = {name: rows["LiRA (M=256, gold)"][3].mean
            for name, rows in per_ds.items()}
    # Fastest / second-fastest per dataset, on the displayed integer speedups.
    speed = {n: {m: gold[n] / per_ds[n][m][3].mean for m in METHOD_ORDER}
             for n in DEFAULT_BENCHMARKS}
    marks = {n: dense_marks({m: round(v) for m, v in speed[n].items()}, 0)
             for n in DEFAULT_BENCHMARKS}
    target_train = np.mean([c["base"].mean for c in collected.values()])

    lines = [
        "% Auto-generated by paper_tabs_figs/tab_pipeline_cost.py -- do not edit by hand.",
        r"""\begin{table}[t!]
\centering
\caption{Per-target pipeline runtime (in s) on WRN-28-2, mean$\pm$std over 10
targets. \emph{Train} counts only models trained beyond the target
itself, so the target's own training (also """ + f"$\\approx {target_train:.0f}$" + r"""\,s under the
same configuration) is excluded from every row. Speedup is computed
from the unrounded mean total runtimes relative to LiRA-256 on the same
dataset and rounded to the nearest integer; 
bold denotes the fastest and underline the second fastest
method.}
\label{tab:pipeline-cost}
\setlength{\tabcolsep}{2.5pt}
\renewcommand{\arraystretch}{0.98}""",
        ADJUSTBOX_OPEN,
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "\\textbf{Method} & \\textbf{Data} & \\textbf{Train} & \\textbf{Train artifacts}"
        " & \\textbf{Inference} & \\textbf{Total} & \\textbf{Speedup} \\\\",
        "\\midrule",
    ]
    for mi, method in enumerate(METHOD_ORDER):
        for di, name in enumerate(DEFAULT_BENCHMARKS):
            train, art, inf, total = per_ds[name][method]
            head = f"\\multirow{{3}}{{*}}{{{TEX_LABEL[method]}}} " if di == 0 else ""
            lines.append(
                f"{head}& {name} & {_fmt_cell(train)} & {_fmt_cell(art)}"
                f" & {_fmt_cell(inf)} & {_fmt_cell(total)}"
                f" & {_fmt_speedup(speed[name][method], marks[name][method])} \\\\"
            )
        if mi < len(METHOD_ORDER) - 1:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]

    tex = "\n".join(lines)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(tex + "\n")
    print(tex)

    # Numbers referenced in the running text of sec:results-runtime.
    print("\n% ---- prose numbers ----")
    rare = {n: per_ds[n]["RARE (M=1)"] for n in DEFAULT_BENCHMARKS}
    totals = " / ".join(f"{rare[n][3].mean:.0f}" for n in DEFAULT_BENCHMARKS)
    speeds = [gold[n] / rare[n][3].mean for n in DEFAULT_BENCHMARKS]
    print(f"% RARE totals (s): {totals}")
    print(f"% RARE speedup range: {min(speeds):.0f}x - {max(speeds):.0f}x")
    for n in DEFAULT_BENCHMARKS:
        c = collected[n]
        print(f"% {n}: ref train (no-trace rate) {c['base'].mean:.0f}s; "
              f"artifacts = target trace {c['rare_oh'].mean:.1f}s "
              f"+ ref trace {c['ref0_oh'].mean:.1f}s; "
              f"RARE score {c['rare_score'].mean:.3f}s")


if __name__ == "__main__":
    main()
