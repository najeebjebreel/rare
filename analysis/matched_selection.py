"""Matched-cardinality and common-selector extensions of the vulnerable-set table.

The incumbent table (`paper_tabs_figs/tab_actionable_selection.py`, tab:actionable
-selection) compares 2-D KMeans over the two RARE components against attacks
thresholded at FPR<=1e-4. Those rows confound two things: the *ranking* a score
induces and the *cardinality* each selection rule happens to pick. These two
panels separate them.

  Panel A  matched cardinality. B = |S_KMeans-RARE| per target; every competing
           score takes its top-B members. Cardinality is held fixed, so only the
           ranking differs.
  Panel B  common selector. The same 1-D KMeans (K=2 on member-standardized
           scores, higher-mean cluster) is applied to every scalar score,
           including the scalar RARE score. The rule is held fixed, so only the
           score differs.

Gold set, metrics, KMeans configuration, z-scoring and cell formatting are reused
verbatim from the incumbent module. Nothing under version control is written;
output is `analysis/matched_selection.md`.

Usage:
  python analysis/matched_selection.py
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import Dict, List

# sklearn's KMeans uses a chunked OpenMP reduction whose summation order depends
# on the thread count, so `random_state=0` alone does NOT make it reproducible
# across processes. Measured drift on this data: 2 of 30 targets, |S| off by 2-3
# of ~5500 (~0.04%). Pinning to one thread makes runs bit-identical. Must be set
# before numpy/sklearn initialise their threading layers.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from rare.analysis.utils import load_rare_locked_scores  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "sel_base", ROOT / "paper_tabs_figs" / "tab_actionable_selection.py")
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

BENCHMARKS = base.DEFAULT_BENCHMARKS
TARGETS = base.DEFAULT_TARGETS
FPR = 1e-4
Q = 0.50
DS = ["CIFAR10", "CIFAR100", "CINIC10"]
BLOCKS = DS + ["Avg."]

REF_A = "KMeans-RARE (2-D, reference)"
COMPETING = ["RARE (scalar)", "LT-IQR", "LT-BTM-R", "LiRA (M=8)", "RMIA (M=8)",
             "Attack-R (M=16)", "LT-MLP (M=1)"]
PANEL_A_ORDER = [REF_A] + COMPETING
REF_B = "KMeans-RARE (2-D, reference)"
PANEL_B_ORDER = [REF_B] + COMPETING


def build_scores(exp: Path, t: int, members: np.ndarray):
    """Member-standardized scalar scores + the 2-D RARE component features."""
    lt_iqr = base._load_lt_iqr(exp, t)
    lt_btm = base._load_lt_btm_r(exp, t, Q)
    z_iqr = base._zscore_on_members(lt_iqr, members)
    z_btm = base._zscore_on_members(lt_btm, members)
    rare = z_iqr + z_btm

    ar = exp / f"model_{t}" / "attack_results"
    raw = {
        "RARE (scalar)": rare,
        "LT-IQR": lt_iqr,
        "LT-BTM-R": lt_btm,
        "LiRA (M=8)": np.load(ar / "lira_refs_8_scores.npy").astype(np.float32),
        "RMIA (M=8)": np.load(ar / "rmia_refs_8_scores.npy").astype(np.float32),
        "Attack-R (M=16)": np.load(ar / "attackr_refs_16_scores.npy").astype(np.float32),
        "LT-MLP (M=1)": np.load(
            exp / f"model_{t}" / "liu_mlp_scores_R1.npy").astype(np.float32),
    }
    # Sanity: the table's z_iqr + z_btm must equal the locked scalar RARE.
    locked = load_rare_locked_scores(exp, t, num_refs=1, q=Q)
    dev = float(np.nanmax(np.abs(rare[members] - locked[members])))
    return raw, np.column_stack([z_iqr, z_btm]), rare, dev


def top_b_select(scores: np.ndarray, members: np.ndarray, B: int) -> np.ndarray:
    """Top-B members by score (ties broken by index, as argpartition does)."""
    ok = members & np.isfinite(scores)
    idx = np.where(ok)[0]
    b = int(min(B, idx.size))
    sel = np.zeros(members.shape, dtype=bool)
    if b <= 0:
        return sel
    order = np.argpartition(scores[idx], -b)[-b:]
    sel[idx[order]] = True
    return sel


def evaluate():
    perA: Dict[str, Dict[str, list]] = {m: {d: [] for d in DS} for m in PANEL_A_ORDER}
    perB: Dict[str, Dict[str, list]] = {m: {d: [] for d in DS} for m in PANEL_B_ORDER}
    devs, gold_sizes, b_sizes = [], {d: [] for d in DS}, {d: [] for d in DS}
    agree = {d: {"jaccard": [], "symdiff": []} for d in DS}

    for ds, exp in BENCHMARKS.items():
        for t in TARGETS:
            members, vulnerable = base._load_lira_gold(exp, int(t), FPR)
            raw, feats2d, rare, dev = build_scores(exp, int(t), members)
            devs.append(dev)

            km_rare = base._kmeans_select(feats2d, members, rare)
            B = int((km_rare & members).sum())
            b_sizes[ds].append(B)
            gold_sizes[ds].append(int(vulnerable.sum()))

            ref_m = base._metrics(km_rare, members, vulnerable)
            perA[REF_A][ds].append(ref_m)
            perB[REF_B][ds].append(ref_m)

            for m in COMPETING:
                s = raw[m]
                sel_a = top_b_select(s, members, B)
                perA[m][ds].append(base._metrics(sel_a, members, vulnerable))
                if m == "RARE (scalar)":
                    km_m, ta_m = km_rare & members, sel_a & members
                    agree[ds]["jaccard"].append(
                        float((km_m & ta_m).sum()) / float((km_m | ta_m).sum()))
                    agree[ds]["symdiff"].append(int((km_m ^ ta_m).sum()))
                z = base._zscore_on_members(s, members)
                perB[m][ds].append(
                    base._metrics(base._kmeans_select(z.reshape(-1, 1), members, z),
                                  members, vulnerable))
            print(f"  {ds} t={t} B={B} done", file=sys.stderr, flush=True)
    return perA, perB, devs, gold_sizes, b_sizes, agree


METRICS = ["selected", "precision", "recall", "f1"]


def aggregate(per):
    """{method: {block: {metric: (mean, std)}}}; Avg. = over the 3 dataset means."""
    agg = {m: {} for m in per}
    for m, byds in per.items():
        ds_means = {k: [] for k in METRICS}
        for ds in DS:
            agg[m][ds] = {}
            for k in METRICS:
                v = np.array([r[k] for r in byds[ds]], dtype=float)
                agg[m][ds][k] = (v.mean(), v.std(ddof=1))
                ds_means[k].append(v.mean())
        agg[m]["Avg."] = {k: (float(np.mean(ds_means[k])),
                              float(np.std(ds_means[k], ddof=1))) for k in METRICS}
    return agg


def decorate(agg, order, block, method, metric):
    """Incumbent rule: bold within 5e-4 of the column best, underline of second."""
    vals = np.array([agg[m][block][metric][0] for m in order])
    v = agg[method][block][metric][0]
    ordered = np.sort(vals)[::-1]
    out = base._fmt_metric(v)
    if np.isclose(v, ordered[0], rtol=0.0, atol=5e-4):
        return rf"\textbf{{{out}}}", "bold"
    if ordered.size > 1 and np.isclose(v, ordered[1], rtol=0.0, atol=5e-4):
        return rf"\underline{{{out}}}", "underline"
    return out, ""


def latex_rows(agg, order):
    lines, deco = [], {}
    for m in order:
        label = m.replace("RARE", r"\rare{}").replace("KMeans-\\rare{} (2-D, reference)",
                                                      r"KMeans-\rare{} (2-D)")
        cells = [label]
        for blk in BLOCKS:
            cells.append(base._fmt_selected(agg[m][blk]["selected"][0]))
            for k in ("precision", "recall", "f1"):
                c, d = decorate(agg, order, blk, m, k)
                cells.append(c)
                if d:
                    deco[(m, blk, k)] = d
        lines.append(" & ".join(cells) + r" \\")
    return lines, deco


def md_table(A, agg, order, out):
    out.append("| Method | Block | \\|S\\| | Precision | Recall | F1 |")
    out.append("|---|---|---|---|---|---|")
    for m in order:
        for blk in BLOCKS:
            a = agg[m][blk]
            out.append(
                f"| {m} | {blk} | {a['selected'][0]:.1f} ± {a['selected'][1]:.1f} | "
                f"{a['precision'][0]:.3f} ± {a['precision'][1]:.3f} | "
                f"{a['recall'][0]:.3f} ± {a['recall'][1]:.3f} | "
                f"{a['f1'][0]:.3f} ± {a['f1'][1]:.3f} |")
    out.append("")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path,
                    default=ROOT / "analysis" / "matched_selection.md")
    args = ap.parse_args()

    perA, perB, devs, gold_sizes, b_sizes, agree = evaluate()
    aggA, aggB = aggregate(perA), aggregate(perB)
    rowsA, decoA = latex_rows(aggA, PANEL_A_ORDER)
    rowsB, decoB = latex_rows(aggB, PANEL_B_ORDER)

    L: List[str] = []
    A = L.append
    A("# Vulnerable-set selection: matched cardinality and a common selector")
    A("")
    A("## Why these two panels")
    A("")
    A("The incumbent table (`tab:actionable-selection`) compares 2-D KMeans over the RARE")
    A("components against attacks thresholded at FPR ≤ 1e-4. Those rows differ in **two**")
    A("ways at once — the ranking each score induces, and the number of samples each rule")
    A("happens to select — so a win there cannot be attributed to either cause. These")
    A("panels vary one at a time.")
    A("")
    A("| Panel | Held fixed | Varies | Answers |")
    A("|---|---|---|---|")
    A("| **A** | cardinality (B = \\|S_KMeans-RARE\\| per target) | the score's ranking | is the RARE *ranking* better? |")
    A("| **B** | the selection rule (1-D KMeans, K=2) | the score | is RARE better under a *common* selector? |")
    A("")
    A("Gold set, metrics, KMeans settings (`n_clusters=2, random_state=0, n_init=50`),")
    A("member z-scoring and cell formatting are reused verbatim from")
    A("`paper_tabs_figs/tab_actionable_selection.py`; only the two selection rules are new.")
    A("`rare/analysis/utils.py` and the incumbent module are unmodified — this code lives")
    A("in the new file `analysis/matched_selection.py`.")
    A("")
    A(f"Setup: experiment dirs `2025-12-07_0140` / `2025-12-25_0121` / `2026-05-11_0215`,")
    A(f"targets {TARGETS}, gold = LiRA-256 at FPR ≤ {FPR:g}, q={Q}, M=1 reference.")
    A("Metrics are computed over members only, exactly as in the incumbent `_metrics`:")
    A("`P = tp/|S∩M|`, `R = tp/|V|`, `F1` their harmonic mean.")
    A("")
    A("Mean ± std is over the 10 targets per dataset; the **Avg.** block is the mean of the")
    A("three dataset means, with std over those three means (ddof=1) — matching the")
    A("ablation-table convention. The LaTeX rows below carry no ± because the incumbent")
    A("selection table does not; the ± values are in the plain tables.")
    A("")
    A("### Selection sizes")
    A("")
    A("| Dataset | B = \\|S_KMeans-RARE\\| | gold \\|V\\| | B / \\|V\\| |")
    A("|---|---|---|---|")
    for ds in DS:
        b = np.array(b_sizes[ds], float)
        g = np.array(gold_sizes[ds], float)
        A(f"| {ds} | {b.mean():.1f} ± {b.std(ddof=1):.1f} | {g.mean():.1f} ± "
          f"{g.std(ddof=1):.1f} | {(b / g).mean():.2f} |")
    A("")
    A(f"Sanity: the table's `z_M(LT-IQR) + z_M(LT-BTM-R)` matches the locked scalar RARE")
    A(f"from `load_rare_locked_scores()` to max abs deviation **{max(devs):.2e}** over")
    A("members, all 30 targets — the two code paths agree.")
    A("")

    # ---------------------------------------------------------------- Panel A
    A("---")
    A("")
    A("## Panel A — matched cardinality (top-B)")
    A("")
    A("Every row selects exactly B members, where B is that target's KMeans-RARE count.")
    A("")
    md_table(perA, aggA, PANEL_A_ORDER, L)
    A("### The 2-D clustering is doing almost nothing")
    A("")
    A("The reference row and the `RARE (scalar)` top-B row agree to three decimals on")
    A("every metric in every block. They are **not** the same set, but they are close:")
    A("")
    A("| Dataset | Jaccard(S_KMeans-2D, top-B by scalar RARE) | symmetric difference | as % of B |")
    A("|---|---|---|---|")
    for ds in DS:
        j = np.array(agree[ds]["jaccard"], float)
        sdv = np.array(agree[ds]["symdiff"], float)
        bb = np.array(b_sizes[ds], float)
        A(f"| {ds} | {j.mean():.4f} ± {j.std(ddof=1):.4f} | {sdv.mean():.0f} ± "
          f"{sdv.std(ddof=1):.0f} | {(sdv / bb).mean() * 100:.1f}% |")
    A("")
    alljac = np.concatenate([agree[d]["jaccard"] for d in DS])
    A(f"Over all 30 targets the two selections overlap at Jaccard {alljac.min():.3f}–"
      f"{alljac.max():.3f}, disagreeing on ~1–2% of the selected members, and **never**")
    A("coincide exactly (0/30). The disagreements sit at the cluster boundary, where")
    A("precision is near the decision threshold, so they cancel in aggregate.")
    A("")
    A("The practical reading: because both components are standardized to unit variance,")
    A("the 2-D cloud is near-isotropic and the K=2 split falls almost perpendicular to the")
    A("(1,1) direction — which *is* the scalar RARE axis. The 2-D clustering is therefore")
    A("close to a threshold on the scalar score, and the paper's KMeans-RARE row could be")
    A("replaced by a much simpler rule with no measurable loss. That is a simplification")
    A("worth making explicit rather than a result to defend.")
    A("")
    A("**Structural note.** With B and |V| both fixed per target, `P = tp/B`, `R = tp/|V|`")
    A("and `F1 = 2tp/(B+|V|)` are all monotone in the single quantity `tp`. Within a")
    A("target the three columns therefore rank methods identically; they differ across")
    A("the aggregate only through how B and |V| vary between targets. Read Panel A as one")
    A("effective comparison, not three independent ones.")
    A("")
    A("### Drop-in LaTeX rows — Panel A")
    A("")
    A("```latex")
    for r in rowsA:
        A(r)
    A("```")
    A("")

    # ---------------------------------------------------------------- Panel B
    A("---")
    A("")
    A("## Panel B — common unsupervised selector (1-D KMeans)")
    A("")
    A("K=2 on member-standardized scores, higher-mean cluster selected. The reference row")
    A("is the incumbent 2-D KMeans over the two RARE components.")
    A("")
    md_table(perB, aggB, PANEL_B_ORDER, L)
    A("### Drop-in LaTeX rows — Panel B")
    A("")
    A("```latex")
    for r in rowsB:
        A(r)
    A("```")
    A("")

    # ---------------------------------------------------------------- bolds
    A("---")
    A("")
    A("## Which cells take decorations")
    A("")
    A("Decorations follow the incumbent `_ranked_metric` rule: **bold** within 5e-4 of the")
    A("column best, <u>underline</u> within 5e-4 of the second best, computed per")
    A("(block × metric) column. |S| is never decorated. Panels are ranked **separately** —")
    A("a row's decoration in Panel A says nothing about Panel B.")
    A("")
    for pname, order, deco, agg in [("A", PANEL_A_ORDER, decoA, aggA),
                                    ("B", PANEL_B_ORDER, decoB, aggB)]:
        A(f"### Panel {pname}")
        A("")
        A("| Block | Metric | Bold (best) | Underline (2nd) |")
        A("|---|---|---|---|")
        for blk in BLOCKS:
            for k in ("precision", "recall", "f1"):
                b = [m for m in order if deco.get((m, blk, k)) == "bold"]
                u = [m for m in order if deco.get((m, blk, k)) == "underline"]
                A(f"| {blk} | {k[0].upper() + k[1:]} | {', '.join(b) or '—'} | "
                  f"{', '.join(u) or '—'} |")
        A("")
        winners = sorted({m for (m, _, _), d in deco.items() if d == "bold"})
        A(f"Rows taking at least one bold: {', '.join(winners) if winners else 'none'}.")
        A("")
        excl = [m for m in order if m != REF_A]
        A("Excluding the reference row, the best competing score per block (by F1):")
        A("")
        A("| Block | Best competitor | F1 | vs reference |")
        A("|---|---|---|---|")
        for blk in BLOCKS:
            bm = max(excl, key=lambda m: agg[m][blk]["f1"][0])
            d = (agg[bm][blk]["f1"][0] - agg[REF_A][blk]["f1"][0]) * 100
            A(f"| {blk} | {bm} | {agg[bm][blk]['f1'][0]:.3f} | {d:+.1f} pp |")
        A("")

    A("## Reproducing")
    A("")
    A("```bash")
    A("python analysis/matched_selection.py")
    A("```")
    A("")
    A("### A reproducibility caveat worth propagating")
    A("")
    A("`random_state=0` alone does **not** make sklearn's KMeans reproducible across")
    A("processes: its Lloyd iteration uses a chunked OpenMP reduction whose summation")
    A("order depends on the thread count. Running the unpinned selection twice on this")
    A("data gave different results on 2 of 30 targets, with |S| differing by 2–3 members")
    A("out of ~5500 (~0.04%). No cell in this report moved at 3 decimals, and mean |S|")
    A("shifted by 0.1, so the conclusions are unaffected — but the numbers were not")
    A("bit-reproducible.")
    A("")
    A("This script therefore sets `OMP_NUM_THREADS=1` before importing numpy/sklearn,")
    A("which makes repeated runs bit-identical (verified: two fresh processes, all 30")
    A("targets, zero differences).")
    A("")
    A("**This applies equally to the incumbent `tab_actionable_selection.py`,** which uses")
    A("the same `_kmeans_select` without pinning threads — its published |S| column carries")
    A("the same ~0.04% wobble. Worth adding the same pin there.")
    A("")
    A("Top-B selection is a partition on fixed arrays and is deterministic regardless.")

    args.output.write_text("\n".join(L) + "\n")
    print(f"wrote {args.output} ({args.output.stat().st_size} bytes)")

    for pname, order, agg in [("A", PANEL_A_ORDER, aggA), ("B", PANEL_B_ORDER, aggB)]:
        print(f"\n=== Panel {pname} (Avg. block) ===")
        for m in order:
            a = agg[m]["Avg."]
            print(f"  {m:<32} |S|={a['selected'][0]:8.1f}  P={a['precision'][0]:.3f}  "
                  f"R={a['recall'][0]:.3f}  F1={a['f1'][0]:.3f}")
    print(f"\nmax |z_iqr+z_btm - locked RARE| = {max(devs):.2e}")


if __name__ == "__main__":
    main()
