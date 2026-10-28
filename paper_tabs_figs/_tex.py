"""Shared LaTeX formatting for the paper table generators.

Every table in the paper follows the same conventions:
  * wide content is wrapped in ``\\adjustbox{max width=\\linewidth}{...}``;
  * bold marks the best and underline the second-best value in a column,
    decided on the *displayed* (rounded) values, so ties after rounding share a
    mark ("dense" ranking: the second mark goes to the next distinct value,
    but never to the worst value of the column).
"""

from __future__ import annotations

from typing import Dict, Hashable, Iterable, Optional

ADJUSTBOX_OPEN = "\\adjustbox{max width=\\linewidth}{%"
ADJUSTBOX_CLOSE = "\\end{tabular}}"


def dense_marks(values: Dict[Hashable, float], decimals: int, *, higher_is_better: bool = True,
                levels: int = 2) -> Dict[Hashable, Optional[str]]:
    """Return {key: "best" | "second" | None} from values rounded to `decimals`.

    The worst displayed value never earns the second mark: when a column has
    only two distinct values (e.g. every method ties for best except one), the
    remaining value stays unmarked.
    """
    shown = {k: float(f"{v:.{decimals}f}") for k, v in values.items()}
    ranked = sorted(set(shown.values()), reverse=higher_is_better)
    names = ["best", "second"][:levels]
    tier = {v: (names[i] if i < len(names) else None) for i, v in enumerate(ranked)}
    if len(ranked) == 2:
        tier[ranked[-1]] = None
    return {k: tier[v] for k, v in shown.items()}


def column_marks(table: Dict[Hashable, Iterable[float]], decimals: int, *,
                 higher_is_better: bool = True, levels: int = 2):
    """Per-column dense marks for {row: [v_col0, v_col1, ...]} -> {row: [mark, ...]}."""
    rows = list(table)
    cols = list(zip(*[list(table[r]) for r in rows]))
    out = {r: [] for r in rows}
    for col in cols:
        m = dense_marks(dict(zip(rows, col)), decimals,
                        higher_is_better=higher_is_better, levels=levels)
        for r in rows:
            out[r].append(m[r])
    return out


def mark_text(body: str, mark: Optional[str]) -> str:
    """Text-mode mark (\\textbf / \\underline)."""
    if mark == "best":
        return f"\\textbf{{{body}}}"
    if mark == "second":
        return f"\\underline{{{body}}}"
    return body


def mark_math(body: str, mark: Optional[str]) -> str:
    """Math-mode mark (\\mathbf / \\underline)."""
    if mark == "best":
        return f"\\mathbf{{{body}}}"
    if mark == "second":
        return f"\\underline{{{body}}}"
    return body


def pm_scriptsize(mean: float, std: float, decimals: int = 2, mark: Optional[str] = None) -> str:
    """``0.95{\\scriptsize$\\pm$0.02}`` with the mark on the mean only."""
    return f"{mark_text(f'{mean:.{decimals}f}', mark)}{{\\scriptsize$\\pm${std:.{decimals}f}}}"


def pm_scriptstyle(mean: float, std: float, decimals: int = 3, mark: Optional[str] = None) -> str:
    """``$0.952{\\scriptstyle\\pm0.024}$`` with the mark on the mean only."""
    return f"${mark_math(f'{mean:.{decimals}f}', mark)}{{\\scriptstyle\\pm{std:.{decimals}f}}}$"


def figure_env(pdf_name: str, caption: str, label: str, *, star: bool = True,
               width: str = "\\textwidth") -> str:
    """Figure wrapper as in the manuscript, which keeps PDFs under figures/."""
    env = "figure*" if star else "figure"
    return (f"\\begin{{{env}}}[t]\n\\centering\n"
            f"\\includegraphics[width={width}]{{figures/{pdf_name}}}\n"
            f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{{env}}}\n")
