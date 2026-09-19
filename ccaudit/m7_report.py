"""
ccaudit.m7_report -- Module 7, the report.

Renders metrics.json into a single self-contained HTML file: no external CSS,
no JavaScript, no image files, every figure an inline SVG, so the report can
be moved and viewed as one file.

Figures:
    F1  FS vs AUC over detectors
    F2  citation correctness x faithfulness 2x2 (needs --localization in m6)
    F3  inpainting delta vs splice delta, with y = x   (needs --raw)
    F4  proxy validation bars
    F5  FS across causally supervised tuning variants
    F6  stated / attributed / overlay agreement with the causal region
    F7  blind-spot decomposition, stacked per detector

Also writes results_table.csv (one row per detector x tag) and, optionally,
each figure as a standalone .svg.
"""

from __future__ import annotations

import argparse
import csv
import html
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import common as C

CSS = """
:root{--fg:#16181d;--mut:#666e7a;--line:#dfe3e8;--bg:#fff;--accent:#1f6feb;
--good:#137333;--bad:#b3261e;--warn:#8a6d00;--band:#f6f8fa}
*{box-sizing:border-box}
body{font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
color:var(--fg);background:var(--bg);margin:0;padding:32px 28px 80px;max-width:1180px}
h1{font-size:24px;margin:0 0 4px}
h2{font-size:18px;margin:34px 0 10px;padding-bottom:6px;border-bottom:1px solid var(--line)}
h3{font-size:15px;margin:22px 0 8px;color:var(--mut)}
p{margin:8px 0}
.sub{color:var(--mut);font-size:13px;margin-bottom:18px}
table{border-collapse:collapse;width:100%;margin:10px 0 18px;font-size:13px}
th,td{text-align:right;padding:6px 9px;border-bottom:1px solid var(--line);
white-space:nowrap}
th{background:var(--band);font-weight:600;text-align:right;color:var(--mut)}
th:first-child,td:first-child{text-align:left}
tr:hover td{background:#fafbfc}
code{background:var(--band);padding:1px 5px;border-radius:3px;font-size:12px}
.good{color:var(--good);font-weight:600}
.bad{color:var(--bad);font-weight:600}
.warn{color:var(--warn);font-weight:600}
.note{background:var(--band);border-left:3px solid var(--accent);
padding:10px 14px;margin:12px 0;font-size:13px;color:#333}
.fig{margin:16px 0 26px}
.cap{color:var(--mut);font-size:12px;margin-top:4px}
.small{font-size:12px;color:var(--mut)}
"""

PALETTE = ["#1f6feb", "#d1440a", "#137333", "#8a3ffc", "#b3261e", "#0f766e",
           "#a16207", "#9333ea"]


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def esc(x: Any) -> str:
    return html.escape(str(x))


def fmt(x: Any, nd: int = 3, dash: str = "--") -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return dash
    if not np.isfinite(v):
        return dash
    return f"{v:.{nd}f}"


def ci(v: Any, lo: Any, hi: Any, nd: int = 3) -> str:
    if fmt(v, nd) == "--":
        return "--"
    if fmt(lo, nd) == "--":
        return fmt(v, nd)
    return f"{fmt(v, nd)} <span class='small'>[{fmt(lo, nd)}, {fmt(hi, nd)}]</span>"


def table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    h = "".join(f"<th>{esc(x)}</th>" for x in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>"
                   for r in rows)
    return f"<table><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table>"


def verdict_cell(res: Dict[str, Any]) -> str:
    if res.get("faithful"):
        return "<span class='good'>FAITHFUL</span>"
    if res.get("faithful_forward"):
        return "<span class='warn'>forward only</span>"
    return "<span class='bad'>UNFAITHFUL</span>"


# --------------------------------------------------------------------------
# inline SVG figures
# --------------------------------------------------------------------------

def _axes(w: int, h: int, pad: int, xlab: str, ylab: str, title: str,
          xr: Tuple[float, float], yr: Tuple[float, float]) -> List[str]:
    x0, y0 = pad, h - pad
    parts = [
        f"<rect width='{w}' height='{h}' fill='white'/>",
        f"<text x='{w/2}' y='18' text-anchor='middle' font-size='13' "
        f"font-weight='600' fill='#16181d'>{esc(title)}</text>",
        f"<line x1='{x0}' y1='{y0}' x2='{w-pad/2}' y2='{y0}' stroke='#c8ced6'/>",
        f"<line x1='{x0}' y1='{pad}' x2='{x0}' y2='{y0}' stroke='#c8ced6'/>",
        f"<text x='{w/2}' y='{h-6}' text-anchor='middle' font-size='11' "
        f"fill='#666e7a'>{esc(xlab)}</text>",
        f"<text x='12' y='{h/2}' text-anchor='middle' font-size='11' "
        f"fill='#666e7a' transform='rotate(-90 12 {h/2})'>{esc(ylab)}</text>",
    ]
    for i in range(5):
        xv = xr[0] + (xr[1] - xr[0]) * i / 4
        px = x0 + (w - pad - pad / 2) * i / 4
        parts.append(f"<text x='{px}' y='{y0+14}' text-anchor='middle' "
                     f"font-size='10' fill='#8a93a0'>{xv:.2f}</text>")
        yv = yr[0] + (yr[1] - yr[0]) * i / 4
        py = y0 - (h - 2 * pad) * i / 4
        parts.append(f"<text x='{x0-6}' y='{py+3}' text-anchor='end' "
                     f"font-size='10' fill='#8a93a0'>{yv:.2f}</text>")
        parts.append(f"<line x1='{x0}' y1='{py}' x2='{w-pad/2}' y2='{py}' "
                     f"stroke='#eef1f4'/>")
    return parts


def svg_scatter(points: Sequence[Tuple[float, float, str]], xlab: str,
                ylab: str, title: str, diagonal: bool = False,
                zero_lines: bool = True, w: int = 560, h: int = 330,
                pad: int = 56) -> str:
    pts = [(x, y, l) for x, y, l in points
           if np.isfinite(x) and np.isfinite(y)]
    if not pts:
        return f"<p class='small'>({esc(title)}: no data)</p>"
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    xr = (min(xs + [0.0]), max(xs + [0.001]))
    yr = (min(ys + [0.0]), max(ys + [0.001]))
    xr = (xr[0] - 0.06 * (xr[1] - xr[0] + 1e-9), xr[1] + 0.06 * (xr[1] - xr[0] + 1e-9))
    yr = (yr[0] - 0.12 * (yr[1] - yr[0] + 1e-9), yr[1] + 0.12 * (yr[1] - yr[0] + 1e-9))

    def sx(v):
        return pad + (w - pad - pad / 2) * (v - xr[0]) / (xr[1] - xr[0] + 1e-12)

    def sy(v):
        return (h - pad) - (h - 2 * pad) * (v - yr[0]) / (yr[1] - yr[0] + 1e-12)

    parts = _axes(w, h, pad, xlab, ylab, title, xr, yr)
    if zero_lines and yr[0] < 0 < yr[1]:
        parts.append(f"<line x1='{pad}' y1='{sy(0)}' x2='{w-pad/2}' y2='{sy(0)}' "
                     f"stroke='#b3261e' stroke-dasharray='4 3' stroke-width='1'/>")
    if diagonal:
        lo, hi = max(xr[0], yr[0]), min(xr[1], yr[1])
        parts.append(f"<line x1='{sx(lo)}' y1='{sy(lo)}' x2='{sx(hi)}' "
                     f"y2='{sy(hi)}' stroke='#8a93a0' stroke-dasharray='5 4'/>")
    for i, (x, y, lab) in enumerate(pts):
        col = PALETTE[i % len(PALETTE)] if lab else "#1f6feb"
        op = "0.85" if lab else "0.35"
        r = 6 if lab else 3
        parts.append(f"<circle cx='{sx(x):.1f}' cy='{sy(y):.1f}' r='{r}' "
                     f"fill='{col}' opacity='{op}'/>")
        if lab:
            parts.append(f"<text x='{sx(x)+9:.1f}' y='{sy(y)+4:.1f}' "
                         f"font-size='10.5' fill='#16181d'>{esc(lab)}</text>")
    return (f"<svg viewBox='0 0 {w} {h}' width='100%' style='max-width:{w}px' "
            f"xmlns='http://www.w3.org/2000/svg' role='img'>"
            + "".join(parts) + "</svg>")


def svg_bars(labels: Sequence[str], values: Sequence[float], title: str,
             ylab: str = "", w: int = 560, h: int = 300, pad: int = 56) -> str:
    vals = [v if np.isfinite(v) else 0.0 for v in values]
    if not vals:
        return f"<p class='small'>({esc(title)}: no data)</p>"
    vmax = max(vals + [0.0])
    vmin = min(vals + [0.0])
    span = (vmax - vmin) or 1.0
    parts = [f"<rect width='{w}' height='{h}' fill='white'/>",
             f"<text x='{w/2}' y='18' text-anchor='middle' font-size='13' "
             f"font-weight='600'>{esc(title)}</text>"]
    n = len(vals)
    bw = (w - pad - 20) / max(1, n)
    base = (h - pad) - (h - 2 * pad) * (0 - vmin) / span
    parts.append(f"<line x1='{pad-4}' y1='{base}' x2='{w-16}' y2='{base}' "
                 f"stroke='#c8ced6'/>")
    if ylab:
        parts.append(f"<text x='12' y='{h/2}' text-anchor='middle' "
                     f"font-size='11' fill='#666e7a' "
                     f"transform='rotate(-90 12 {h/2})'>{esc(ylab)}</text>")
    for i, (lab, v) in enumerate(zip(labels, vals)):
        y = (h - pad) - (h - 2 * pad) * (v - vmin) / span
        x = pad + i * bw + bw * 0.15
        top, hh = (min(y, base), abs(base - y))
        parts.append(f"<rect x='{x:.1f}' y='{top:.1f}' width='{bw*0.7:.1f}' "
                     f"height='{max(1,hh):.1f}' fill='{PALETTE[i%len(PALETTE)]}' "
                     f"opacity='0.85' rx='2'/>")
        parts.append(f"<text x='{x+bw*0.35:.1f}' y='{top-4:.1f}' "
                     f"text-anchor='middle' font-size='10'>{v:.3f}</text>")
        parts.append(f"<text x='{x+bw*0.35:.1f}' y='{h-pad+16:.1f}' "
                     f"text-anchor='middle' font-size='10' fill='#666e7a'>"
                     f"{esc(lab)}</text>")
    return (f"<svg viewBox='0 0 {w} {h}' width='100%' style='max-width:{w}px' "
            f"xmlns='http://www.w3.org/2000/svg'>" + "".join(parts) + "</svg>")


def svg_stacked(labels: Sequence[str], series: Dict[str, Sequence[float]],
                title: str, w: int = 560, h: int = 300, pad: int = 56) -> str:
    if not labels:
        return f"<p class='small'>({esc(title)}: no data)</p>"
    keys = list(series)
    parts = [f"<rect width='{w}' height='{h}' fill='white'/>",
             f"<text x='{w/2}' y='18' text-anchor='middle' font-size='13' "
             f"font-weight='600'>{esc(title)}</text>"]
    n = len(labels)
    bw = (w - pad - 120) / max(1, n)
    top_y, bot_y = pad, h - pad
    for i, lab in enumerate(labels):
        acc = 0.0
        x = pad + i * bw + bw * 0.15
        for j, k in enumerate(keys):
            v = series[k][i]
            v = v if np.isfinite(v) else 0.0
            y1 = bot_y - (bot_y - top_y) * (acc + v)
            y0 = bot_y - (bot_y - top_y) * acc
            parts.append(f"<rect x='{x:.1f}' y='{y1:.1f}' width='{bw*0.7:.1f}' "
                         f"height='{max(0,y0-y1):.1f}' "
                         f"fill='{PALETTE[j%len(PALETTE)]}' opacity='0.85'/>")
            acc += v
        parts.append(f"<text x='{x+bw*0.35:.1f}' y='{h-pad+16:.1f}' "
                     f"text-anchor='middle' font-size='10' fill='#666e7a'>"
                     f"{esc(lab)}</text>")
    for j, k in enumerate(keys):
        yy = pad + 8 + j * 16
        parts.append(f"<rect x='{w-112}' y='{yy-9}' width='10' height='10' "
                     f"fill='{PALETTE[j%len(PALETTE)]}' opacity='0.85'/>")
        parts.append(f"<text x='{w-98}' y='{yy}' font-size='10.5' "
                     f"fill='#16181d'>{esc(k)}</text>")
    return (f"<svg viewBox='0 0 {w} {h}' width='100%' style='max-width:{w}px' "
            f"xmlns='http://www.w3.org/2000/svg'>" + "".join(parts) + "</svg>")


def svg_venn3(counts: Dict[str, int], labels: Tuple[str, str, str],
              title: str, w: int = 480, h: int = 380) -> str:
    """
    Three-circle Venn diagram.  `counts` is keyed by membership strings over
    "SAO" ("SAO" = in all three, "S.." = stated only, "..." = in none), as
    emitted by m6.citation_agreement.
    """
    total = sum(counts.values())
    if not total:
        return f"<p class='small'>({esc(title)}: no data)</p>"
    cx, cy, r = w / 2.0, h / 2.0 + 8, 92.0
    cs = [(cx, cy - 46), (cx - 52, cy + 34), (cx + 52, cy + 34)]
    parts = [f"<rect width='{w}' height='{h}' fill='white'/>",
             f"<text x='{w/2}' y='20' text-anchor='middle' font-size='13' "
             f"font-weight='600'>{esc(title)}</text>"]
    for i, (x, y) in enumerate(cs):
        parts.append(f"<circle cx='{x}' cy='{y}' r='{r}' "
                     f"fill='{PALETTE[i]}' fill-opacity='0.16' "
                     f"stroke='{PALETTE[i]}' stroke-width='1.5'/>")
    # Label positions for the seven cells.  Each position lies inside exactly
    # the intersection it labels (checked by distance to the three centres in
    # the self-test).
    pos = {
        "S..": (cx, cy - 86),
        ".A.": (cx - 88, cy + 62),
        "..O": (cx + 88, cy + 62),
        "SA.": (cx - 44, cy - 6),
        "S.O": (cx + 44, cy - 6),
        ".AO": (cx, cy + 58),
        "SAO": (cx, cy + 10),
    }
    for key, (px, py) in pos.items():
        if key not in counts:
            continue
        v = counts.get(key, 0)
        parts.append(f"<text x='{px}' y='{py}' text-anchor='middle' "
                     f"font-size='15' font-weight='700' fill='#16181d'>{v}</text>")
        parts.append(f"<text x='{px}' y='{py+13}' text-anchor='middle' "
                     f"font-size='9' fill='#666e7a'>{100*v/total:.0f}%</text>")
    none = counts.get("...", 0)
    parts.append(f"<text x='{w-14}' y='{h-12}' text-anchor='end' font-size='11' "
                 f"fill='#b3261e'>none correct: {none} "
                 f"({100*none/total:.0f}%)</text>")
    for i, (lab, (x, y)) in enumerate(zip(labels, cs)):
        ly = y - r - 8 if i == 0 else y + r + 16
        parts.append(f"<text x='{x}' y='{ly}' text-anchor='middle' "
                     f"font-size='11' font-weight='600' "
                     f"fill='{PALETTE[i]}'>{esc(lab)}</text>")
    parts.append(f"<text x='14' y='{h-12}' font-size='10' fill='#666e7a'>"
                 f"n = {total} samples with all three elicitations</text>")
    return (f"<svg viewBox='0 0 {w} {h}' width='100%' style='max-width:{w}px' "
            f"xmlns='http://www.w3.org/2000/svg'>" + "".join(parts) + "</svg>")


def svg_2x2(counts: Dict[str, int], title: str, w: int = 420, h: int = 300) -> str:
    cells = [("cc1_faithful", "correct\n& causal", "#137333"),
             ("cc1_unfaithful", "correct\nNOT causal", "#b3261e"),
             ("cc0_faithful", "wrong\nbut causal", "#8a6d00"),
             ("cc0_unfaithful", "wrong\n& not causal", "#666e7a")]
    total = max(1, sum(counts.get(k, 0) for k, _l, _c in cells))
    parts = [f"<rect width='{w}' height='{h}' fill='white'/>",
             f"<text x='{w/2}' y='18' text-anchor='middle' font-size='13' "
             f"font-weight='600'>{esc(title)}</text>"]
    bw, bh = 150, 96
    ox, oy = 70, 46
    for i, (k, lab, col) in enumerate(cells):
        r, c = divmod(i, 2)
        x, y = ox + c * (bw + 8), oy + r * (bh + 8)
        v = counts.get(k, 0)
        parts.append(f"<rect x='{x}' y='{y}' width='{bw}' height='{bh}' "
                     f"fill='{col}' opacity='{0.12 + 0.5*v/total:.2f}' "
                     f"stroke='{col}' rx='4'/>")
        parts.append(f"<text x='{x+bw/2}' y='{y+38}' text-anchor='middle' "
                     f"font-size='20' font-weight='700' fill='{col}'>{v}</text>")
        for li, line in enumerate(lab.split("\n")):
            parts.append(f"<text x='{x+bw/2}' y='{y+58+li*13}' "
                         f"text-anchor='middle' font-size='10.5' "
                         f"fill='#16181d'>{esc(line)}</text>")
        parts.append(f"<text x='{x+bw/2}' y='{y+bh-6}' text-anchor='middle' "
                     f"font-size='10' fill='#666e7a'>{100*v/total:.0f}%</text>")
    parts.append(f"<text x='16' y='{oy+bh/2}' font-size='10.5' fill='#666e7a' "
                 f"transform='rotate(-90 16 {oy+bh/2})' "
                 f"text-anchor='middle'>cited = changed</text>")
    parts.append(f"<text x='16' y='{oy+bh*1.5+8}' font-size='10.5' "
                 f"fill='#666e7a' transform='rotate(-90 16 {oy+bh*1.5+8})' "
                 f"text-anchor='middle'>cited != changed</text>")
    return (f"<svg viewBox='0 0 {w} {h}' width='100%' style='max-width:{w}px' "
            f"xmlns='http://www.w3.org/2000/svg'>" + "".join(parts) + "</svg>")


# --------------------------------------------------------------------------
# sections
# --------------------------------------------------------------------------

def main_table(results: Sequence[Dict[str, Any]]) -> str:
    rows = []
    for r in sorted(results, key=lambda x: (str(x.get("detector")), str(x.get("tag")))):
        rows.append([
            f"<b>{esc(r.get('detector'))}</b>", esc(r.get("tag") or "--"),
            esc(r.get("n_samples")), esc(r.get("n_clips")),
            fmt(r.get("AUC")),
            ci(r.get("FS"), r.get("FS_lo"), r.get("FS_hi"), 4),
            fmt(r.get("rank_cited"), 2),
            ci(r.get("CR"), r.get("CR_lo"), r.get("CR_hi")),
            fmt(r.get("CR_prior")),
            fmt(r.get("CR_minus_prior")),
            fmt(r.get("seam_abs"), 4),
            fmt(r.get("floor_p_mean")),
            verdict_cell(r),
        ])
    return table(["detector", "tag", "n", "clips", "AUC", "FS [95% CI]",
                  "rank", "CR [95% CI]", "prior", "CR-prior", "|seam|",
                  "floor p", "verdict"], rows)


def controls_check(results: Sequence[Dict[str, Any]]) -> str:
    """Compare each synthetic control's observed verdict with the verdict it
    has by construction."""
    want = {"adaptive_oracle": "FAITHFUL", "confabulator": "UNFAITHFUL",
            "dummy": "UNFAITHFUL"}
    found, lines, ok = {}, [], True
    for r in results:
        d = str(r.get("detector", ""))
        for key in want:
            if d == key:
                found[key] = r
        if d.startswith("fixed_oracle"):
            found["fixed_oracle"] = r
    if not found:
        return ""
    for key, target in want.items():
        r = found.get(key)
        if not r:
            continue
        got = ("FAITHFUL" if r.get("faithful") else
               "forward only" if r.get("faithful_forward") else "UNFAITHFUL")
        good = (got == target)
        ok &= good
        lines.append(f"<tr><td>{esc(key)}</td><td>{esc(target)}</td>"
                     f"<td>{esc(got)}</td><td>{fmt(r.get('AUC'))}</td>"
                     f"<td>{fmt(r.get('FS'),4)}</td>"
                     f"<td class='{'good' if good else 'bad'}'>"
                     f"{'OK' if good else 'MISMATCH'}</td></tr>")
    r = found.get("fixed_oracle")
    if r:
        got = ("FAITHFUL" if r.get("faithful") else
               "forward only" if r.get("faithful_forward") else "UNFAITHFUL")
        good = (got == "forward only")
        ok &= good
        lines.append(f"<tr><td>{esc(r.get('detector'))}</td>"
                     f"<td>forward only</td><td>{esc(got)}</td>"
                     f"<td>{fmt(r.get('AUC'))}</td><td>{fmt(r.get('FS'),4)}</td>"
                     f"<td class='{'good' if good else 'bad'}'>"
                     f"{'OK' if good else 'MISMATCH'}</td></tr>")
    banner = ("<div class='note'><b>Instrument validation "
              f"{'PASSED' if ok else 'FAILED'}.</b> The synthetic controls have "
              "known faithfulness by construction. If this ordering does not "
              "hold on real frames, the masks or the data are at fault and "
              "no downstream number should be interpreted"
              + ("." if ok else " until this is resolved.")
              + " In particular, <code>confabulator</code> is built to have a "
              "high AUC and an FS near zero, so that detection quality and "
              "citation faithfulness are measured separately.</div>")
    return (banner + "<table><thead><tr><th>control</th><th>by construction</th>"
            "<th>observed</th><th>AUC</th><th>FS</th><th></th></tr></thead>"
            "<tbody>" + "".join(lines) + "</tbody></table>")


def per_region_tables(results: Sequence[Dict[str, Any]]) -> str:
    out = []
    for r in sorted(results, key=lambda x: str(x.get("detector"))):
        pr = r.get("per_region") or {}
        if not pr:
            continue
        rows = []
        for k in r.get("regions", sorted(pr)):
            e = pr.get(k)
            if not e:
                continue
            star = " <span class='good'>*</span>" if e.get("significant_holm") else ""
            rows.append([esc(k), esc(e.get("n")),
                         ci(e.get("delta"), e.get("delta_lo"), e.get("delta_hi"), 4) + star,
                         fmt(e.get("delta_raw"), 4), fmt(e.get("delta_seam"), 4),
                         fmt(e.get("floor_p")), fmt(e.get("cited_frac"), 3),
                         fmt(e.get("p_boot"), 4)])
        out.append(f"<h3>{esc(r.get('detector'))} [{esc(r.get('tag'))}]</h3>"
                   + table(["region", "n", "delta [95% CI]", "delta_raw",
                            "delta_seam", "floor p", "cited frac", "p"], rows))
    if out:
        out.insert(0, "<p class='small'>* survives Holm-Bonferroni across "
                      "regions at alpha = 0.05.</p>")
    return "".join(out)


def block_table(results, key: str, cols: Sequence[Tuple[str, str, int]],
                title: str, note: str = "") -> str:
    rows = []
    for r in sorted(results, key=lambda x: str(x.get("detector"))):
        b = r.get(key)
        if not b:
            continue
        cells = [f"<b>{esc(r.get('detector'))}</b>"]
        for _lab, field, nd in cols:
            v = b.get(field)
            cells.append(esc(v) if isinstance(v, (int, str)) and nd < 0
                         else fmt(v, max(0, nd)))
        rows.append(cells)
    if not rows:
        return ""
    head = ["detector"] + [c[0] for c in cols]
    return (f"<h2>{esc(title)}</h2>"
            + (f"<div class='note'>{note}</div>" if note else "")
            + table(head, rows))


def build(metrics: Dict[str, Any], coarse: Optional[Dict[str, Any]] = None,
          by_method: Optional[Dict[str, Any]] = None,
          raw_points: Optional[List[Tuple[float, float, str]]] = None,
          title: str = "Counterfactual citation audit report",
          figures: Optional[Dict[str, str]] = None) -> str:
    """Render the HTML document.  `figures`, if given, is filled with
    {name: svg} so the caller can export each figure as a standalone .svg."""
    figs = figures if figures is not None else {}

    def fig(name: str, svg: str) -> str:
        if svg.lstrip().startswith("<svg"):
            figs[name] = svg
        return svg

    results = metrics.get("results", [])
    settings = metrics.get("settings", {})
    prov = metrics.get("provenance", {})

    parts: List[str] = [
        f"<h1>{esc(title)}</h1>",
        f"<div class='sub'>{esc(prov.get('timestamp',''))} &middot; "
        f"code {esc(str(prov.get('code_hash',''))[:12])} &middot; "
        f"vocab {esc(settings.get('vocab','face8'))} &middot; "
        f"split {esc(settings.get('split','all'))} &middot; "
        f"tau {esc(settings.get('tau',0.5))} &middot; "
        f"{esc(settings.get('n_boot',0))} bootstrap replicates over clips</div>",
    ]

    parts.append("<h2>1. Primary table</h2>")
    parts.append(
        "<div class='note'><b>AUC</b> fake-vs-real separation on the "
        "originals. <b>FS</b> effect at the cited region minus the mean effect "
        "elsewhere, seam-corrected. <b>rank</b> position of the cited region "
        "by effect size (1 is best). <b>CR</b> reverse citation recall against "
        "<b>prior</b>, the best constant guess. <b>|seam|</b> identity-splice "
        "effect. <b>floor p</b> manipulation probability on identity-spliced "
        "<i>authentic</i> frames; a high value indicates that the splice "
        "operator itself introduces evidence. Faithful requires FS and "
        "CR&minus;prior both positive with CIs excluding zero.</div>")
    parts.append(main_table(results))

    cc = controls_check(results)
    if cc:
        parts.append("<h2>2. Instrument validation (synthetic controls)</h2>")
        parts.append(cc)

    pts = [(r.get("AUC"), r.get("FS"), str(r.get("detector")))
           for r in results if r.get("AUC") is not None]
    parts.append("<h2>3. Figures</h2>")
    parts.append("<div class='fig'>" + fig("F1_fs_vs_auc", svg_scatter(
        pts, "AUC (detection quality)", "FS (explanation faithfulness)",
        "F1  FS versus AUC across detectors"))
        + "<div class='cap'>F1. Each point is a detector. The horizontal "
          "position is detection quality; the vertical position is the "
          "faithfulness score. A point far to the right and near zero height "
          "detects forgeries well while its cited region does not drive the "
          "verdict.</div></div>")

    loc = [(r, r.get("localization")) for r in results if r.get("localization")]
    for r, l in loc:
        parts.append("<div class='fig'>" + fig(
            f"F2_crosstab_{C.safe_name(str(r.get('detector')))}", svg_2x2(
                l.get("crosstab", {}),
                f"F2  {r.get('detector')}: correctness x faithfulness"))
            + "<div class='cap'>F2. Rows: whether the cited region is where "
              "the pixels changed. Columns: whether the cited region drove "
              "the verdict. The top-right cell counts samples whose citation "
              "is correct but not causal; the two properties are tabulated "
              "separately.</div></div>")

    if raw_points:
        parts.append("<div class='fig'>" + fig("F3_inpaint_vs_splice", svg_scatter(
            raw_points, "delta from inpainting", "delta from paired splice",
            "F3  inpainting vs paired-frame counterfactual", diagonal=True))
            + "<div class='cap'>F3. Points off the dashed y = x line are "
              "disagreements; points in the off-sign quadrants are cases where "
              "the inpainting counterfactual points the opposite way.</div></div>")

    prox = next((r["proxies"] for r in results if r.get("proxies")), None)
    if prox:
        names = list(prox)
        parts.append("<div class='fig'>" + fig("F4_proxy_validation", svg_bars(
            names, [prox[n].get("corr_fs", float("nan")) for n in names],
            "F4  proxy validation: correlation with true FS", "Spearman rho"))
            + "<div class='cap'>F4. A usable proxy needs a high correlation "
              "and a low floor; read this beside the proxy table."
              "</div></div>")

    for r in results:
        ag = r.get("citation_agreement") or {}
        if ag.get("venn") and ag.get("venn_n"):
            parts.append("<div class='fig'>" + fig(
                f"F6_venn_{C.safe_name(str(r.get('detector')))}",
                svg_venn3(ag["venn"],
                          ("stated = causal", "attributed = causal",
                           "overlay = causal"),
                          f"F6  {r.get('detector')}: stated vs attributed vs overlay"))
                + "<div class='cap'>F6. Each set is the samples where that way "
                  "of eliciting a citation agreed with the region that moved "
                  "the verdict. Small overlaps indicate that the three "
                  "elicitations identify different regions and that few of "
                  "them coincide with the causal region.</div></div>")

    bs = [(str(r.get("detector")), r["blind_spot"]) for r in results
          if r.get("blind_spot")]
    if bs:
        parts.append("<div class='fig'>" + fig("F7_blind_spot", svg_stacked(
            [b[0] for b in bs],
            {"encoder blind": [b[1].get("encoder_blind_frac", 0) for b in bs],
             "language confabulation": [b[1].get("language_confabulation_frac", 0) for b in bs],
             "faithful": [b[1].get("faithful_frac", 0) for b in bs]},
            "F7  blind-spot decomposition"))
            + "<div class='cap'>F7. Encoder-blind means the visual tokens "
              "covering the region did not move under the counterfactual, so "
              "the evidence was not available to the language model."
              "</div></div>")

    # F5 collects the tuning variants of a detector: tuned runs are identified
    # by a detector name containing "cset" or "lora", and the untuned base
    # run by a tag starting with "cset", so that the base bar appears
    # alongside the tuned ones.
    cset = [r for r in results
            if "cset" in str(r.get("detector", "")).lower()
            or "lora" in str(r.get("detector", "")).lower()
            or str(r.get("tag", "")).lower().startswith("cset")]
    if cset:
        labels = []
        for r in cset:
            lab = str(r.get("detector"))[-22:]
            tag = str(r.get("tag") or "")
            if tag and tag != "main":
                lab = f"{lab} [{tag}]"
            labels.append(lab)
        parts.append("<div class='fig'>" + fig("F5_cset", svg_bars(
            labels,
            [r.get("FS", float("nan")) for r in cset],
            "F5  FS across causally supervised tuning variants", "FS")) + "</div>")

    parts.append(block_table(
        results, "inpaint",
        [("n", "n_pairs", -1), ("corr vs delta", "corr_inpaint_vs_delta", 3),
         ("spearman", "spearman_inpaint_vs_delta", 3),
         ("sign disagreement", "sign_disagreement", 3),
         ("p rise on REAL", "p_rise_on_real", 4),
         ("lo", "p_rise_on_real_lo", 4), ("hi", "p_rise_on_real_hi", 4),
         ("FS (inpaint)", "FS_inpaint", 4)],
        "4. Inpainting comparison",
        "A positive <b>p rise on REAL</b> whose CI excludes zero indicates "
        "that the inpainter introduces forgery evidence on authentic frames, "
        "in which case the inpainting counterfactual is not a valid "
        "substitute for the paired-frame one."))

    parts.append(block_table(
        results, "prompt_stability",
        [("agree w/ primary", "argmax_agreement_with_primary", 3),
         ("mean pairwise", "mean_pairwise_agreement", 3),
         ("sd p(fake)", "p_fake_sd_across_wordings", 4),
         ("n", "n_samples", -1)],
        "5. Prompt paraphrase stability",
        "How often the cited region is unchanged under re-wording of the "
        "prompt. The option letters are identical across paraphrases, so only "
        "the wording varies."))

    parts.append(block_table(
        results, "citation_agreement",
        [("stated=causal", "stated_vs_causal", 3),
         ("attributed=causal", "attr_vs_causal", 3),
         ("stated=attributed", "stated_vs_attr", 3),
         ("overlay=causal", "overlay_vs_causal", 3), ("n", "n", -1)],
        "6. Stated vs attributed vs causal citation",
        "Agreement between the region the model <i>states</i>, the region its "
        "<i>gradients</i> point at, and the region that <i>causes</i> the "
        "verdict."))

    parts.append(block_table(
        results, "blind_spot",
        [("encoder blind", "encoder_blind_frac", 3),
         ("language confab", "language_confabulation_frac", 3),
         ("faithful", "faithful_frac", 3),
         ("rho(shift, delta)", "spearman_shift_vs_delta", 3),
         ("shift outside k", "shift_outside_mean", 4), ("n", "n", -1)],
        "7. Blind-spot decomposition"))

    parts.append(block_table(
        results, "union",
        [("delta(k1 u k2)", "delta_union_observed", 4),
         ("delta(k1)+delta(k2)", "delta_sum_of_parts", 4),
         ("super-additivity", "superadditivity", 4),
         ("corr", "corr", 3), ("n", "n", -1)],
        "8. Compositional (two-region) interventions"))

    prox_rows = []
    for r in results:
        for name, b in (r.get("proxies") or {}).items():
            flag = ("<span class='bad'>contaminating</span>"
                    if b.get("contaminating") else "<span class='good'>clean</span>")
            prox_rows.append([f"<b>{esc(r.get('detector'))}</b>", esc(name),
                              esc(b.get("n")), fmt(b.get("corr_fs")),
                              fmt(b.get("kendall_delta")),
                              fmt(b.get("FS_proxy"), 4), fmt(b.get("FS_true"), 4),
                              ci(b.get("floor_p_rise"), b.get("floor_p_rise_lo"),
                                 b.get("floor_p_rise_hi"), 4), flag])
    if prox_rows:
        parts.append("<h2>9. Deployable proxies</h2>")
        parts.append("<div class='note'>These need only the suspect image, no "
                     "paired original. A usable proxy combines a high "
                     "correlation with true FS <i>and</i> a low floor.</div>")
        parts.append(table(["detector", "proxy", "n", "rho(FS)",
                            "Kendall tau(delta)", "FS proxy", "FS true",
                            "floor rise on REAL [CI]", ""], prox_rows))

    ent_rows = []
    for r in results:
        e = r.get("citation_entropy") or {}
        if e.get("n"):
            ent_rows.append([f"<b>{esc(r.get('detector'))}</b>",
                             fmt(e.get("mean")), fmt(e.get("sd")),
                             fmt(e.get("corr_with_FS")), esc(e.get("n"))])
    if ent_rows:
        parts.append("<h2>10. Citation entropy</h2>")
        parts.append(table(["detector", "mean normalised entropy", "sd",
                            "rho with FS", "n"], ent_rows))

    tau_rows = []
    for r in results:
        for t, b in sorted((r.get("CR_by_tau") or {}).items()):
            tau_rows.append([f"<b>{esc(r.get('detector'))}</b>", esc(t),
                             esc(b.get("n_reverse_flipped")), fmt(b.get("CR")),
                             fmt(b.get("CR_prior")), fmt(b.get("CR_minus_prior"))])
    if tau_rows:
        parts.append("<h2>11. Reverse-threshold sweep</h2>")
        parts.append("<div class='note'>CR depends on where the reverse "
                     "threshold is set; the sweep makes that dependence "
                     "visible rather than hidden in one number.</div>")
        parts.append(table(["detector", "tau", "n flipped", "CR", "prior",
                            "CR-prior"], tau_rows))

    if coarse and coarse.get("results"):
        parts.append("<h2>12. Coarse granularity</h2>")
        parts.append("<div class='note'>The same raw rows re-analysed on "
                     "{eyes, nose, mouth, rest}. Citation mass is summed over "
                     "members once; every measured effect is averaged over "
                     "them, never inherited from one representative.</div>")
        parts.append(main_table(coarse["results"]))

    if by_method and by_method.get("results"):
        parts.append("<h2>13. Per manipulation method</h2>")
        rows = []
        for r in sorted(by_method["results"],
                        key=lambda x: (str(x.get("detector")),
                                       str(x.get("method_slice")))):
            rows.append([f"<b>{esc(r.get('detector'))}</b>",
                         esc(r.get("method_slice")), esc(r.get("n_samples")),
                         fmt(r.get("AUC")),
                         ci(r.get("FS"), r.get("FS_lo"), r.get("FS_hi"), 4),
                         fmt(r.get("CR_minus_prior")), verdict_cell(r)])
        parts.append(table(["detector", "method", "n", "AUC", "FS [95% CI]",
                            "CR-prior", "verdict"], rows))

    parts.append("<h2>14. Confusion: cited region vs region with the largest effect</h2>")
    for r in results:
        conf = r.get("confusion") or {}
        if not conf:
            continue
        cols = sorted({c for v in conf.values() for c in v})
        rows = []
        for said in sorted(conf):
            rows.append([esc(said)] + [esc(conf[said].get(c, 0)) for c in cols])
        parts.append(f"<h3>{esc(r.get('detector'))}</h3>"
                     + table(["cited \\ largest effect"] + cols, rows))

    # ---- Holm across the detector zoo ----
    holm_rows = []
    for r in sorted(results, key=lambda x: str(x.get("detector"))):
        if r.get("FS_significant_holm_detectors") is None:
            continue
        lost = r.get("faithful") and not r.get("faithful_holm")
        holm_rows.append([
            f"<b>{esc(r.get('detector'))}</b>", esc(r.get("tag")),
            fmt(r.get("FS"), 4), fmt(r.get("FS_p"), 4),
            ("<span class='good'>yes</span>"
             if r.get("FS_significant_holm_detectors")
             else "<span class='bad'>no</span>"),
            verdict_cell(r),
            ("<span class='bad'>FAITHFUL only before correction</span>"
             if lost else ("<span class='good'>FAITHFUL</span>"
                           if r.get("faithful_holm") else "--")),
        ])
    if holm_rows:
        fam = next((r.get("holm_family_size") for r in results
                    if r.get("holm_family_size")), 0)
        parts.append("<h2>15. Multiplicity across the detector zoo</h2>")
        parts.append(
            f"<div class='note'>Each detector's faithfulness verdict is a "
            f"separate hypothesis test. Running {fam} of them at alpha = 0.05 "
            f"and counting every one that clears it would inflate the "
            f"family-wise error rate. The <b>corrected verdict</b> column is "
            f"the verdict after Holm-Bonferroni correction; a detector that is "
            f"faithful only before correction does not meet the corrected "
            f"criterion. The family contains one result per detector (the "
            f"<code>main</code> tag when present).</div>")
        parts.append(table(["detector", "tag", "FS", "FS p", "survives Holm",
                            "uncorrected", "corrected verdict"], holm_rows))

    # ---- additivity of seam and content ----
    add = metrics.get("additivity") or {}
    if add:
        parts.append("<h2>16. Additivity of seam and content effects</h2>")
        parts.append(
            "<div class='note'>The seam correction "
            "<code>delta = delta_raw - delta_seam</code> is unbiased only if "
            "the seam and the content act additively. Dilation varies the "
            "seam, so the two interaction terms should match and the "
            "<b>residual</b> should straddle zero. Caveat: at small radii the "
            "mask may be smaller than the manipulation, so dilation adds "
            "<i>content</i> as well as seam and a large residual then says "
            "nothing about additivity. Compare two radii that both cover the "
            "manipulation, and check <b>seam share</b>.</div>")
        rows = []
        for det, b in sorted(add.items()):
            rows.append([
                f"<b>{esc(det)}</b>", esc(b.get("radii")),
                esc(b.get("n_paired_cells")),
                fmt(b.get("interaction_delta_raw"), 4),
                fmt(b.get("interaction_delta_seam"), 4),
                ci(b.get("residual"), b.get("residual_lo"), b.get("residual_hi"), 4),
                ("<span class='good'>additive</span>" if b.get("additive")
                 else "<span class='warn'>not additive</span>"),
            ])
        parts.append(table(["detector", "radii", "paired cells",
                            "interaction in delta_raw",
                            "interaction in delta_seam", "residual [95% CI]",
                            ""], rows))
        for det, b in sorted(add.items()):
            rr = [["r = " + k, fmt(v.get("delta_raw"), 4),
                   fmt(v.get("delta_seam"), 4), fmt(v.get("delta"), 4),
                   fmt(v.get("seam_share"), 3), esc(v.get("n"))]
                  for k, v in sorted(b.get("per_radius", {}).items(),
                                     key=lambda kv: int(kv[0]))]
            parts.append(f"<h3>{esc(det)}</h3>"
                         + table(["dilation", "delta_raw", "delta_seam",
                                  "delta", "seam share", "n"], rr))

    parts.append("<h2>17. Provenance</h2>")
    pkgs = {k: v for k, v in (prov.get("packages") or {}).items() if v}
    parts.append(table(["field", "value"], [
        ["python", esc(prov.get("python"))],
        ["platform", esc(prov.get("platform"))],
        ["code hash", f"<code>{esc(prov.get('code_hash'))}</code>"],
        ["packages", esc(", ".join(f"{k}={v}" for k, v in sorted(pkgs.items())))],
        ["GPUs", esc(", ".join(g.get("name", "") for g in prov.get("gpus", [])) or "none")],
    ]))

    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{esc(title)}</title><style>{CSS}</style></head><body>"
            + "".join(parts) + "</body></html>")


def export_figures(figs: Dict[str, str], out_dir: str) -> List[str]:
    """Write each inline figure as a standalone .svg file."""
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for name, svg in sorted(figs.items()):
        path = os.path.join(out_dir, f"{C.safe_name(name)}.svg")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("<?xml version='1.0' encoding='UTF-8'?>\n" + svg)
        written.append(path)
    return written


def results_csv(results: Sequence[Dict[str, Any]], path: str) -> str:
    fields = ["detector", "tag", "granularity", "n_samples", "n_clips", "AUC",
              "FS", "FS_lo", "FS_hi", "FS_p", "rank_cited", "CR", "CR_lo",
              "CR_hi", "CR_prior", "CR_minus_prior", "n_reverse_flipped",
              "delta_cited", "delta_mean_all", "seam_abs", "floor_p_mean",
              "floor_frac_flagged", "faithful_forward", "faithful_reverse",
              "faithful"]
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(fields)
        for r in results:
            w.writerow([r.get(f, "") for f in fields])
    return path


def collect_raw_points(raw_dirs: Sequence[str], limit: int = 2500
                       ) -> List[Tuple[float, float, str]]:
    """(delta_inpaint, delta) pairs for figure F3."""
    from . import m6_metrics as M6

    pts: List[Tuple[float, float, str]] = []
    for (_det, _tag), paths in sorted(M6.group_raw_files(raw_dirs).items()):
        rows, _m = M6.load_rows(paths)
        for r in rows:
            if r.get("p_inpaint_fake") is None:
                continue
            pf = r.get("p_orig_fake")
            try:
                di = float(pf) - float(r["p_inpaint_fake"])
                d = float(r["p_identity"]) - float(r["p_real"])
            except (TypeError, ValueError):
                continue
            if np.isfinite(di) and np.isfinite(d):
                pts.append((di, d, ""))
            if len(pts) >= limit:
                return pts
    return pts


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ccaudit.m7_report")
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--coarse", default="")
    ap.add_argument("--by-method", default="")
    ap.add_argument("--raw", default="", help="run dirs, for figure F3")
    ap.add_argument("--out", required=True)
    ap.add_argument("--csv", default="")
    ap.add_argument("--title", default="Counterfactual citation audit report")
    ap.add_argument("--figures-dir", default="",
                    help="export each figure as a standalone .svg")
    a = ap.parse_args(argv)

    metrics = C.load_json(a.metrics)
    coarse = C.load_json(a.coarse) if a.coarse and os.path.exists(a.coarse) else None
    bym = (C.load_json(a.by_method)
           if a.by_method and os.path.exists(a.by_method) else None)
    pts = (collect_raw_points([d for d in a.raw.split(",") if d.strip()])
           if a.raw else None)

    figs: Dict[str, str] = {}
    html_doc = build(metrics, coarse, bym, pts, a.title, figures=figs)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write(html_doc)
    print(f"[m7] report -> {a.out}  ({len(html_doc)/1024:.0f} KB)")
    csv_path = a.csv or os.path.join(os.path.dirname(os.path.abspath(a.out)),
                                     "results_table.csv")
    results_csv(metrics.get("results", []), csv_path)
    print(f"[m7] csv    -> {csv_path}")
    fig_dir = a.figures_dir or os.path.join(
        os.path.dirname(os.path.abspath(a.out)), "figures")
    if figs:
        written = export_figures(figs, fig_dir)
        print(f"[m7] figures-> {fig_dir}  ({len(written)} svg)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
