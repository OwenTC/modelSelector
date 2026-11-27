"""Turn a completed synthetic search benchmark into CSV, figures, and a PDF summary."""

from __future__ import annotations

import argparse
import csv
from io import BytesIO
import json
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .benchmark import BASELINE, DESIGNS


DESIGN_LABEL = {"identifiable": "Identifiable", "pairwise-sums": "Pairwise sums"}
COLORS = {"off": "#64748b", "reverse": "#087e8b", "saliency": "#4e55a5", "one-step": "#b55425"}
MARKERS = {"off": "o", "reverse": "v", "saliency": "^", "one-step": "D"}
INK = "#142b38"
MUTED = "#49636e"


def load_results(directory: Path) -> tuple[dict, list[dict]]:
    plan = json.loads((directory / "plan.json").read_text())
    expected = {method["name"] for method in plan["methods"]}
    rows, missing = [], []
    for case in plan["cases"]:
        path = directory / "cases" / f"{case['index']:06d}.json"
        if not path.exists():
            missing.append(case["index"])
            continue
        result = json.loads(path.read_text())
        methods = {row["method"]: row for row in result["results"]}
        if result["case"] != case or methods.keys() != expected or len(result["results"]) != len(expected):
            raise ValueError(f"Result does not match the plan: {path}")
        baseline = methods[BASELINE]["fits"]
        oracle = result["oracle"]
        for row in result["results"]:
            rows.append({**case, "rank": result["rank"], "true_support": str(result["true_support"]),
                         **row, "baseline_fits": baseline, "delta_fits": row["fits"] - baseline,
                         "saving_pct": 100 * (baseline - row["fits"]) / baseline,
                         "oracle_best_aicc": oracle["best_aicc"] if oracle else None,
                         "oracle_tied_models": oracle["tied_models"] if oracle else None})
    if missing:
        raise ValueError(f"Missing {len(missing)} cases. First missing indices: {missing[:12]}")
    return plan, rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def condition(row: dict) -> tuple:
    return row["design"], row["features"], row["snr"]


def condition_label(group: tuple) -> str:
    design, features, snr = group
    return f"{DESIGN_LABEL[design]} / {features}p / SNR {snr:g}"


def median_saving(rows: list[dict]) -> float:
    return median(row["saving_pct"] for row in rows)


def format_pct(value: float) -> str:
    rounded = round(value, 1)
    return f"{rounded:+.1f}%" if rounded else "0.0%"


def make_figures(path: Path, methods: list[dict], rows: list[dict], by_method: dict) -> BytesIO:
    groups = sorted({condition(row) for row in rows})
    variants = [method for method in methods if method["name"] != BASELINE]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.labelcolor": INK, "text.color": INK,
                         "pdf.fonttype": 42})
    with PdfPages(path, metadata={"Title": "Synthetic search heuristic comparisons"}) as pdf:
        for features, snr in sorted({group[1:] for group in groups}):
            fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey="row")
            fig.subplots_adjust(left=.10, right=.97, bottom=.11, top=.78, hspace=.35, wspace=.15)
            fig.text(.10, .955, "Model calibrations by true support size", fontsize=19, weight="bold")
            fig.text(.10, .915, f"{features} candidate parameters   /   SNR {snr:g}   /   Points show paired datasets; lines show medians.", color=MUTED)
            handles = []
            for method in [methods[0], *(m for m in variants if m["backward"])]:
                pruning = method["pruning"]
                style = ":" if method["name"] == BASELINE else "--" if method["larger_only"] else "-"
                color = INK if method["name"] == BASELINE else COLORS[pruning]
                label = "Backward only" if method["name"] == BASELINE else pruning.replace("off", "no pruning")
                if method["larger_only"]:
                    label += " (larger only)"
                handles.append(Line2D([], [], color=color, marker=MARKERS[pruning], ls=style, label=label))
            fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.09, .895), ncols=4, frameon=False)
            for row_index, design in enumerate(DESIGNS):
                group = [row for row in rows if condition(row) == (design, features, snr)]
                supports = sorted({row["support"] for row in group})
                separation = min((b - a for a, b in zip(supports, supports[1:])), default=1) * .065
                for col_index, backward in enumerate((True, False)):
                    ax = axes[row_index, col_index]
                    ax.set_title(f"{DESIGN_LABEL[design]}  /  Backward {'on' if backward else 'off'}", loc="left", fontweight="bold")
                    panel = [methods[0], *(method for method in variants if method["backward"] == backward)]
                    for index, method in enumerate(panel):
                        method_rows = [row for row in group if row["method"] == method["name"]]
                        offset = (index - (len(panel) - 1) / 2) * separation
                        color = INK if index == 0 else COLORS[method["pruning"]]
                        marker = MARKERS[method["pruning"]]
                        ax.scatter([row["support"] + offset for row in method_rows], [row["fits"] for row in method_rows],
                                   color=color, marker=marker, s=16, alpha=.22, linewidths=0)
                        ax.plot([support + offset for support in supports],
                                [median(row["fits"] for row in method_rows if row["support"] == support) for support in supports],
                                color=color, ls=":" if index == 0 else "--" if method["larger_only"] else "-",
                                marker=marker, markersize=4.5, linewidth=1.3)
                    ax.set_xticks(supports)
                    ax.grid(axis="y", color="#dce5e9", linewidth=.65)
                    ax.set_axisbelow(True)
                axes[row_index, 0].set_ylabel("Distinct fits")
            fig.supxlabel("Number of generating parameters", y=.045)
            pdf.savefig(fig)
            plt.close(fig)

        values = np.array([[median_saving([row for row in by_method[method["name"]] if condition(row) == group])
                            for group in groups] for method in variants])
        figure, ax = plt.subplots(figsize=(max(9, len(groups) * 1.03 + 2.0), 6.5), layout="constrained")
        norm = TwoSlopeNorm(vmin=min(-1.0, values.min()), vcenter=0, vmax=max(1.0, values.max()))
        chart = ax.imshow(values, aspect="auto", cmap="BrBG", norm=norm)
        for y, row in enumerate(values):
            for x, value in enumerate(row):
                ax.text(x, y, format_pct(value), va="center", ha="center", fontsize=8,
                        color="white" if abs(norm(value) - .5) > .35 else INK)
        ax.set_xticks(range(len(groups)), [f"{DESIGN_LABEL[d]}\n{p}p / SNR {s:g}" for d, p, s in groups])
        ax.set_yticks(range(len(variants)), [method["name"] for method in variants])
        ax.tick_params(length=0, labelsize=8)
        ax.set_title("Median paired fit saving vs. backward only", fontsize=16, loc="left", pad=12, fontweight="bold")
        ticks = [norm.vmin, norm.vmin / 2, 0, norm.vmax / 2, norm.vmax]
        bar = figure.colorbar(chart, ax=ax, fraction=.025, pad=.02, ticks=ticks, label="Fit saving (%)")
        bar.ax.set_yticklabels([f"{tick:.3g}" for tick in ticks])
        pdf.savefig(figure)
        preview = BytesIO()
        figure.savefig(preview, format="png", dpi=200)
        plt.close(figure)
    preview.seek(0)
    return preview


def make_report(path: Path, plan: dict, by_method: dict, preview: BytesIO) -> None:
    cases = plan["cases"]
    names = [method["name"] for method in plan["methods"] if method["name"] != BASELINE]
    groups = sorted({condition(case) for case in cases})
    baseline = by_method[BASELINE]
    oracle_count = sum(row["oracle_best_aicc"] is not None for row in baseline)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Headline", fontName="Helvetica-Bold", fontSize=26, leading=31,
                              textColor=colors.HexColor(INK), spaceAfter=9))
    styles.add(ParagraphStyle(name="Section", fontName="Helvetica-Bold", fontSize=14, leading=18,
                              textColor=colors.HexColor(INK), spaceBefore=8, spaceAfter=6))
    styles.add(ParagraphStyle(name="Text", fontName="Helvetica", fontSize=10.2, leading=14.2,
                              textColor=colors.HexColor(INK), spaceAfter=7))
    styles.add(ParagraphStyle(name="CaptionSmall", fontName="Helvetica", fontSize=8.3, leading=11,
                              textColor=colors.HexColor(MUTED), spaceAfter=8))
    styles.add(ParagraphStyle(name="Cell", fontName="Helvetica", fontSize=8.3, leading=10.5,
                              textColor=colors.HexColor(INK)))
    doc = SimpleDocTemplate(str(path), pagesize=A4, rightMargin=42, leftMargin=42, topMargin=39,
                            bottomMargin=45, title="Exhaustive search on synthetic data", author="ModelSelector benchmark")

    def paragraph(text: str, style: str = "Text") -> Paragraph:
        return Paragraph(text, styles[style])

    def table(data: list, widths: list[float]) -> Table:
        result = Table(data, colWidths=widths, hAlign="LEFT", repeatRows=1)
        result.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(INK)), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 8.3),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#eef4f5"), colors.white]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        return result

    def pooled(rows: list[dict]) -> float:
        return -100 * sum(row["delta_fits"] for row in rows) / sum(row["baseline_fits"] for row in rows)

    story = [paragraph("EXPERIMENT RESULTS  /  EXHAUSTIVE MODEL SELECTION", "CaptionSmall"),
             paragraph("Exhaustive search on synthetic data", "Headline"),
             paragraph(f"{len(cases)} paired datasets and {len(names) + 1} methods. "
                       f"All methods agreed on the best AICc within numerical tolerance. "
                       f"Every subset was separately scored in {oracle_count} of {len(cases)} cases."),
             paragraph("Where the heuristics helped", "Section")]
    design_rows = [["Design", "Lowest total fits", "Fits vs. baseline", "Fewer / same / more"]]
    for design in DESIGNS:
        candidates = {name: [row for row in by_method[name] if row["design"] == design] for name in by_method}
        totals = {name: sum(row["fits"] for row in rows) for name, rows in candidates.items()}
        best_total = min(totals.values())
        tied = [name for name, total in totals.items() if total == best_total]
        selected = candidates[tied[0]]
        wins = sum(row["delta_fits"] < 0 for row in selected)
        losses = sum(row["delta_fits"] > 0 for row in selected)
        label = tied[0] if len(tied) == 1 else f"{tied[0]} (+{len(tied) - 1} tied)"
        design_rows.append([DESIGN_LABEL[design], label,
                            f"{best_total} / {sum(row['baseline_fits'] for row in selected)}",
                            f"{wins} / {len(selected) - wins - losses} / {losses}"])
    story.extend([table(design_rows, [105, 133, 116, 140]), Spacer(1, 11)])

    best_name = max(by_method, key=lambda name: pooled(by_method[name]))
    comparisons = [(median_saving([row for row in by_method[name] if condition(row) == group]), name, group)
                   for name in names for group in groups]
    strong, weak = max(comparisons, key=lambda item: item[0]), min(comparisons, key=lambda item: item[0])
    story.append(paragraph(f"<b>{best_name}</b> had the largest pooled saving, <b>{format_pct(pooled(by_method[best_name]))}</b>. "
                           f"The condition-level medians ranged from <b>{format_pct(weak[0])}</b> for "
                           f"<b>{weak[1]}</b> ({condition_label(weak[2])}) to <b>{format_pct(strong[0])}</b> "
                           f"for <b>{strong[1]}</b> ({condition_label(strong[2])}). "
                           "Positive values mean fewer fits than backward only."))
    image = Image(preview)
    scale = min(493 / image.imageWidth, 335 / image.imageHeight)
    image.drawWidth, image.drawHeight = image.imageWidth * scale, image.imageHeight * scale
    story.extend([image, paragraph("Each cell is the median of paired savings across support sizes and replicates. "
                                   "B and N indicate whether backward search is enabled. The table reports wins and losses "
                                   "for the first named method if total fits are tied.", "CaptionSmall"), PageBreak(),
                  paragraph("METHOD COMPARISON  /  PAIRED WITHIN EACH DATASET", "CaptionSmall"),
                  paragraph("Where each method gained or lost", "Headline"),
                  paragraph("Pooled saving compares total fits. The median gives each dataset equal weight. "
                            "The strongest and weakest conditions use median paired savings within each condition.")])
    detail = [["Method", "Pooled", "Median", "Win/Tie/Loss", "Strongest condition", "Weakest condition"]]
    for name in names:
        rows = by_method[name]
        wins = sum(row["delta_fits"] < 0 for row in rows)
        losses = sum(row["delta_fits"] > 0 for row in rows)
        per_group = [(median_saving([row for row in rows if condition(row) == group]), group) for group in groups]
        low, high = min(per_group), max(per_group)
        detail.append([name, format_pct(pooled(rows)), format_pct(median_saving(rows)),
                       f"{wins}/{len(rows) - wins - losses}/{losses}",
                       paragraph(f"{condition_label(high[1])}<br/>{format_pct(high[0])}", "Cell"),
                       paragraph(f"{condition_label(low[1])}<br/>{format_pct(low[0])}", "Cell")])
    story.extend([table(detail, [95, 45, 44, 61, 125, 125]), Spacer(1, 10),
                  paragraph("A win means fewer fits on the same dataset. Larger-only starts pruning only when the current model has more parameters than the best model so far. All experimental methods include forward searching.", "CaptionSmall"),
                  PageBreak(), paragraph("BENCHMARK PROTOCOL", "CaptionSmall"), paragraph("How to read the results", "Headline")])
    settings = cases[0]
    features = ", ".join(str(value) for value in sorted({case["features"] for case in cases}))
    supports = ", ".join(str(value) for value in sorted({case["support"] for case in cases}))
    snrs = ", ".join(f"{value:g}" for value in sorted({case["snr"] for case in cases}))
    protocol = [
        ("Synthetic designs", f"Each dataset has {settings['observations']} observations. Candidate counts were {features}; "
         f"generating support sizes were {supports}; SNR values were {snrs}. The root seed was {settings['seed']}. "
         "The identifiable design uses centered orthogonal predictors. The pairwise-sum design samples predictors "
         "from a smaller orthogonal basis and its pairwise sums, producing a rank-deficient design."),
        ("Signal and noise", "Both generators use unit coefficients on a random support and noise projected out of "
         "the basis span and rescaled to the requested SNR. This creates a controlled structural comparison. "
         "A rank-deficient fit can have several equivalent supports, so the comparison checks the optimal criterion value."),
        ("Search rules", f"All methods use the package's linear fit and AICc. Ordered forward pruning uses {settings['saliency']} "
         "saliency from the smallest fitted proper supermodel, with the full model ordering the remaining parameters. "
         "One-step pruning fits every unseen immediate supermodel. The backward-only baseline includes exhaustive fallback "
         "and AICc bounds, with backward deletion as its sole heuristic. The bound combines the calibrated supermodel's "
         "twice negative log likelihood with the AICc penalty for the target submodel size."),
        ("Counting and validation", "The number of distinct calibrations is counted independently for each method. "
         "A shared deterministic cache reduces benchmark computation. The figures compare calibration counts. "
         "The exhaustive score check includes the empty model and accepts numerically tied optima."),
    ]
    for title, body in protocol:
        story.extend([paragraph(title, "Section"), paragraph(body)])
    if oracle_count < len(cases):
        story.append(paragraph("The remaining cases were checked for agreement across methods. Use a higher verification limit to score every subset there as well."))
    rank_rows = [["Design", "Candidate columns", "Realized rank", "Cases with tied optima"]]
    for design, width in sorted({(row["design"], row["features"]) for row in baseline}):
        selected = [row for row in baseline if (row["design"], row["features"]) == (design, width)]
        low, high = min(row["rank"] for row in selected), max(row["rank"] for row in selected)
        oracles = [row["oracle_tied_models"] for row in selected if row["oracle_tied_models"] is not None]
        ties = f"{sum(value > 1 for value in oracles)} / {len(oracles)}" if oracles else "Unavailable"
        rank_rows.append([DESIGN_LABEL[design], width, low if low == high else f"{low}-{high}", ties])
    story.extend([paragraph("Design rank and optimal ties", "Section"), table(rank_rows, [120, 114, 104, 156])])

    def footer(canvas, document) -> None:
        canvas.setStrokeColor(colors.HexColor("#c8d8dd"))
        canvas.line(42, 35, A4[0] - 42, 35)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor(MUTED))
        canvas.drawString(42, 23, "MODELSELECTOR  /  SYNTHETIC SEARCH BENCHMARK")
        canvas.drawRightString(A4[0] - 42, 23, str(document.page))

    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    try:
        plan, rows = load_results(args.directory)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"{exc}\n")
    output = args.output_dir or args.directory / "report"
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "results.csv", rows)
    by_method = {method["name"]: [row for row in rows if row["method"] == method["name"]] for method in plan["methods"]}
    preview = make_figures(output / "calibration_comparison.pdf", plan["methods"], rows, by_method)
    make_report(output / "synthetic_search_report.pdf", plan, by_method, preview)
    print(f"Wrote results, figures, and summary to {output}")


if __name__ == "__main__":
    main()
