"""Check and regenerate paper-facing numeric summaries.

This script keeps the manuscript's headline numbers tied to the CSV tables
produced by the pipeline. By default it only prints checks. With ``--write`` it
also writes:

- results/tables/paper_numbers_summary.csv
- results/tables/paper_numbers.json
- paper/paper_macros.tex
- paper/seed_supplement.tex

With ``--refresh-validation`` it replaces results/tables/phase2_validation.csv
with a compact diagnostic row derived from results/tables/fidelity.csv. This is
useful when old E2 validation rows were produced by stale caches or a previous
schema.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "tables"
PAPER = ROOT / "paper"


def _fmt_pct(x: float) -> str:
    return f"{x:+.2f}"


def _fmt_p(p: float, prefix: str = "p") -> str:
    """Math-mode p-value: thresholds below 1e-4, two significant figures above."""
    if not math.isfinite(p):
        return f"{prefix}=\\text{{n/a}}"
    if p < 1e-4:
        return f"{prefix}<10^{{-{math.floor(-math.log10(p))}}}"
    return f"{prefix}={p:.2g}"


def _fmt_p_math(p: float) -> str:
    """Bare math-mode p-value body (no $ delimiters) for table cells."""
    if not math.isfinite(p):
        return "\\text{--}"
    if p < 1e-4:
        return f"<10^{{-{math.floor(-math.log10(p))}}}"
    return f"{p:.2g}"


def _load_csv(name: str) -> pd.DataFrame:
    path = RESULTS / name
    if not path.exists():
        raise FileNotFoundError(f"missing {path}")
    return pd.read_csv(path)


def _load_csv_prefer(*names: str) -> pd.DataFrame:
    for name in names:
        path = RESULTS / name
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError(f"missing all candidate CSVs: {', '.join(names)}")


def _maybe_csv(name: str) -> pd.DataFrame | None:
    path = RESULTS / name
    return pd.read_csv(path) if path.exists() else None


def build_summary() -> pd.DataFrame:
    seed_sig = _load_csv_prefer("seed_significance_hard_co280.csv", "seed_significance.csv")
    fidelity = _load_csv("fidelity.csv")
    surrogate = _load_csv("surrogate_eval.csv")
    sweep = _load_csv("carbon_sweep.csv")
    carbon_ensemble = _maybe_csv("carbon_ensemble.csv")
    speed = _maybe_csv("speed_benchmark_repeated.csv")
    speed_single = _maybe_csv("speed_benchmark.csv")
    tout = _maybe_csv("tout_audit.csv")
    cost_decomp = _maybe_csv("cost_decomposition_summary.csv")
    svh = _maybe_csv("soft_vs_hard.csv")
    svh_contrast = _maybe_csv("soft_vs_hard_contrast.csv")
    fabrication = _maybe_csv("fabrication_diagnostic.csv")
    mixing = _maybe_csv("mixing_sensitivity.csv")
    volmatch_sig = _maybe_csv("seed_significance_hard_volmatch_co280.csv")
    volmatch_decomp = _maybe_csv("cost_decomposition_summary_volmatch.csv")

    lagged = seed_sig[seed_sig["baseline"] == "lagged"].copy()
    aggregate = lagged[lagged["scope"].astype(str).str.startswith("ALL_")].iloc[0]

    sweep0 = sweep[sweep["price_co2_eur_per_t"] == 0.0]["saving_vs_lagged_pct"].mean()
    sweep80 = sweep[sweep["price_co2_eur_per_t"] == 80.0]["saving_vs_lagged_pct"].mean()

    constant = sweep.copy()
    constant["saving_vs_constant_pct"] = (
        (constant["constant"] - constant["aware"]) / constant["constant"] * 100.0
    )
    constant0 = constant[constant["price_co2_eur_per_t"] == 0.0][
        "saving_vs_constant_pct"
    ].mean()
    constant80 = constant[constant["price_co2_eur_per_t"] == 80.0][
        "saving_vs_constant_pct"
    ].mean()

    w_total = surrogate[surrogate["channel"] == "W_total"].iloc[0]
    market_hours = len(
        pd.date_range("2021-01-01", "2026-01-01", freq="h", inclusive="left")
    )

    rows = [
        {
            "metric": "record_count_market_hours",
            "value": market_hours,
            "display": f"{market_hours:,}",
        },
        {
            "metric": "headline_saving_vs_lagged_pct",
            "value": aggregate["mean_pct"],
            "display": _fmt_pct(float(aggregate["mean_pct"])),
        },
        {
            "metric": "headline_ci95_lo_pct",
            "value": aggregate["ci95_lo_pct"],
            "display": _fmt_pct(float(aggregate["ci95_lo_pct"])),
        },
        {
            "metric": "headline_ci95_hi_pct",
            "value": aggregate["ci95_hi_pct"],
            "display": _fmt_pct(float(aggregate["ci95_hi_pct"])),
        },
        {
            "metric": "headline_p_two_sided",
            "value": aggregate["p_two_sided"],
            "display": _fmt_p(float(aggregate["p_two_sided"])),
        },
        {
            "metric": "headline_seed_n",
            "value": aggregate["n"],
            "display": str(int(aggregate["n"])),
        },
        {
            "metric": "single_seed_sweep_mean_saving_at_0_pct",
            "value": sweep0,
            "display": _fmt_pct(float(sweep0)),
        },
        {
            "metric": "single_seed_sweep_mean_saving_at_80_pct",
            "value": sweep80,
            "display": _fmt_pct(float(sweep80)),
        },
        {
            "metric": "single_seed_constant_flow_saving_at_0_pct",
            "value": constant0,
            "display": _fmt_pct(float(constant0)),
        },
        {
            "metric": "single_seed_constant_flow_saving_at_80_pct",
            "value": constant80,
            "display": _fmt_pct(float(constant80)),
        },
        {
            "metric": "w_total_mae_kwh_per_kg",
            "value": w_total["MAE"],
            "display": f"{float(w_total['MAE']):.3g}",
        },
        {
            "metric": "w_total_r2",
            "value": w_total["R2"],
            "display": f"{float(w_total['R2']):.6f}",
        },
        {
            "metric": "fidelity_median_abs_rel_error",
            "value": fidelity["rel_error"].abs().median(),
            "display": f"{float(fidelity['rel_error'].abs().median()):.3g}",
        },
        {
            "metric": "fidelity_p95_abs_rel_error",
            "value": fidelity["rel_error"].abs().quantile(0.95),
            "display": f"{float(fidelity['rel_error'].abs().quantile(0.95)):.3g}",
        },
        {
            "metric": "fidelity_max_abs_rel_error",
            "value": fidelity["rel_error"].abs().max(),
            "display": f"{float(fidelity['rel_error'].abs().max()):.3g}",
        },
    ]
    wilcoxon_val = float(aggregate.get("wilcoxon_p_two_sided", float("nan")))
    if math.isfinite(wilcoxon_val):
        rows.append(
            {
                "metric": "headline_wilcoxon_p_two_sided",
                "value": wilcoxon_val,
                "display": _fmt_p(wilcoxon_val),
            }
        )
    if svh is not None and not svh.empty:
        cells = svh[svh["baseline"].astype(str) == "lagged"]
        for _, row in cells.iterrows():
            tag = f"{row['surrogate']}_at_{int(float(row['carbon_price_eur_per_t']))}"
            rows.extend(
                [
                    {
                        "metric": f"svh_{tag}_mean_pct",
                        "value": row["mean_saving_pct"],
                        "display": f"{float(row['mean_saving_pct']):+.3f}",
                    },
                    {
                        "metric": f"svh_{tag}_ci95_lo_pct",
                        "value": row["ci95_lo_pct"],
                        "display": f"{float(row['ci95_lo_pct']):+.3f}",
                    },
                    {
                        "metric": f"svh_{tag}_ci95_hi_pct",
                        "value": row["ci95_hi_pct"],
                        "display": f"{float(row['ci95_hi_pct']):+.3f}",
                    },
                ]
            )
    if svh_contrast is not None and not svh_contrast.empty:
        for _, row in svh_contrast.iterrows():
            tag = int(float(row["carbon_price_eur_per_t"]))
            rows.append(
                {
                    "metric": f"svh_contrast_at_{tag}_pp",
                    "value": row["soft_minus_hard_saving_pct"],
                    "display": f"{float(row['soft_minus_hard_saving_pct']):+.4f}",
                }
            )
        max_abs = float(svh_contrast["soft_minus_hard_saving_pct"].abs().max())
        rows.append(
            {
                "metric": "svh_max_abs_contrast_pp",
                "value": max_abs,
                "display": f"{max_abs:.4f}",
            }
        )
    if fabrication is not None and not fabrication.empty:
        for arch, g in fabrication.groupby(fabrication["surrogate"].astype(str)):
            flagged = int(
                round((g["frac_windows_flagged"] * g["n_windows"]).sum())
            )
            rows.append(
                {
                    "metric": f"fab_{arch}_flagged_windows",
                    "value": flagged,
                    "display": str(flagged),
                }
            )
        rows.extend(
            [
                {
                    "metric": "fab_seed_count",
                    "value": fabrication["seed"].nunique(),
                    "display": str(int(fabrication["seed"].nunique())),
                },
                {
                    "metric": "fab_windows_per_seed",
                    "value": fabrication["n_windows"].max(),
                    "display": str(int(fabrication["n_windows"].max())),
                },
            ]
        )
    if speed_single is not None and not speed_single.empty:
        row = speed_single.iloc[0]
        rows.extend(
            [
                {
                    "metric": "bench_pinn_ms_per_point",
                    "value": row["pinn_ms_per_point"],
                    "display": f"{float(row['pinn_ms_per_point']):.3g}",
                },
                {
                    "metric": "bench_coolprop_ms_per_point",
                    "value": row["coolprop_ms_per_point"],
                    "display": f"{float(row['coolprop_ms_per_point']):.0f}",
                },
                {
                    "metric": "bench_machine",
                    "value": str(row["machine"]),
                    "display": str(row["machine"]),
                },
            ]
        )
    if mixing is not None and not mixing.empty:
        for kernel, label in [("linear", "linear"), ("exp", "cstr")]:
            g = mixing[mixing["kernel"].astype(str) == kernel]
            if g.empty:
                continue
            rows.extend(
                [
                    {
                        "metric": f"mixing_{label}_min_pct",
                        "value": g["mean_pct"].min(),
                        "display": _fmt_pct(float(g["mean_pct"].min())),
                    },
                    {
                        "metric": f"mixing_{label}_max_pct",
                        "value": g["mean_pct"].max(),
                        "display": _fmt_pct(float(g["mean_pct"].max())),
                    },
                ]
            )
        rows.append(
            {
                "metric": "mixing_min_ci_lo_pct",
                "value": mixing["ci_low"].min(),
                "display": _fmt_pct(float(mixing["ci_low"].min())),
            }
        )
        if "wilcoxon_p_two_sided" in mixing.columns:
            wmax = float(mixing["wilcoxon_p_two_sided"].max())
            if math.isfinite(wmax):
                rows.append(
                    {
                        "metric": "mixing_wilcoxon_max_p",
                        "value": wmax,
                        "display": _fmt_p(wmax),
                    }
                )
    if volmatch_sig is not None and not volmatch_sig.empty:
        vm = volmatch_sig[
            (volmatch_sig["baseline"].astype(str) == "lagged")
            & volmatch_sig["scope"].astype(str).str.startswith("ALL_")
        ]
        if not vm.empty:
            row = vm.iloc[0]
            rows.extend(
                [
                    {
                        "metric": "volmatch_saving_pct",
                        "value": row["mean_pct"],
                        "display": f"{float(row['mean_pct']):+.3f}",
                    },
                    {
                        "metric": "volmatch_ci95_lo_pct",
                        "value": row["ci95_lo_pct"],
                        "display": f"{float(row['ci95_lo_pct']):+.3f}",
                    },
                    {
                        "metric": "volmatch_ci95_hi_pct",
                        "value": row["ci95_hi_pct"],
                        "display": f"{float(row['ci95_hi_pct']):+.3f}",
                    },
                    {
                        "metric": "volmatch_p_two_sided",
                        "value": row["p_two_sided"],
                        "display": _fmt_p(float(row["p_two_sided"])),
                    },
                    {
                        "metric": "volmatch_seed_n",
                        "value": row["n"],
                        "display": str(int(row["n"])),
                    },
                ]
            )
            vm_wilcoxon = float(row.get("wilcoxon_p_two_sided", float("nan")))
            if math.isfinite(vm_wilcoxon):
                rows.append(
                    {
                        "metric": "volmatch_wilcoxon_p_two_sided",
                        "value": vm_wilcoxon,
                        "display": _fmt_p(vm_wilcoxon),
                    }
                )
    if volmatch_decomp is not None and not volmatch_decomp.empty:
        hit = volmatch_decomp[
            (volmatch_decomp["baseline"].astype(str) == "lagged")
            & (volmatch_decomp["scope"].astype(str) == "ALL_5yr_seed_mean")
        ]
        if not hit.empty:
            row = hit.iloc[0]
            rows.append(
                {
                    "metric": "volmatch_delivered_mass_delta_pct",
                    "value": row["mean_delivered_mass_delta_pct"],
                    "display": f"{float(row['mean_delivered_mass_delta_pct']):+.3f}",
                }
            )
            if "mean_saving_per_kg_pct" in volmatch_decomp.columns:
                rows.append(
                    {
                        "metric": "volmatch_saving_per_kg_pct",
                        "value": row["mean_saving_per_kg_pct"],
                        "display": f"{float(row['mean_saving_per_kg_pct']):+.3f}",
                    }
                )
        year_rows = volmatch_decomp[
            (volmatch_decomp["baseline"].astype(str) == "lagged")
            & (volmatch_decomp["scope"].astype(str) != "ALL_5yr_seed_mean")
        ]
        if not year_rows.empty:
            vmax = float(year_rows["mean_delivered_mass_delta_pct"].abs().max())
            rows.append(
                {
                    "metric": "volmatch_max_abs_year_mass_delta_pct",
                    "value": vmax,
                    "display": f"{vmax:.2f}",
                }
            )
    if carbon_ensemble is not None:
        ce_lagged = carbon_ensemble[
            (carbon_ensemble["surrogate"].astype(str) == "hard")
            & (carbon_ensemble["baseline"].astype(str) == "lagged")
        ]
        for price in [0.0, 80.0, 160.0]:
            hit = ce_lagged[ce_lagged["carbon_price_eur_per_t"].astype(float) == price]
            if hit.empty:
                continue
            row = hit.iloc[0]
            price_tag = str(int(price))
            rows.extend(
                [
                    {
                        "metric": f"ensemble_saving_at_{price_tag}_pct",
                        "value": row["mean_saving_pct"],
                        "display": _fmt_pct(float(row["mean_saving_pct"])),
                    },
                    {
                        "metric": f"ensemble_ci95_lo_at_{price_tag}_pct",
                        "value": row["ci95_lo_pct"],
                        "display": _fmt_pct(float(row["ci95_lo_pct"])),
                    },
                    {
                        "metric": f"ensemble_ci95_hi_at_{price_tag}_pct",
                        "value": row["ci95_hi_pct"],
                        "display": _fmt_pct(float(row["ci95_hi_pct"])),
                    },
                ]
            )
    if speed is not None and not speed.empty:
        row = speed.iloc[0]
        rows.extend(
            [
                {
                    "metric": "benchmark_surrogate_median_s",
                    "value": row["surrogate_median_s"],
                    "display": f"{float(row['surrogate_median_s']):.5f}",
                },
                {
                    "metric": "benchmark_coolprop_median_s",
                    "value": row["coolprop_median_s"],
                    "display": f"{float(row['coolprop_median_s']):.2f}",
                },
                {
                    "metric": "benchmark_speedup_x",
                    "value": row["speedup_x"],
                    "display": f"{float(row['speedup_x']):.2g}",
                },
            ]
        )
    if tout is not None and not tout.empty:
        row = tout.iloc[0]
        rows.extend(
            [
                {
                    "metric": "tout_train_span_K",
                    "value": row["train_span_K"],
                    "display": f"{float(row['train_span_K']):.3g}",
                },
                {
                    "metric": "tout_degenerate_setpoint_channel",
                    "value": bool(row["degenerate_setpoint_channel"]),
                    "display": str(bool(row["degenerate_setpoint_channel"])),
                },
            ]
        )
        if "train_std_K" in tout.columns:
            rows.append(
                {
                    "metric": "tout_train_std_K",
                    "value": row["train_std_K"],
                    "display": f"{float(row['train_std_K']):.3g}",
                }
            )
    if cost_decomp is not None and not cost_decomp.empty:
        hit = cost_decomp[
            (cost_decomp["baseline"].astype(str) == "lagged")
            & (cost_decomp["scope"].astype(str) == "ALL_5yr_seed_mean")
        ]
        if not hit.empty:
            row = hit.iloc[0]
            rows.extend(
                [
                    {
                        "metric": "decomp_carbon_fraction_of_saving",
                        "value": row["aggregate_carbon_fraction_of_saving"],
                        "display": f"{float(row['aggregate_carbon_fraction_of_saving']):.2f}",
                    },
                    {
                        "metric": "decomp_carbon_share_pct",
                        "value": float(row["aggregate_carbon_fraction_of_saving"]) * 100.0,
                        "display": f"{float(row['aggregate_carbon_fraction_of_saving']) * 100.0:.0f}",
                    },
                    {
                        "metric": "decomp_mean_co2_saving_t",
                        "value": row["mean_co2_saving_t"],
                        "display": f"{float(row['mean_co2_saving_t']):.1f}",
                    },
                    {
                        "metric": "decomp_delivered_mass_delta_pct",
                        "value": row["mean_delivered_mass_delta_pct"],
                        "display": f"{float(row['mean_delivered_mass_delta_pct']):+.3f}",
                    },
                ]
            )
            if "mean_saving_per_kg_pct" in cost_decomp.columns:
                rows.extend(
                    [
                        {
                            "metric": "decomp_saving_per_kg_pct",
                            "value": row["mean_saving_per_kg_pct"],
                            "display": f"{float(row['mean_saving_per_kg_pct']):+.3f}",
                        },
                        {
                            "metric": "decomp_per_kg_ci95_lo_pct",
                            "value": row["ci95_lo_per_kg_pct"],
                            "display": f"{float(row['ci95_lo_per_kg_pct']):+.3f}",
                        },
                        {
                            "metric": "decomp_per_kg_ci95_hi_pct",
                            "value": row["ci95_hi_per_kg_pct"],
                            "display": f"{float(row['ci95_hi_per_kg_pct']):+.3f}",
                        },
                        {
                            "metric": "decomp_co2_intensity_saving_pct",
                            "value": row["mean_co2_intensity_saving_pct"],
                            "display": f"{float(row['mean_co2_intensity_saving_pct']):+.3f}",
                        },
                    ]
                )
    return pd.DataFrame(rows)


# LaTeX control sequences may only contain letters, so digits in metric names
# are spelled out (ci95 -> CiNineFive, at_80 -> AtEightZero).
_DIGIT_WORDS = {
    "0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four",
    "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine",
}


def _macro_name(metric: str) -> str:
    parts = re.sub(r"[^0-9A-Za-z]+", " ", metric).title().split()
    name = "lng" + "".join(parts)
    return "".join(_DIGIT_WORDS.get(ch, ch) for ch in name)


def _json_value(value: object) -> object:
    if hasattr(value, "item"):
        return value.item()
    return value


def write_machine_readable(summary: pd.DataFrame) -> None:
    payload = {
        str(row.metric): {
            "value": _json_value(row.value),
            "display": str(row.display),
        }
        for row in summary.itertuples(index=False)
    }
    (RESULTS / "paper_numbers.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    lines = [
        "% Auto-generated by scripts/09_paper_numbers.py --write",
        "% Use display strings for manuscript text; numeric values live in JSON.",
    ]
    for row in summary.itertuples(index=False):
        lines.append(f"\\newcommand{{\\{_macro_name(str(row.metric))}}}{{{row.display}}}")
    lines.append("")
    (PAPER / "paper_macros.tex").write_text("\n".join(lines), encoding="utf-8")


def write_seed_supplement() -> None:
    df = _load_csv("seed_sensitivity.csv")
    pivot = (
        df.pivot_table(
            index="seed",
            columns="year",
            values="saving_vs_lagged_pct",
            aggfunc="first",
        )
        .sort_index()
        .round(2)
    )

    lines = [
        "% Auto-generated by scripts/09_paper_numbers.py --write",
        "\\begin{table}[h]",
        "  \\centering",
        "  \\caption{Composition-aware saving versus the lagged-composition "
        "baseline by cargo-schedule seed and year at "
        "\\SI{80}{\\EUR\\per\\tCOtwo}. Values are percentages.}",
        "  \\label{tab:seed-supplement}",
        "  \\begin{tabular}{rrrrrr}",
        "    \\toprule",
        "    Seed & 2021 & 2022 & 2023 & 2024 & 2025 \\\\",
        "    \\midrule",
    ]
    for seed, row in pivot.iterrows():
        values = " & ".join(f"{float(row[year]):+.2f}" for year in pivot.columns)
        lines.append(f"    {int(seed)} & {values} \\\\")
    lines.extend([
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
        "",
    ])
    (PAPER / "seed_supplement.tex").write_text("\n".join(lines), encoding="utf-8")


def write_seed_significance_table() -> None:
    """Emit the per-year significance tabular consumed by tab:seed-significance."""
    df = _load_csv_prefer("seed_significance_hard_co280.csv", "seed_significance.csv")
    lagged = df[df["baseline"].astype(str) == "lagged"].copy()
    has_wilcoxon = "wilcoxon_p_two_sided" in lagged.columns

    def render(row: pd.Series, bold: bool = False) -> str:
        scope = str(row["scope"])
        label = "5-year mean" if scope.startswith("ALL_") else scope
        ci_lo, ci_hi = float(row["ci95_lo_pct"]), float(row["ci95_hi_pct"])
        dagger = "^{\\dagger}" if ci_lo <= 0.0 <= ci_hi else ""

        def m(body: str) -> str:
            return f"$\\mathbf{{{body}}}$" if bold else f"${body}$"

        cells = [
            f"\\textbf{{{label}}}" if bold else label,
            f"\\textbf{{{int(row['n'])}}}" if bold else str(int(row["n"])),
            m(f"{float(row['mean_pct']):+.2f}"),
            m(f"{float(row['se_pct']):.2f}"),
            m(f"[{ci_lo:+.2f}, {ci_hi:+.2f}]"),
            m(f"{float(row['t_stat']):.2f}"),
            m(_fmt_p_math(float(row["p_two_sided"])) + dagger),
        ]
        if has_wilcoxon:
            cells.append(m(_fmt_p_math(float(row["wilcoxon_p_two_sided"]))))
        return "    " + " & ".join(cells) + " \\\\"

    lines = [
        "% Auto-generated by scripts/09_paper_numbers.py --write",
        f"\\begin{{tabular}}{{lrrrrrr{'r' if has_wilcoxon else ''}}}",
        "    \\toprule",
        "    Scope & $n$ & Mean (\\%) & SE (\\%) & 95\\% CI (\\%) & $t$ & $p$"
        + (" & Wilcoxon $p$" if has_wilcoxon else "")
        + " \\\\",
        "    \\midrule",
    ]
    for year in range(2021, 2026):
        hit = lagged[lagged["scope"].astype(str) == str(year)]
        if not hit.empty:
            lines.append(render(hit.iloc[0]))
    agg = lagged[lagged["scope"].astype(str).str.startswith("ALL_")]
    if not agg.empty:
        lines.append("    \\midrule")
        lines.append(render(agg.iloc[0], bold=True))
    lines.extend(["    \\bottomrule", "\\end{tabular}", ""])
    (PAPER / "seed_significance_table.tex").write_text("\n".join(lines), encoding="utf-8")


def write_mixing_table() -> None:
    """Emit the tank-mixing robustness tabular consumed by tab:mixing-sensitivity."""
    path = RESULTS / "mixing_table3.csv"
    if not path.exists():
        print("skipping mixing table include: missing mixing_table3.csv")
        return
    df = pd.read_csv(path)

    def cell(text: object, bold: bool) -> str:
        if pd.isna(text) or str(text) == "":
            return "--"
        body = str(text).replace("[", "\\;[")
        return f"$\\mathbf{{{body}}}$" if bold else f"${body}$"

    lines = [
        "% Auto-generated by scripts/09_paper_numbers.py --write",
        "\\begin{tabular}{lrcrc}",
        "    \\toprule",
        "    $\\tau_{\\text{mix}}$ (d) & $n$ & Linear ramp \\% [95\\% CI]"
        " & $n$ & First-order CSTR \\% [95\\% CI] \\\\",
        "    \\midrule",
    ]
    for _, row in df.sort_values("tau_days").iterrows():
        tau = float(row["tau_days"])
        bold = tau == 5.0  # default mixing timescale
        tau_label = f"{tau:g}"
        lines.append(
            "    "
            + " & ".join(
                [
                    f"\\textbf{{{tau_label}}}" if bold else tau_label,
                    str(int(row.get("linear_n", 0))),
                    cell(row.get("linear_saving_pct_ci95"), bold),
                    str(int(row.get("exp_n", 0))),
                    cell(row.get("exp_saving_pct_ci95"), bold),
                ]
            )
            + " \\\\"
        )
    lines.extend(["    \\bottomrule", "\\end{tabular}", ""])
    (PAPER / "mixing_table.tex").write_text("\n".join(lines), encoding="utf-8")


_CHANNEL_LABELS = {
    "W_pump": "$W_{\\text{pump}}$ (kWh/kg)",
    "W_total": "$W_{\\text{total}}$ (kWh/kg)",
    "T_out": "$T_{\\text{out}}$ (K)",
    "exergy_destruction": "$e_{\\text{exergy}}$ (kWh/kg)",
}


def write_surrogate_eval_table() -> None:
    """Emit the per-channel fidelity tabular (hard and, when present, soft)."""
    hard = _load_csv("surrogate_eval.csv").set_index("channel")
    soft_df = _maybe_csv("surrogate_eval_soft.csv")
    soft = soft_df.set_index("channel") if soft_df is not None else None

    cols = "lrr" + ("rr" if soft is not None else "")
    header = "    Channel & MAE & $R^2$"
    if soft is not None:
        header = "    Channel & Hard MAE & Hard $R^2$ & Soft MAE & Soft $R^2$"
    lines = [
        "% Auto-generated by scripts/09_paper_numbers.py --write",
        f"\\begin{{tabular}}{{{cols}}}",
        "    \\toprule",
        header + " \\\\",
        "    \\midrule",
    ]
    for channel, label in _CHANNEL_LABELS.items():
        if channel not in hard.index:
            continue
        h = hard.loc[channel]
        cells = [label, f"\\num{{{float(h['MAE']):.3g}}}", f"{float(h['R2']):.6f}"]
        if soft is not None:
            if channel in soft.index:
                s = soft.loc[channel]
                cells.extend(
                    [f"\\num{{{float(s['MAE']):.3g}}}", f"{float(s['R2']):.6f}"]
                )
            else:
                cells.extend(["--", "--"])
        lines.append("    " + " & ".join(cells) + " \\\\")
    lines.extend(["    \\bottomrule", "\\end{tabular}", ""])
    (PAPER / "surrogate_eval_table.tex").write_text("\n".join(lines), encoding="utf-8")


def refresh_phase2_validation() -> None:
    fidelity = _load_csv("fidelity.csv")
    rel = fidelity["rel_error"].dropna()
    row = pd.DataFrame([{
        "script": "05_make_figures_fidelity",
        "carbon_price_eur_per_t": 80.0,
        "seed": 42,
        "strategy": "aware",
        "n_total": len(fidelity),
        "n_sampled": len(fidelity),
        "mean_rel_err": float(rel.mean()),
        "median_abs_rel_err": float(rel.abs().median()),
        "p95_abs_rel_err": float(rel.abs().quantile(0.95)),
        "max_abs_rel_err": float(rel.abs().max()),
    }])
    row.to_csv(RESULTS / "phase2_validation.csv", index=False)


def check_phase2_validation() -> str:
    path = RESULTS / "phase2_validation.csv"
    if not path.exists():
        return "phase2_validation.csv: missing"
    df = pd.read_csv(path)
    if df.empty or "max_abs_rel_err" not in df.columns:
        return "phase2_validation.csv: malformed"
    max_err = float(df["max_abs_rel_err"].max())
    if max_err > 1e-3:
        return (
            "phase2_validation.csv: suspicious "
            f"(max_abs_rel_err={max_err:.3g}); refresh from fidelity.csv"
        )
    return f"phase2_validation.csv: ok (max_abs_rel_err={max_err:.3g})"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write",
        action="store_true",
        help="write summary CSV, JSON/macros, and appendix table",
    )
    parser.add_argument(
        "--refresh-validation",
        action="store_true",
        help="replace phase2_validation.csv with a fidelity-derived diagnostic row",
    )
    args = parser.parse_args()

    summary = build_summary()
    print(summary[["metric", "display"]].to_string(index=False))
    print(check_phase2_validation())

    if args.write:
        summary.to_csv(RESULTS / "paper_numbers_summary.csv", index=False)
        write_machine_readable(summary)
        write_seed_supplement()
        write_seed_significance_table()
        write_mixing_table()
        write_surrogate_eval_table()
        print("wrote results/tables/paper_numbers_summary.csv")
        print("wrote results/tables/paper_numbers.json")
        print("wrote paper/paper_macros.tex")
        print("wrote paper/seed_supplement.tex")
        print("wrote paper/seed_significance_table.tex")
        print("wrote paper/mixing_table.tex")
        print("wrote paper/surrogate_eval_table.tex")

    if args.refresh_validation:
        refresh_phase2_validation()
        print("refreshed results/tables/phase2_validation.csv")


if __name__ == "__main__":
    main()
