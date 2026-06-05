"""All paper figures generated here - one function per figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

FIG_DIR = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")


def _seed_stats_per_year(
    baseline: str = "lagged",
) -> tuple[dict[int, float], dict[int, float]] | None:
    """Return ({year: mean}, {year: std}) of aware-vs-<baseline>% across seeds.

    The mean is the seed-mean point estimate (the value to trust as the
    best estimator of "what saving this year would produce on average across
    cargo schedules"). The single-seed carbon sweep is one draw from this
    distribution and may sit anywhere within the std band.
    """
    seed_files = sorted(Path("data/processed").glob("seed_sensitivity_seed*.parquet"))
    seed_files = [p for p in seed_files if "partial" not in p.name]
    if not seed_files:
        return None
    try:
        rows = []
        for path in seed_files:
            try:
                seed_id = int(path.stem.replace("seed_sensitivity_seed", ""))
            except ValueError:
                continue
            df = pd.read_parquet(path)
            if "_strategy" not in df.columns or "time" not in df.columns:
                continue
            df["time"] = pd.to_datetime(df["time"], utc=True)
            df["year"] = df["time"].dt.year
            pivot = df.groupby(["year", "_strategy"])["cost_eur"].sum().unstack()
            if not ({"aware", baseline} <= set(pivot.columns)):
                continue
            for year, row in pivot.iterrows():
                rows.append({
                    "seed": seed_id,
                    "year": int(year),
                    "saving_pct": (row[baseline] - row["aware"]) / row[baseline] * 100.0,
                })
        if rows:
            g = pd.DataFrame(rows).groupby("year")["saving_pct"]
            means = {int(y): float(v) for y, v in g.mean().items()}
            stds = {int(y): float(v) for y, v in g.std().items()}
            return means, stds
    except Exception:
        pass
    return None


def _seed_noise_per_year(baseline: str = "lagged") -> dict[int, float] | None:
    """Return {year: std of aware-vs-<baseline>% across seeds} for the year-by-year
    error bars in fig6 Panel A.

    Preserves the per-year noise structure rather than collapsing to a single
    scalar — 2022 (energy crisis) has materially higher cross-seed variance
    than 2021/2023, and the figure should show that honestly.
    """
    seed_files = sorted(Path("data/processed").glob("seed_sensitivity_seed*.parquet"))
    seed_files = [p for p in seed_files if "partial" not in p.name]
    if not seed_files:
        return None
    try:
        rows = []
        for path in seed_files:
            try:
                seed_id = int(path.stem.replace("seed_sensitivity_seed", ""))
            except ValueError:
                continue
            df = pd.read_parquet(path)
            if "_strategy" not in df.columns or "time" not in df.columns:
                continue
            df["time"] = pd.to_datetime(df["time"], utc=True)
            df["year"] = df["time"].dt.year
            pivot = df.groupby(["year", "_strategy"])["cost_eur"].sum().unstack()
            if not ({"aware", baseline} <= set(pivot.columns)):
                continue
            for year, row in pivot.iterrows():
                rows.append({
                    "seed": seed_id,
                    "year": int(year),
                    "saving_pct": (row[baseline] - row["aware"]) / row[baseline] * 100.0,
                })
        if rows:
            per_year = pd.DataFrame(rows).groupby("year")["saving_pct"].std()
            return {int(y): float(s) for y, s in per_year.items()}
    except Exception:
        pass
    return None


def _seed_mean_se_per_year(
    baseline: str = "lagged",
) -> tuple[dict[int, float], dict[int, float]] | None:
    """Return ({year: mean}, {year: standard error}) of aware-vs-<baseline>%.

    SE = std / sqrt(n_seeds) is the spread of the *mean estimator* — the
    quantity the paper's confidence interval is built on, distinct from the
    single-seed prediction spread (std) used for the wider error bar.
    """
    seed_files = sorted(Path("data/processed").glob("seed_sensitivity_seed*.parquet"))
    seed_files = [p for p in seed_files if "partial" not in p.name]
    if not seed_files:
        return None
    try:
        rows = []
        for path in seed_files:
            try:
                seed_id = int(path.stem.replace("seed_sensitivity_seed", ""))
            except ValueError:
                continue
            df = pd.read_parquet(path)
            if "_strategy" not in df.columns or "time" not in df.columns:
                continue
            df["time"] = pd.to_datetime(df["time"], utc=True)
            df["year"] = df["time"].dt.year
            pivot = df.groupby(["year", "_strategy"])["cost_eur"].sum().unstack()
            if not ({"aware", baseline} <= set(pivot.columns)):
                continue
            for year, row in pivot.iterrows():
                rows.append({
                    "seed": seed_id,
                    "year": int(year),
                    "saving_pct": (row[baseline] - row["aware"]) / row[baseline] * 100.0,
                })
        if rows:
            g = pd.DataFrame(rows).groupby("year")["saving_pct"]
            means = {int(y): float(v) for y, v in g.mean().items()}
            counts = g.count()
            stds = g.std()
            ses = {int(y): float(stds[y] / (counts[y] ** 0.5)) for y in counts.index}
            return means, ses
    except Exception:
        pass
    return None


def _seed_noise_std(baseline: str = "lagged") -> float | None:
    """Estimate the cross-year mean of the per-year std of aware-vs-<baseline>%.

    Tries sources in order:
    1. data/processed/seed_sensitivity_seed*.parquet — per-seed Phase-1
       dispatch caches. Preferred because they're regenerated by every fresh
       06 run; v1.3 PINN-predicted costs match CoolProp truth to ~1e-7
       relative error so the noise band computed from PINN costs is
       interchangeable with the post-Phase-2 truth-based one.
    2. results/tables/seed_sensitivity_summary.csv — the published summary
       written by scripts/06_seed_sensitivity.py at the very end of Phase 2.
       Used only when no Phase-1 caches are on disk (e.g. after a clean
       checkout from git).

    Returns None if neither source is usable.
    """
    seed_files = sorted(Path("data/processed").glob("seed_sensitivity_seed*.parquet"))
    seed_files = [p for p in seed_files if "partial" not in p.name]
    if seed_files:
        try:
            rows = []
            for path in seed_files:
                try:
                    seed_id = int(path.stem.replace("seed_sensitivity_seed", ""))
                except ValueError:
                    continue
                df = pd.read_parquet(path)
                if "_strategy" not in df.columns or "time" not in df.columns:
                    continue
                df["time"] = pd.to_datetime(df["time"], utc=True)
                df["year"] = df["time"].dt.year
                pivot = df.groupby(["year", "_strategy"])["cost_eur"].sum().unstack()
                if not ({"aware", baseline} <= set(pivot.columns)):
                    continue
                for year, row in pivot.iterrows():
                    rows.append({
                        "seed": seed_id,
                        "year": int(year),
                        "saving_pct": (row[baseline] - row["aware"]) / row[baseline] * 100.0,
                    })
            if rows:
                per_year_std = pd.DataFrame(rows).groupby("year")["saving_pct"].std()
                return float(per_year_std.mean())
        except Exception:
            pass

    summary_path = Path("results/tables/seed_sensitivity_summary.csv")
    if summary_path.exists():
        try:
            df = pd.read_csv(summary_path)
            rows = df[df["baseline"] == baseline]
            if not rows.empty and "std" in rows.columns:
                return float(rows["std"].mean())
        except Exception:
            pass

    return None


def fig_carbon_sweep(sweep_df: pd.DataFrame) -> None:
    """v1.3 headline figure: aware-vs-lagged saving + temporal-arbitrage collapse.

    Two panels share an x-axis (carbon price):

    Panel A — composition-awareness value: aware vs lagged-blind baseline.
              The lagged baseline is the realistic operator assumption ("I know
              last cargo's composition but not today's blend"). Per-year lines
              are drawn faintly; the cross-year mean is bold. If a fresh
              ``seed_sensitivity_summary.csv`` exists in results/tables/, a
              ±2σ seed-noise band is overlaid as a horizontal shaded region,
              giving the reader the threshold above which the saving is
              statistically meaningful.

    Panel B — temporal-arbitrage value: aware vs constant-flow baseline. This
              collapses from ~16% at zero carbon to <1% at any positive
              carbon price, because emissions are pinned by the demand
              constraint and dominate the cost — the surprising finding that
              motivates the composition story.

    ``sweep_df`` must have columns: price_co2_eur_per_t, year,
    aware, horizon, lagged, annual, constant, saving_vs_lagged_pct.
    """
    sweep = sweep_df.copy()
    sweep["saving_vs_constant_pct"] = (
        (sweep["constant"] - sweep["aware"]) / sweep["constant"] * 100
    )

    years = sorted(sweep["year"].unique())
    # Quiet per-year context lines; one warm accent for the bold mean.
    year_palette = sns.color_palette("Blues", n_colors=len(years) + 1)[1:]
    mean_color = "#b2182b"  # warm red, stands out against the blue year lines

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6), sharex=True)

    def _draw_year_context(ax: plt.Axes, ycol: str) -> None:
        """Thin, quiet single-seed per-year lines — context, not the message."""
        for year, color in zip(years, year_palette):
            sub = sweep[sweep["year"] == year].sort_values("price_co2_eur_per_t")
            ax.plot(
                sub["price_co2_eur_per_t"], sub[ycol],
                linewidth=1.0, alpha=0.5, color=color, zorder=3,
                label=f"{year} (single seed)",
            )

    # ---- Panel A: aware vs lagged ------------------------------------------
    # The message is the bold cross-year mean line; the only uncertainty shown
    # is ONE 95%-CI whisker on the seed-mean at €80 (the paper's claim). Per-year
    # spreads and the wider single-seed prediction interval live in the per-year
    # companion figures, keeping this panel uncluttered.
    _draw_year_context(ax1, "saving_vs_lagged_pct")
    mean_lagged = sweep.groupby("price_co2_eur_per_t")["saving_vs_lagged_pct"].mean()
    ax1.plot(
        mean_lagged.index, mean_lagged.values,
        color=mean_color, linewidth=2.6, marker="s", markersize=5,
        label="3-yr mean (single seed)", zorder=10,
    )

    meanse = _seed_mean_se_per_year("lagged")
    if meanse is not None:
        # Pool the per-year seed means/SEs into the headline cross-year CI.
        ymeans = list(meanse[0].values())
        yses = list(meanse[1].values())
        if ymeans:
            pooled_mean = float(np.mean(ymeans))
            # SE of the average of independent per-year means.
            pooled_se = float(np.sqrt(np.mean(np.square(yses))) / np.sqrt(len(yses)))
            ax1.errorbar(
                80.0, pooled_mean, yerr=2 * pooled_se,
                fmt="o", color="black", markersize=7, ecolor="black",
                elinewidth=2.2, capsize=5, capthick=2.0, zorder=12,
                label="10-seed mean ±2·SE (95% CI)",
            )
            ax1.annotate(
                f"{pooled_mean:+.1f}% ± {2*pooled_se:.1f}",
                xy=(80.0, pooled_mean), xytext=(8, 10),
                textcoords="offset points", fontsize=9, color="black",
                fontweight="bold",
            )

    if 80.0 in mean_lagged.index:
        ax1.axvline(80.0, color="grey", linestyle="--", alpha=0.7, linewidth=1.0)
    ax1.axhline(0.0, color="grey", linewidth=0.8)
    ax1.set_xlabel("Carbon price (EUR / tCO$_2$)")
    ax1.set_ylabel("Saving vs lagged baseline (%)")
    ax1.set_title("A. Composition-awareness value")
    ax1.legend(loc="upper left", fontsize=8, framealpha=0.9)

    # ---- Panel B: aware vs constant (the surprising collapse) --------------
    _draw_year_context(ax2, "saving_vs_constant_pct")
    mean_const = sweep.groupby("price_co2_eur_per_t")["saving_vs_constant_pct"].mean()
    ax2.plot(
        mean_const.index, mean_const.values,
        color=mean_color, linewidth=2.6, marker="s", markersize=5,
        label="3-yr mean", zorder=10,
    )
    if 80.0 in mean_const.index:
        ax2.axvline(80.0, color="grey", linestyle="--", alpha=0.7, linewidth=1.0)
    ax2.axhline(0.0, color="grey", linewidth=0.8)
    ax2.set_xlabel("Carbon price (EUR / tCO$_2$)")
    ax2.set_ylabel("Saving vs constant-flow baseline (%)")
    ax2.set_title("B. Temporal-arbitrage value")
    ax2.legend(loc="best", fontsize=8, framealpha=0.9)

    # Shared EU-ETS annotation in figure coordinates so both panels reference
    # the same x position without overlapping panel content.
    fig.text(
        0.5, 0.965, r"Vertical dashed line: current EU ETS (~€80/tCO$_2$)",
        ha="center", va="top", fontsize=9, color="firebrick",
    )

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG_DIR / "fig6_carbon_sweep.pdf")
    plt.close(fig)


def fig_carbon_sweep_per_year(sweep_df: pd.DataFrame) -> list[Path]:
    """Per-year companion figures to fig6 — one PDF per year.

    Each figure has the same two-panel layout as fig6 but shows only one
    year's data, the ★ 10-seed mean ± 2σ marker at €80, and no 3-year mean
    line. Useful for the paper's per-year discussion sections where the
    reader needs to focus on a single year without the others overlapping.

    Saves results/figures/fig6_carbon_sweep_<year>.pdf for each year in
    ``sweep_df``. Returns the list of paths written.
    """
    sweep = sweep_df.copy()
    if "saving_vs_constant_pct" not in sweep.columns:
        sweep["saving_vs_constant_pct"] = (
            (sweep["constant"] - sweep["aware"]) / sweep["constant"] * 100
        )

    seed_stats = _seed_stats_per_year("lagged")
    per_year_mean = seed_stats[0] if seed_stats else {}
    per_year_std = seed_stats[1] if seed_stats else {}
    meanse = _seed_mean_se_per_year("lagged")
    per_year_se = meanse[1] if meanse else {}

    years = sorted(sweep["year"].unique())
    line_color = "#2166ac"  # single muted blue for the single-seed sweep line
    mean_color = "#b2182b"  # warm red for the 10-seed mean marker

    written: list[Path] = []
    for year in years:
        sub = sweep[sweep["year"] == year].sort_values("price_co2_eur_per_t")
        if sub.empty:
            continue

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2), sharex=True)

        # Panel A — aware vs lagged. Single-seed sweep line + the 10-seed mean
        # at €80 with a bold ±2·SE (95% CI) whisker and a light ±2σ
        # (single-seed prediction) whisker, so one focused year reads cleanly.
        ax1.plot(
            sub["price_co2_eur_per_t"], sub["saving_vs_lagged_pct"],
            marker="o", markersize=4, linewidth=1.6, color=line_color,
            label=f"{year} (single seed)",
        )
        if year in per_year_std:
            ax1.errorbar(
                80.0, per_year_mean[year], yerr=2 * per_year_std[year],
                fmt="none", ecolor=mean_color, elinewidth=1.2,
                capsize=6, capthick=1.2, alpha=0.4, zorder=8,
            )
        if year in per_year_se:
            ax1.errorbar(
                80.0, per_year_mean[year], yerr=2 * per_year_se[year],
                fmt="o", color=mean_color, markersize=7, ecolor=mean_color,
                elinewidth=2.4, capsize=4, capthick=2.0, zorder=9,
                label=(
                    f"10-seed mean {per_year_mean[year]:+.2f}% "
                    f"(95% CI ±{2*per_year_se[year]:.2f}, ±2σ ±{2*per_year_std[year]:.2f})"
                ),
            )
        if 80.0 in sub["price_co2_eur_per_t"].values:
            ax1.axvline(80.0, color="grey", linestyle="--", alpha=0.7, linewidth=1.0)
        ax1.axhline(0.0, color="grey", linewidth=0.8)
        ax1.set_xlabel("Carbon price (EUR / tCO$_2$)")
        ax1.set_ylabel("Saving vs lagged baseline (%)")
        ax1.set_title(f"{year} — A. Composition-awareness value")
        ax1.legend(loc="best", fontsize=8)

        # Panel B — aware vs constant-flow
        ax2.plot(
            sub["price_co2_eur_per_t"], sub["saving_vs_constant_pct"],
            marker="o", markersize=4, linewidth=1.6, color=line_color, label=str(year),
        )
        if 80.0 in sub["price_co2_eur_per_t"].values:
            ax2.axvline(80.0, color="grey", linestyle="--", alpha=0.7, linewidth=1.0)
        ax2.axhline(0.0, color="grey", linewidth=0.8)
        ax2.set_xlabel("Carbon price (EUR / tCO$_2$)")
        ax2.set_ylabel("Saving vs constant-flow baseline (%)")
        ax2.set_title(f"{year} — B. Temporal-arbitrage value")
        ax2.legend(loc="best", fontsize=8)

        fig.text(
            0.5, 0.96, r"Vertical dashed line: current EU ETS (~€80/tCO$_2$)",
            ha="center", va="top", fontsize=9, color="firebrick",
        )
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        out_path = FIG_DIR / f"fig6_carbon_sweep_{year}.pdf"
        fig.savefig(out_path)
        plt.close(fig)
        written.append(out_path)

    return written


def fig_foresight_gap(cascade_df: pd.DataFrame) -> None:
    """v1.4 A — savings cascade: constant -> lagged -> aware -> oracle.

    ``cascade_df`` has one row per strategy with columns:
        strategy (constant|lagged|aware|oracle), saving_vs_lagged_pct,
        and optionally year. If a ``year`` column is present the bars are
        grouped by year; otherwise a single pooled set of bars is drawn.

    The figure frames the headline as "aware captures X% of the
    perfect-foresight ceiling" by placing aware between the realistic lagged
    baseline (0% reference) and the oracle ceiling.
    """
    order = ["lagged", "aware", "oracle"]
    labels = {
        "lagged": "lagged\n(realistic baseline)",
        "aware": "aware\n(7-day foresight)",
        "oracle": "oracle\n(perfect foresight)",
    }
    df = cascade_df[cascade_df["strategy"].isin(order)].copy()
    df["strategy"] = pd.Categorical(df["strategy"], categories=order, ordered=True)

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    if "year" in df.columns and df["year"].nunique() > 1:
        years = sorted(df["year"].unique())
        palette = sns.color_palette("muted", n_colors=len(years))
        width = 0.8 / len(years)
        x = range(len(order))
        for i, (year, color) in enumerate(zip(years, palette)):
            sub = df[df["year"] == year].set_index("strategy").reindex(order)
            offs = [xi + (i - (len(years) - 1) / 2) * width for xi in x]
            ax.bar(offs, sub["saving_vs_lagged_pct"].values, width=width,
                   color=color, label=str(year))
        ax.set_xticks(list(x))
    else:
        pooled = df.groupby("strategy", observed=True)["saving_vs_lagged_pct"].mean().reindex(order)
        ax.bar(range(len(order)), pooled.values, color=sns.color_palette("muted")[:len(order)])
        ax.set_xticks(range(len(order)))
    ax.set_xticklabels([labels[s] for s in order])
    ax.axhline(0.0, color="grey", linewidth=0.8)
    ax.set_ylabel("Cost saving vs lagged baseline (%)")
    ax.set_title("Perfect-foresight cascade (true cost, €80/tCO$_2$)")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="best", fontsize=8, title="year")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig7_foresight_gap.pdf")
    plt.close(fig)


def fig_volatility_vs_saving(diag_df: pd.DataFrame) -> None:
    """v1.4 D — per-window price volatility vs realised aware-vs-lagged saving.

    Explains the weak 2022 (energy-crisis) result: high-volatility windows
    cluster at low/negative saving because the optimiser chases price
    arbitrage, swamping the carbon-composition signal.

    ``diag_df`` columns: price_volatility, saving_eur (or saving_pct), year.
    """
    y_col = "saving_pct" if "saving_pct" in diag_df.columns else "saving_eur"
    years = sorted(diag_df["year"].unique())
    palette = sns.color_palette("muted", n_colors=len(years))
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for year, color in zip(years, palette):
        sub = diag_df[diag_df["year"] == year]
        ax.scatter(sub["price_volatility"], sub[y_col], alpha=0.6, color=color,
                   label=str(year), s=22)
    overall_corr = diag_df[["price_volatility", y_col]].corr().iloc[0, 1]
    ax.axhline(0.0, color="grey", linewidth=0.8)
    ax.set_xlabel("Per-window price volatility (EUR/MWh)")
    ax.set_ylabel("Aware-vs-lagged saving "
                  + ("(%)" if y_col == "saving_pct" else "(EUR)"))
    ax.set_title(f"Saving vs price volatility  (r = {overall_corr:.2f})")
    ax.legend(loc="best", fontsize=8, title="year")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig8_volatility_vs_saving.pdf")
    plt.close(fig)


def fig_cost_delta(
    aware_df: pd.DataFrame,
    blind_df: pd.DataFrame,
) -> None:
    """Bar chart of cost savings (aware vs blind) by month."""
    delta = blind_df["cost_eur"] - aware_df["cost_eur"]
    monthly = delta.resample("ME").sum()
    fig, ax = plt.subplots(figsize=(10, 4))
    monthly.plot(kind="bar", ax=ax, color="steelblue")
    ax.set_xlabel("Month")
    ax.set_ylabel("Cost saving (EUR)")
    ax.set_title("Composition-aware vs composition-blind dispatch cost saving")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_cost_delta.pdf")
    plt.close(fig)


def fig_sensitivity(sensitivity_df: pd.DataFrame) -> None:
    """Cost savings vs composition and price-regime metrics."""
    panels = [
        ("variability", "std(CH4 mole fraction)"),
        ("price_volatility", "price volatility (EUR/MWh)"),
        ("price_ch4_corr", "corr(price, CH4)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, (col, label) in zip(axes, panels, strict=True):
        ax.scatter(sensitivity_df[col], sensitivity_df["saving_eur"], alpha=0.65)
        corr = sensitivity_df[[col, "saving_eur"]].corr().iloc[0, 1]
        ax.set_xlabel(label)
        ax.set_title(f"r = {corr:.2f}")
    axes[0].set_ylabel("Total cost saving (EUR)")
    fig.suptitle("Dispatch saving drivers by rolling window")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG_DIR / "fig2_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_load_shift_heatmap(
    aware_df: pd.DataFrame,
    blind_df: pd.DataFrame,
    timeseries_df: pd.DataFrame,
) -> None:
    """Heatmap: (price quantile, methane number) -> load shift."""
    price_q = pd.qcut(timeseries_df["price_eur_mwh"], q=5, labels=False)
    ch4 = timeseries_df["CH4"]
    ch4_q = pd.qcut(ch4, q=5, labels=False)
    shift = aware_df["m_dot"] - blind_df["m_dot"]
    pivot = pd.DataFrame({"price_q": price_q, "ch4_q": ch4_q, "shift": shift})
    table = pivot.groupby(["price_q", "ch4_q"])["shift"].mean().unstack()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(table, annot=True, fmt=".1f", ax=ax, cmap="RdBu_r", center=0)
    ax.set_xlabel("CH4 quantile (proxy for methane number)")
    ax.set_ylabel("Price quantile")
    ax.set_title("Load shift: aware - blind (kg/s)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_load_shift.pdf")
    plt.close(fig)


def fig_surrogate_eval(eval_df: pd.DataFrame) -> None:
    """Bar chart of per-channel MAE, RMSE, R² from the held-out test split."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, metric in zip(axes, ["MAE", "RMSE", "R2"]):
        ax.bar(eval_df["channel"], eval_df[metric], color="steelblue")
        ax.set_title(metric)
        ax.set_xlabel("Output channel")
        ax.tick_params(axis="x", rotation=30)
    axes[0].set_ylabel("Error (physical units)")
    axes[2].set_ylabel("R²")
    fig.suptitle("Physics-by-construction surrogate — held-out test set (10%)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_surrogate_eval.pdf")
    plt.close(fig)


def fig_surrogate_fidelity(fidelity_df: pd.DataFrame) -> None:
    """PINN-predicted cost error vs CoolProp ground truth."""
    err_pct = fidelity_df["pinn_cost_eur"] / fidelity_df["true_cost_eur"] - 1.0
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(err_pct * 100, bins=40, color="steelblue", edgecolor="white")
    ax.axvline(1.0, color="red", linestyle="--", label="1% threshold")
    ax.axvline(-1.0, color="red", linestyle="--")
    ax.set_xlabel("Cost prediction error (%)")
    ax.set_ylabel("Count")
    ax.set_title("Physics-by-construction surrogate fidelity vs CoolProp ground truth")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_fidelity.pdf")
    plt.close(fig)
