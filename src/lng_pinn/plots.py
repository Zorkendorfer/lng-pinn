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


def fig_carbon_sweep(sweep_df: pd.DataFrame) -> None:
    """v1.3 headline figure: aware-vs-horizon true-cost saving vs CO2 price.

    ``sweep_df`` has columns: price_co2_eur_per_t, year, saving_vs_horizon_pct.
    One line per year + a mean line. Annotates the €80/tCO2 EU-ETS point.
    """
    years = sorted(sweep_df["year"].unique())
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for year in years:
        sub = sweep_df[sweep_df["year"] == year].sort_values("price_co2_eur_per_t")
        ax.plot(
            sub["price_co2_eur_per_t"], sub["saving_vs_horizon_pct"],
            marker="o", linewidth=1.4, label=str(year), alpha=0.65,
        )
    mean = sweep_df.groupby("price_co2_eur_per_t")["saving_vs_horizon_pct"].mean()
    ax.plot(
        mean.index, mean.values,
        color="black", linewidth=2.3, marker="s", label="mean", zorder=10,
    )
    if 80.0 in mean.index:
        ax.axvline(80.0, color="firebrick", linestyle="--", alpha=0.5)
        ax.text(82, ax.get_ylim()[1] * 0.92, "EU ETS\n(~€80/tCO₂)",
                color="firebrick", fontsize=9, va="top")
    ax.axhline(0.0, color="grey", linewidth=0.8)
    ax.set_xlabel("Carbon price (EUR/tCO₂)")
    ax.set_ylabel("Aware vs horizon-blind saving (%)")
    ax.set_title("Composition-aware dispatch saving vs carbon price")
    ax.legend(loc="best", fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig6_carbon_sweep.pdf")
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
    fig.suptitle("Dispatch saving drivers by rolling window", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_sensitivity.pdf")
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
    fig.suptitle("PINN surrogate — held-out test set (10%)")
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
    ax.set_title("PINN surrogate fidelity vs CoolProp ground truth")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_fidelity.pdf")
    plt.close(fig)
