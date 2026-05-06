"""All paper figures generated here — one function per figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

FIG_DIR = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")


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
    """Cost savings vs composition variability metric."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(sensitivity_df["variability"], sensitivity_df["saving_eur"], alpha=0.6)
    ax.set_xlabel("Composition variability (std of CH4 mole fraction)")
    ax.set_ylabel("Total cost saving (EUR)")
    ax.set_title("Cost saving vs LNG compositional variability")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_sensitivity.pdf")
    plt.close(fig)


def fig_load_shift_heatmap(
    aware_df: pd.DataFrame,
    blind_df: pd.DataFrame,
    timeseries_df: pd.DataFrame,
) -> None:
    """Heatmap: (price quantile, methane number) → load shift."""
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
    ax.set_title("Load shift: aware − blind (kg/s)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_load_shift.pdf")
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
