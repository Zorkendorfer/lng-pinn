"""CLI coverage for experiment script switches used in the rework plan."""

from __future__ import annotations

import subprocess
import sys


def test_train_script_accepts_soft_arch_flag() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/03_train_pinn.py",
            "--help",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--arch" in result.stdout
    assert "{hard,soft}" in result.stdout
    assert "--fresh" in result.stdout
    assert "--out" in result.stdout


def test_seed_sensitivity_accepts_model_path_flag() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/06_seed_sensitivity.py",
            "--help",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--model-path" in result.stdout
    assert "--surrogate" in result.stdout


def test_soft_vs_hard_script_has_help() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/12_soft_vs_hard.py",
            "--help",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--results-dir" in result.stdout
    assert "--out" in result.stdout
