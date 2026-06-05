"""CLI coverage for experiment script switches used in the rework plan."""

from __future__ import annotations

import subprocess
import sys


def _help(script: str) -> str:
    result = subprocess.run(
        [sys.executable, script, "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def test_train_script_accepts_soft_arch_flag() -> None:
    stdout = _help("scripts/03_train_pinn.py")
    assert "--arch" in stdout
    assert "{hard,soft}" in stdout
    assert "--fresh" in stdout
    assert "--out" in stdout


def test_seed_sensitivity_accepts_model_path_flag() -> None:
    stdout = _help("scripts/06_seed_sensitivity.py")
    assert "--model-path" in stdout
    assert "--surrogate" in stdout
    assert "--seeds" in stdout
    assert "--composition-csv" in stdout


def test_soft_vs_hard_script_has_help() -> None:
    stdout = _help("scripts/12_soft_vs_hard.py")
    assert "--results-dir" in stdout
    assert "--strict" in stdout
    assert "--out" in stdout


def test_postprocess_scaffold_scripts_have_help() -> None:
    expected = {
        "scripts/13_carbon_ensemble.py": ["--prices", "--strict", "--composition-tag"],
        "scripts/14_mixing_table.py": ["--cells-cache", "--expected-n", "--strict"],
        "scripts/15_run_manifest.py": ["--expected", "--strict", "--out"],
    }
    for script, flags in expected.items():
        stdout = _help(script)
        for flag in flags:
            assert flag in stdout
