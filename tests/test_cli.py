"""
Tests for SentinelPXT CLI interface and execution modes.
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_cli_help():
    """Verifies that python src/main.py --help exits with code 0."""
    res = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "src" / "main.py"), "--help"],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    assert res.returncode == 0
    assert "SentinelPXT" in res.stdout
    assert "--mode" in res.stdout


def test_cli_demo_mode():
    """Verifies that python src/main.py --mode demo runs end-to-end and outputs OSHA report."""
    res = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "src" / "main.py"), "--mode", "demo"],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    assert res.returncode == 0
    assert "OSHA FORM 301 & INCIDENT FORENSIC TRIAGE REPORT" in res.stdout
    assert "Temporal Localization Window" in res.stdout
    assert "dock_bay_4" in res.stdout


def test_cli_investigate_mode():
    """Verifies that python src/main.py --mode investigate works with custom query and zone."""
    res = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "src" / "main.py"),
            "--mode",
            "investigate",
            "--query",
            "smoke plume near solvent tank",
            "--zone",
            "chemical_storage",
        ],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    assert res.returncode == 0
    assert "OSHA FORM 301 & INCIDENT FORENSIC TRIAGE REPORT" in res.stdout
    assert "chemical_storage" in res.stdout

