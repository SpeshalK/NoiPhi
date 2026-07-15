"""
test_demos.py

Integration / smoke tests that ensure every demonstration script under
demos/ still runs end-to-end against the current codebase, exactly the
way a user would run it.

Each demo is executed as a standalone subprocess from its own directory,
because the demos themselves rely on this:
    - They load experimental data via paths relative to their own folder
      (e.g. '../data/795nm_freqNoise_red.csv').
    - many_body_Ising_grEvo.py does a sibling import
      (`from main_Rydbuild import buildRydHamil`), which only resolves
      when the script's own directory is on sys.path — exactly what
      Python does automatically when a script is invoked directly.

A demo is only considered "passing" if the interpreter exits with status
0; a non-zero exit (uncaught exception, etc.) fails the test and the
captured stdout/stderr is surfaced for debugging.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

# ── Discovery ───────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
DEMOS_DIR = REPO_ROOT / "demos"

# Generous enough for the heaviest physics demo (many-body ensemble evolution,
# ~30s observed locally) while still failing well before a CI job timeout if
# a script genuinely hangs.
DEMO_TIMEOUT_SECONDS = 240


def _is_noiphi_demo(path: Path) -> bool:
    """
    A .py file under demos/ counts as a runnable "demo" if it directly
    exercises the noiphi package.

    This excludes supporting library modules that live alongside the demos
    (e.g. demos/physics/main_Rydbuild.py, a Hamiltonian-builder helper that
    many_body_Ising_grEvo.py imports but which never touches noiphi itself
    and produces no output on its own) without needing a hard-coded
    exclude list that would silently go stale as demos are added.
    """
    text = path.read_text(encoding="utf-8")
    return "import noiphi" in text or "from noiphi" in text


def _discover_demo_scripts():
    if not DEMOS_DIR.exists():
        return []
    return sorted(p for p in DEMOS_DIR.rglob("*.py") if _is_noiphi_demo(p))


DEMO_SCRIPTS = _discover_demo_scripts()
DEMO_IDS = [str(p.relative_to(DEMOS_DIR)) for p in DEMO_SCRIPTS]

# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def demos_sandbox(tmp_path_factory):
    """
    Copy the whole demos/ tree into an isolated, writable temp directory.

    Some demos save plots into their own folder (e.g.
    'Rabi_Oscill_Ensemble.png'), and all of them read data via relative
    paths. Running out of a throwaway copy keeps the real repository
    clean and prevents test runs from leaving generated artifacts behind.
    """
    sandbox_root = tmp_path_factory.mktemp("demos_sandbox")
    sandboxed_demos = sandbox_root / "demos"
    shutil.copytree(DEMOS_DIR, sandboxed_demos)
    return sandboxed_demos


# ── Smoke tests ─────────────────────────────────────────────────────────────


def test_demo_scripts_were_discovered():
    """
    Sanity check on the discovery mechanism itself: if this ever returns
    zero, the parametrized test below would silently collect nothing
    (rather than fail), masking a real problem with the demos/ layout.
    """
    assert DEMO_SCRIPTS, (
        f"No noiphi demo scripts found under {DEMOS_DIR}. "
        "Either the demos/ folder is missing/empty, or every script "
        "stopped importing noiphi directly."
    )


@pytest.mark.parametrize("script", DEMO_SCRIPTS, ids=DEMO_IDS)
def test_demo_runs_without_error(script, demos_sandbox):
    """Every demo script must execute to completion with exit status 0."""
    relative_path = script.relative_to(DEMOS_DIR)
    sandboxed_script = demos_sandbox / relative_path

    env = os.environ.copy()
    # Force a non-interactive backend so plt.show() never blocks or tries
    # to open a window on headless CI runners.
    env["MPLBACKEND"] = "Agg"

    result = subprocess.run(
        [sys.executable, str(sandboxed_script)],
        cwd=sandboxed_script.parent,
        env=env,
        capture_output=True,
        text=True,
        timeout=DEMO_TIMEOUT_SECONDS,
    )

    assert result.returncode == 0, (
        f"Demo '{relative_path}' exited with status {result.returncode}.\n\n"
        f"--- stdout (tail) ---\n{result.stdout[-2000:]}\n\n"
        f"--- stderr (tail) ---\n{result.stderr[-2000:]}"
    )
