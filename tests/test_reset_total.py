"""Tests for the --reset-total CLI flag and the reset_state helper.

Covers the function directly (fast, no subprocess) and the CLI surface end-to-end
(slow-ish, one subprocess per case) so we catch argparse wiring regressions.
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from watermeter import Config, reset_state


REPO_ROOT = Path(__file__).resolve().parent.parent
WATERMETER_PY = REPO_ROOT / "watermeter.py"


def _cfg_with(tmp_path):
    state_path = str(tmp_path / "state.json")
    log_path = str(tmp_path / "watermeter.log")
    return Config(
        esp32_base_url="http://localhost",
        state_path=state_path,
        log_path=log_path,
    )


class TestResetStateHelper:
    def test_creates_state_on_fresh_path(self, tmp_path):
        cfg = _cfg_with(tmp_path)
        backup, state = reset_state(cfg, 0.0)
        assert backup is None
        assert state["total"] == 0.0
        assert state["ts"] > 0
        assert state["dial_histories"] == {}
        # File was written
        assert os.path.exists(cfg.state_path)
        persisted = json.loads(Path(cfg.state_path).read_text())
        assert persisted == state

    def test_backs_up_existing_state(self, tmp_path):
        cfg = _cfg_with(tmp_path)
        # Seed an existing state.json
        Path(cfg.state_path).write_text(json.dumps(
            {"total": 98765.432, "ts": 1000.0, "dial_histories": {"dial_0": [1, 2, 3]}}
        ))
        backup, state = reset_state(cfg, 0.0)
        assert backup is not None
        assert os.path.exists(backup)
        # Backup content matches the pre-reset state
        backed = json.loads(Path(backup).read_text())
        assert backed["total"] == 98765.432
        assert backed["dial_histories"] == {"dial_0": [1, 2, 3]}
        # Fresh state overwrote the live file
        live = json.loads(Path(cfg.state_path).read_text())
        assert live["total"] == 0.0
        assert live["dial_histories"] == {}

    def test_nonzero_total(self, tmp_path):
        cfg = _cfg_with(tmp_path)
        _, state = reset_state(cfg, 12345.6)
        assert state["total"] == pytest.approx(12345.6)

    def test_negative_total_rejected(self, tmp_path):
        cfg = _cfg_with(tmp_path)
        with pytest.raises(ValueError, match=">= 0"):
            reset_state(cfg, -5.0)
        assert not os.path.exists(cfg.state_path)

    def test_infinite_total_rejected(self, tmp_path):
        cfg = _cfg_with(tmp_path)
        with pytest.raises(ValueError, match="finite"):
            reset_state(cfg, float("inf"))

    def test_nan_total_rejected(self, tmp_path):
        cfg = _cfg_with(tmp_path)
        with pytest.raises(ValueError, match="finite"):
            reset_state(cfg, float("nan"))


def _write_minimal_config(tmp_path):
    """A minimal config.yaml pointing state_path at tmp_path."""
    state_path = tmp_path / "state.json"
    log_path = tmp_path / "watermeter.log"
    p = tmp_path / "config.yaml"
    p.write_text(
        "esp32:\n"
        "  base_url: \"http://localhost\"\n"
        "paths:\n"
        f"  state_path: \"{state_path}\"\n"
        f"  log_path: \"{log_path}\"\n"
    )
    return p, state_path


class TestResetTotalCli:
    """End-to-end: invoke watermeter.py as a subprocess so argparse wiring is exercised."""

    def _invoke(self, args, tmp_path, env=None):
        cmd = [sys.executable, str(WATERMETER_PY)] + args
        return subprocess.run(
            cmd, cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=15
        )

    def test_flag_no_value_resets_to_zero(self, tmp_path):
        config_path, state_path = _write_minimal_config(tmp_path)
        result = self._invoke(["--config", str(config_path), "--reset-total"], tmp_path)
        assert result.returncode == 0, result.stderr
        assert state_path.exists()
        state = json.loads(state_path.read_text())
        assert state["total"] == 0.0
        assert state["dial_histories"] == {}

    def test_flag_with_value(self, tmp_path):
        config_path, state_path = _write_minimal_config(tmp_path)
        result = self._invoke(
            ["--config", str(config_path), "--reset-total", "54321.1"], tmp_path
        )
        assert result.returncode == 0, result.stderr
        state = json.loads(state_path.read_text())
        assert state["total"] == pytest.approx(54321.1)

    def test_flag_negative_exits_nonzero(self, tmp_path):
        config_path, state_path = _write_minimal_config(tmp_path)
        result = self._invoke(
            ["--config", str(config_path), "--reset-total", "-1"], tmp_path
        )
        assert result.returncode != 0
        assert "error" in result.stderr.lower() or ">= 0" in result.stderr
        assert not state_path.exists()

    def test_flag_creates_backup_of_existing_state(self, tmp_path):
        config_path, state_path = _write_minimal_config(tmp_path)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(
            {"total": 99999.0, "ts": 1.0, "dial_histories": {"dial_0": [5.0]}}
        ))
        result = self._invoke(
            ["--config", str(config_path), "--reset-total", "0"], tmp_path
        )
        assert result.returncode == 0, result.stderr
        backups = glob.glob(f"{state_path}.bak.*")
        assert len(backups) == 1
        backed = json.loads(Path(backups[0]).read_text())
        assert backed["total"] == 99999.0

    def test_flag_does_not_start_main_loop(self, tmp_path):
        """If the reset branch worked, the subprocess must exit quickly —
        no infinite capture loop, no network calls."""
        config_path, _ = _write_minimal_config(tmp_path)
        # A 15s subprocess timeout would mask this; we rely on returncode == 0
        # happening well before that because the reset path bypasses the main loop.
        result = self._invoke(["--config", str(config_path), "--reset-total"], tmp_path)
        assert result.returncode == 0
