"""release.sh: staged install, health gates, keep-previous and rollback.

Drives the bash functions with a stub launchctl and stub binaries in a temp
APP_ROOT; nothing touches the real LaunchAgent.
"""
from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
RELEASE = REPO / "release.sh"


def _exe(path: Path, body: str):
    path.write_text("#!/bin/sh\n" + body + "\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


@pytest.fixture
def env(tmp_path):
    root = tmp_path / "app"
    bindir = root / "bin"
    bindir.mkdir(parents=True)
    (root / "config.yaml").write_text("esp32:\n  base_url: http://192.0.2.1\n")
    _exe(bindir / "watermeter", "echo old")
    _exe(bindir / "ocr", "echo old-ocr")
    state = tmp_path / "launchd"
    state.mkdir()
    (state / "pid").write_text("100")
    (state / "mode").write_text("stable")
    # Stub launchctl: logs calls; `print` reports running with a pid that is
    # stable, or that changes on every call ("crashloop"), or no job ("down").
    launchctl = _exe(tmp_path / "launchctl", f"""
S={state}
echo "$*" >> "$S/calls"
case "$1" in
  print)
    mode=$(cat "$S/mode")
    [ "$mode" = down ] && exit 113
    pid=$(cat "$S/pid")
    if [ "$mode" = crashloop ]; then echo $((pid + 1)) > "$S/pid"; fi
    printf '\\tstate = running\\n\\tpid = %s\\n\\t\\tstate = active\\n' "$pid"
    ;;
esac
exit 0""")
    e = dict(os.environ, APP_ROOT=str(root), BIN_DIR=str(bindir),
             PLIST=str(tmp_path / "com.watermeter.ocr.plist"), LAUNCHCTL=str(launchctl),
             HEALTH_WAIT_SEC="1", HEALTH_POLL_SEC="0.2")
    new = tmp_path / "new"
    new.mkdir()
    return {"env": e, "bin": bindir, "state": state, "new": new, "tmp": tmp_path}


def _new_build(env, check_exit=0):
    """Stub new binaries; --check-config exits with check_exit."""
    nb = _exe(env["new"] / "watermeter",
              f'[ "$1" = --check-config ] && exit {check_exit}\necho new')
    no = _exe(env["new"] / "ocr", "echo new-ocr")
    return nb, no


def _run(env, script):
    return subprocess.run(["bash", "-c", f"set -euo pipefail; source {RELEASE}; {script}"],
                          env=env["env"], capture_output=True, text=True, timeout=60)


def _calls(env):
    p = env["state"] / "calls"
    return p.read_text().splitlines() if p.exists() else []


def test_gate1_failure_leaves_the_live_release_alone(env):
    nb, no = _new_build(env, check_exit=78)
    r = _run(env, f"stage_release {nb} {no}")
    assert r.returncode == 1
    assert "failed --check-config" in r.stdout
    assert "echo old" in (env["bin"] / "watermeter").read_text()
    assert not (env["bin"] / ".staged").exists()
    assert _calls(env) == []


def test_successful_release_keeps_previous(env):
    nb, no = _new_build(env)
    r = _run(env, f"stage_release {nb} {no} && activate_release")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "echo new" in (env["bin"] / "watermeter").read_text()
    assert "echo new-ocr" in (env["bin"] / "ocr").read_text()
    assert "echo old" in (env["bin"] / "watermeter.prev").read_text()
    assert "echo old-ocr" in (env["bin"] / "ocr.prev").read_text()
    assert not (env["bin"] / ".staged").exists()
    calls = _calls(env)
    assert any(c.startswith("bootout") for c in calls)
    assert any(c.startswith("bootstrap") for c in calls)
    assert "Release is healthy" in r.stdout


def test_crash_loop_rolls_back(env):
    nb, no = _new_build(env)
    (env["state"] / "mode").write_text("crashloop")
    # The rollback's own health check also sees the crash loop; what matters is
    # that the previous binaries are back in place and the new ones kept aside.
    r = _run(env, f"stage_release {nb} {no} && activate_release")
    assert r.returncode == 1
    assert "pid changed" in r.stdout and "rolling back" in r.stdout
    assert "echo old" in (env["bin"] / "watermeter").read_text()
    assert "echo old-ocr" in (env["bin"] / "ocr").read_text()
    assert "echo new" in (env["bin"] / "watermeter.rolled-back").read_text()


def test_service_that_never_starts_rolls_back(env):
    nb, no = _new_build(env)
    (env["state"] / "mode").write_text("down")
    r = _run(env, f"stage_release {nb} {no} && activate_release")
    assert r.returncode == 1
    assert "never reached state=running" in r.stdout
    assert "echo old" in (env["bin"] / "watermeter").read_text()


def test_first_install_without_previous_reports_it(env):
    for f in ("watermeter", "ocr"):
        (env["bin"] / f).unlink()
    nb, no = _new_build(env)
    (env["state"] / "mode").write_text("down")
    r = _run(env, f"stage_release {nb} {no} && activate_release")
    assert r.returncode == 1
    assert "no previous release to restore" in r.stdout


def test_manual_rollback_command(env):
    nb, no = _new_build(env)
    assert _run(env, f"stage_release {nb} {no} && activate_release").returncode == 0
    r = subprocess.run(["bash", str(RELEASE), "rollback"], env=env["env"],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "echo old" in (env["bin"] / "watermeter").read_text()
    assert "echo new" in (env["bin"] / "watermeter.rolled-back").read_text()


def test_rollback_without_previous_fails(env):
    r = subprocess.run(["bash", str(RELEASE), "rollback"], env=env["env"],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 1
    assert "no previous release" in r.stdout


def test_status_command(env):
    r = subprocess.run(["bash", str(RELEASE), "status"], env=env["env"],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and "running as pid 100" in r.stdout


def test_install_sh_uses_the_transaction():
    text = (REPO / "install.sh").read_text()
    assert 'source "$REPO_DIR/release.sh"' in text
    assert 'stage_release "$BUILD_TMP/dist/watermeter" "$BUILD_TMP/ocr"' in text
    assert "activate_release" in text
    # The old direct install into the live bin/ is gone.
    assert 'install -m 0755 "$BUILD_TMP/dist/watermeter" "$BIN_DIR/watermeter"' not in text
    assert '-o "$BIN_DIR/ocr"' not in text
