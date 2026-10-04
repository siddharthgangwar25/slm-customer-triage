"""Bootstrap failures must be actionable and preserve previous run evidence."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def reproduction_script(tmp_path):
    project = tmp_path / "project"
    scripts = project / "scripts"
    scripts.mkdir(parents=True)
    target = scripts / "reproduce_cpu.py"
    source = Path(__file__).resolve().parents[1] / "scripts/reproduce_cpu.py"
    shutil.copyfile(source, target)
    return target


@pytest.mark.parametrize("problem", ["outside", "existing", "missing_uv", "missing_source"])
def test_preflight_rejects_invalid_inputs_without_creating_run(reproduction_script, problem):
    project = reproduction_script.parent.parent
    output = project / "artifacts/run"
    command = [sys.executable, str(reproduction_script), "--uv", sys.executable]
    expected = {
        "outside": "--output must be inside the project directory",
        "existing": "Output already exists",
        "missing_uv": "uv executable not found",
        "missing_source": "--source-dir is missing",
    }
    if problem == "outside":
        output = project.parent / "outside"
    elif problem == "existing":
        output.mkdir(parents=True)
        (output / "evidence.json").write_text('{"preserve": true}', encoding="utf-8")
    elif problem == "missing_uv":
        command[-1] = str(project / "missing-uv")
    else:
        command.extend(["--source-dir", str(project / "missing-source")])
    result = subprocess.run(
        [*command, "--output", str(output)], capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 2
    assert expected[problem] in result.stderr
    assert "Traceback" not in result.stderr
    if problem == "existing":
        assert (output / "evidence.json").read_text() == '{"preserve": true}'
        assert len(list(output.iterdir())) == 1
    else:
        assert not output.exists()


def test_failed_stage_keeps_exit_code_and_log(reproduction_script):
    output = reproduction_script.parent.parent / "artifacts/run"
    # Python rejects uv's arguments, causing a real child-process failure.
    result = subprocess.run(
        [sys.executable, str(reproduction_script), "--uv", sys.executable, "--output", str(output)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 1
    assert "sync failed; inspect" in result.stderr
    assert "Traceback" not in result.stderr
    commands = json.loads((output / "commands.json").read_text())
    assert len(commands) == 1
    assert commands[0]["stage"] == "sync"
    assert commands[0]["exit_code"] != 0
    assert (output / "sync.log").stat().st_size > 0
    assert not (output / "bundle").exists()


def test_unstartable_executable_records_failure(reproduction_script):
    project = reproduction_script.parent.parent
    executable = project / "invalid-uv.exe"
    executable.write_bytes(b"not an executable\n")
    executable.chmod(0o755)
    output = project / "artifacts/run"
    result = subprocess.run(
        [
            sys.executable,
            str(reproduction_script),
            "--uv",
            str(executable),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 1
    assert "sync failed; inspect" in result.stderr
    commands = json.loads((output / "commands.json").read_text())
    assert commands[0]["exit_code"] is None
    assert commands[0]["error_type"] == "OSError"
    assert "Could not start sync" in (output / "sync.log").read_text()
