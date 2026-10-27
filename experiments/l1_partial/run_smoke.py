"""Drive setup then train under a 30-minute wall clock. PARTIAL_REPRODUCTION is not assumed."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIMIT_S = 30 * 60


def _run(mode: str, hydra_dir: str, log_name: str, deadline: float) -> dict:
    remaining = deadline - time.time()
    log_path = HERE / log_name
    started = time.time()
    if remaining <= 1:
        log_path.write_text("not started: wall-clock budget already spent\n", encoding="utf-8")
        return {"mode": mode, "exit_code": None, "timed_out": True, "seconds": 0.0, "log": log_name}
    command = [
        sys.executable,
        str(HERE / "launch_partial.py"),
        mode,
        "hydra.run.dir=" + hydra_dir,
    ]
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(command, cwd=HERE, stdout=handle, stderr=subprocess.STDOUT)
        try:
            exit_code = process.wait(timeout=remaining)
            timed_out = False
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=30)
            exit_code = None
            timed_out = True
            handle.write("\nWALL_CLOCK_CAP: process killed at 30 minutes\n")
    return {
        "mode": mode,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "seconds": round(time.time() - started, 3),
        "log": log_name,
        "command": command,
    }


def main() -> int:
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    t0 = time.time()
    deadline = t0 + LIMIT_S
    preflight = subprocess.run(
        [sys.executable, str(HERE / "launch_partial.py"), "preflight"],
        cwd=HERE,
        capture_output=True,
        text=True,
        timeout=180,
    )
    (HERE / "preflight.log").write_text(preflight.stdout + "\n" + preflight.stderr, encoding="utf-8")
    if preflight.returncode != 0:
        payload = {
            "label": "BLOCKED",
            "reason": "preflight failed before setup",
            "started": started,
            "preflight_exit": preflight.returncode,
        }
        (HERE / "RUN_STATUS.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return preflight.returncode or 1
    setup = _run(
        "setup",
        "outputs/unconstrained_qp/data_setup_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}",
        "setup.log",
        deadline,
    )
    train = None
    if setup["exit_code"] == 0 and not setup["timed_out"]:
        train = _run(
            "train",
            "outputs/unconstrained_qp/train_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}",
            "train.log",
            deadline,
        )
    finished = time.strftime("%Y-%m-%dT%H:%M:%S")
    elapsed = round(time.time() - t0, 3)
    if setup.get("timed_out") or (train and train.get("timed_out")):
        label = "FAILED"
        reason = "wall-clock cap of 30 minutes"
    elif setup["exit_code"] != 0:
        label = "FAILED"
        reason = "setup exited nonzero"
    elif train is None or train["exit_code"] != 0:
        label = "FAILED"
        reason = "train did not finish with exit code 0"
    else:
        label = "PARTIAL_REPRODUCTION"
        reason = "setup and the capped train both exited 0"
    payload = {
        "label": label,
        "reason": reason,
        "started": started,
        "finished": finished,
        "elapsed_seconds": elapsed,
        "limit_seconds": LIMIT_S,
        "preflight_exit": preflight.returncode,
        "setup": setup,
        "train": train,
    }
    (HERE / "RUN_STATUS.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if label == "PARTIAL_REPRODUCTION" else 1


if __name__ == "__main__":
    raise SystemExit(main())
