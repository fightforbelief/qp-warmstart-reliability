"""Command line. Every scientific command is meant to run on hanqing-win."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from . import __version__
from .manifest import HoldoutLocked, assert_holdout_released, file_sha256, protocol_sha
from .sources import L2WS_SHA, OSQP_BENCHMARK_SHA, load_provenance


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def cmd_env_audit(args: argparse.Namespace) -> int:
    root = project_root()
    lines = [
        "# Remote environment audit",
        "",
        f"- Package version: `{__version__}`",
        f"- Hostname: `{platform.node()}`",
        f"- Platform: `{platform.platform()}`",
        f"- Python: `{sys.version.replace(chr(10), ' ')}`",
        f"- Executable: `{sys.executable}`",
        f"- Project root: `{root}`",
        "",
        "## Imports in this interpreter",
        "",
    ]
    for name in ("numpy", "scipy", "osqp", "cvxpy", "pandas", "matplotlib", "yaml", "jax", "torch"):
        try:
            mod = __import__(name if name != "yaml" else "yaml")
            ver = getattr(mod, "__version__", "unknown")
            extra = ""
            if name == "jax":
                extra = f" devices={list(mod.devices())}"
            if name == "torch":
                extra = f" cuda={mod.cuda.is_available()}"
            lines.append(f"- {name}: `{ver}`{extra}")
        except Exception as exc:
            lines.append(f"- {name}: NOT IMPORTED ({type(exc).__name__}: {exc})")
    lines += ["", "## nvidia-smi", "", "```"]
    exe = shutil.which("nvidia-smi")
    if exe is None:
        lines.append("nvidia-smi not on PATH")
    else:
        try:
            proc = subprocess.run([exe, "-L"], capture_output=True, text=True, timeout=30)
            lines.append(proc.stdout.strip() or proc.stderr.strip())
            lines.append(f"exit={proc.returncode}")
        except Exception as exc:
            lines.append(f"nvidia-smi failed: {exc}")
    lines.append("```")
    lines += [
        "",
        "## What this audit does not claim",
        "",
        "- A CUDA driver version does not mean legacy JAX 0.4.20 can use the GPU.",
        "- RTX 50-series (Blackwell) support is not assumed for 2023 JAX wheels.",
        "- The author reproduction environment is separate from this eval interpreter.",
        "",
    ]
    _write(Path(args.output), "\n".join(lines))
    print(args.output)
    return 0


def cmd_source_audit(args: argparse.Namespace) -> int:
    root = project_root()
    prov = load_provenance(root)
    l2ws = root / "external" / "l2ws"
    l2ws_sha = "MISSING"
    dirty = "missing"
    if (l2ws / ".git").exists():
        l2ws_sha = subprocess.check_output(["git", "-C", str(l2ws), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(l2ws), "status", "--porcelain"], text=True).strip()
    mpc_yaml = (l2ws / "benchmarks" / "configs" / "mpc" / "mpc_run.yaml").read_text(encoding="utf-8")
    conflict = "<<<<<<<" in mpc_yaml
    lines = [
        "# Source and data audit",
        "",
        "## OSQP benchmarks",
        "",
        f"- Requested commit: `{OSQP_BENCHMARK_SHA}`",
        f"- Provenance file commit: `{prov.get('commit')}`",
        f"- Retrieval: {prov.get('retrieval')}",
        f"- Full git clone: {prov.get('git_clone')}",
        f"- License: {prov.get('license')}",
        "",
        "File SHA256:",
        "",
    ]
    for rel, digest in prov.get("files", {}).items():
        disk = file_sha256(root / "external" / "osqp_benchmarks" / rel)
        flag = "OK" if disk.lower() == digest.lower() else "MISMATCH"
        lines.append(f"- `{rel}` recorded `{digest}` disk `{disk}` {flag}")
    lines += [
        "",
        "## L2WS",
        "",
        f"- Expected HEAD: `{L2WS_SHA}`",
        f"- git rev-parse HEAD: `{l2ws_sha}`",
        f"- porcelain: `{dirty or 'clean'}`",
        f"- pyproject license text: Apache 2.0",
        "- LICENSE file in the clone: ABSENT. Metadata claims Apache 2.0; the file itself was not in the tree.",
        f"- `benchmarks/configs/mpc/mpc_run.yaml` contains unresolved conflict markers: {conflict}",
        "",
        "## Group structure read from upstream code",
        "",
        "- `ControlExample(n, seed=1)` draws one plant (A, B, Q, bounds, x0). Steps of `MPCParametric` are one closed loop, not independent samples.",
        "- `LassoExample(n, seed=1)` draws one design matrix. The lambda path stays in that group.",
        "- Upstream `run_parametric_problems.py` dimensions: MPC [20, 40, 60, 80], Lasso [50, 100, 150, 200]. The parametric drivers do not pass a non-default seed.",
        "- The cold MPC loop calls `OSQP(); setup` every step and propagates its own state. The warm loop calls setup once and `update(l, u)`. Those branches are not a paired warm-start comparison.",
        "",
        "## Robotics static set",
        "",
        "- `qpsolvers/mpc_qpbenchmark` HEAD observed via the GitHub API: `954906ac1f48b77e84a5bc314b7c2f00fb0d4ebc`.",
        "- Git clone from this host failed (connection reset). Static `.npz` QA was not run.",
        "",
    ]
    _write(Path(args.output), "\n".join(lines))
    print(args.output)
    return 0


def cmd_preflight(args: argparse.Namespace) -> int:
    from .experiments.preflight import run_preflight

    from .experiments.report_preflight import write_preflight_reports

    out = Path(args.output)
    payload = run_preflight(project_root(), Path(args.config), out)
    write_preflight_reports(project_root(), payload, out / "results.csv")
    print(json.dumps({"run_id": payload["run_id"], "solves_used": payload["solves_used"], "headroom": payload["headroom"]}))
    return 0


def cmd_check_protocol(args: argparse.Namespace) -> int:
    protocol = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    digest = protocol_sha(protocol)
    print(f"protocol_sha={digest}")
    print(f"status={protocol.get('status')}")
    try:
        assert_holdout_released(protocol, "test")
    except HoldoutLocked as exc:
        print(f"test_partition=SEALED {exc}")
        return 0
    print("test_partition=OPEN")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    protocol = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    try:
        assert_holdout_released(protocol, args.partition)
    except HoldoutLocked as exc:
        print(str(exc), file=sys.stderr)
        return 3
    print("Refusing to run a test evaluation from this build: no approved confirmatory runner is enabled.")
    return 3


def cmd_reproduce(args: argparse.Namespace) -> int:
    """Bounded author-environment smoke. Does not launch the paper's 1e6-epoch runs."""
    root = project_root()
    anchor = root / ".venv-anchor" / "Scripts" / "python.exe"
    lines = [
        "# Author reproduction attempt",
        "",
        "Status: **PARTIAL / BLOCKED pending the notes below**. This command does not run the paper's 1e6-epoch training.",
        "",
        f"- L2WS SHA target: `{L2WS_SHA}`",
        f"- Anchor interpreter exists: {anchor.exists()} path `{anchor}`",
        "",
    ]
    if anchor.exists():
        code = (
            "import jax, jaxlib, numpy, sys\n"
            "print('python', sys.version.split()[0])\n"
            "print('jax', jax.__version__)\n"
            "print('jaxlib', jaxlib.__version__)\n"
            "print('devices', jax.devices())\n"
            "print('numpy', numpy.__version__)\n"
        )
        proc = subprocess.run([str(anchor), "-c", code], capture_output=True, text=True, timeout=120)
        lines += ["## Interpreter smoke", "", "```", proc.stdout + proc.stderr, "```", f"exit={proc.returncode}", ""]
    else:
        lines.append("Anchor venv is absent, so no author code was executed.")
    lines += [
        "",
        "## Why this is not an exact reproduction",
        "",
        "- Upstream `benchmarks/configs/mpc/mpc_run.yaml` contains committed merge-conflict markers, so that config cannot be parsed.",
        "- Upstream run configs set `epochs: 1e6`. A full training run is outside the 30-minute pilot cap and was not started.",
        "- The selected bounded target, if the anchor environment imports, is a smoke of JAX itself. A reduced-epoch partial curve needs a separate approval because even a small unroll can be expensive.",
        "- No PyTorch substitute was trained and labeled as L2WS.",
        "",
    ]
    out = Path(args.output)
    if out.suffix:
        _write(out, "\n".join(lines))
    else:
        out.mkdir(parents=True, exist_ok=True)
        _write(out / "REPRODUCTION.md", "\n".join(lines))
    print(args.output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="qpwr")
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("env-audit")
    e.add_argument("--output", required=True)
    e.set_defaults(func=cmd_env_audit)
    s = sub.add_parser("source-audit")
    s.add_argument("--output", required=True)
    s.set_defaults(func=cmd_source_audit)
    pre = sub.add_parser("preflight")
    pre.add_argument("--config", required=True)
    pre.add_argument("--output", required=True)
    pre.set_defaults(func=cmd_preflight)
    c = sub.add_parser("check-protocol")
    c.add_argument("--config", required=True)
    c.set_defaults(func=cmd_check_protocol)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--config", required=True)
    ev.add_argument("--partition", required=True)
    ev.add_argument("--output", required=True)
    ev.set_defaults(func=cmd_evaluate)
    r = sub.add_parser("reproduce")
    r.add_argument("--config", required=True)
    r.add_argument("--output", required=True)
    r.set_defaults(func=cmd_reproduce)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
