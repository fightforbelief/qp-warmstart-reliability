"""Launch the approved unconstrained_qp smoke against the copied config.

This file lives outside external/l2ws. It does not edit the upstream tree.
The result label is PARTIAL_REPRODUCTION only when a later status file says so.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

REPO = Path(__file__).resolve().parents[2]
L2WS = REPO / "external" / "l2ws"
sys.path.insert(0, str(L2WS))

def _stage_hydra_configs() -> str:
    """Hydra refuses config paths that climb out of the calling file. Stage the approved copy beside this launcher."""
    destination = Path(__file__).resolve().parent / "_hydra_conf"
    destination.mkdir(exist_ok=True)
    source = REPO / "configs" / "l1_partial"
    for name in ("unconstrained_qp_setup.yaml", "unconstrained_qp_run.yaml"):
        (destination / name).write_bytes((source / name).read_bytes())
    return "_hydra_conf"


def _devices():
    import jax

    return list(jax.devices())


def _disable_usetex() -> None:
    """Upstream modules set text.usetex at import. This process has no LaTeX binary.

    The switch is process-local. It does not edit external/l2ws.
    """
    import matplotlib

    matplotlib.rcParams["text.usetex"] = False


def _require_cpu() -> None:
    devices = _devices()
    bad = [str(device) for device in devices if getattr(device, "platform", "") != "cpu"]
    if bad or not devices:
        raise SystemExit(f"refusing to run without a CPU device: {devices}")
    print("DEVICES", devices, flush=True)


def preflight() -> None:
    import yaml

    setup = yaml.safe_load((REPO / "configs" / "l1_partial" / "unconstrained_qp_setup.yaml").read_text(encoding="utf-8"))
    run = yaml.safe_load((REPO / "configs" / "l1_partial" / "unconstrained_qp_run.yaml").read_text(encoding="utf-8"))
    epochs = int(run["nn_cfg"]["epochs"])
    epochs_jit = int(run["epochs_jit"])
    executed = int(epochs / epochs_jit) * epochs_jit
    print("SETUP", {key: setup[key] for key in ("n_orig", "N_train", "N_test", "seed", "P_split", "split_factor")})
    print(
        "RUN",
        {
            "epochs": epochs,
            "epochs_jit": epochs_jit,
            "executed_epochs": executed,
            "eval_unrolls": int(run["eval_unrolls"]),
            "eval_every_x_epochs": int(run["eval_every_x_epochs"]),
            "train_unrolls": int(run["train_unrolls"]),
            "N_train": int(run["N_train"]),
            "N_test": int(run["N_test"]),
            "output_root": str(Path(__file__).resolve().parent / "outputs"),
        },
    )
    if not (1 <= epochs <= 5 and executed <= 5 and executed > 0):
        raise SystemExit(f"epoch cap violated: requested={epochs} executed={executed}")
    if int(run["eval_unrolls"]) != 50:
        raise SystemExit("eval_unrolls cap violated")
    if int(setup["n_orig"]) != 20 or int(setup["N_train"]) != 100 or int(setup["N_test"]) != 10:
        raise SystemExit("setup sizes differ from the approved example")
    if int(run["N_train"]) != 100 or int(run["N_test"]) != 10:
        raise SystemExit("run sizes differ from the approved example")
    _require_cpu()
    import l2ws.examples.unconstrained_qp  # noqa: F401

    print("IMPORT_OK", flush=True)


def _run_setup_or_train(mode: str) -> None:
    import hydra

    from l2ws.examples import unconstrained_qp as example
    from l2ws.utils.data_utils import copy_data_file, recover_last_datetime

    config_path = _stage_hydra_configs()
    if mode == "setup":

        @hydra.main(version_base="1.1", config_path=config_path, config_name="unconstrained_qp_setup.yaml")
        def setup_main(cfg):
            _require_cpu()
            _disable_usetex()
            print(
                "RESOLVED_SETUP",
                f"n_orig={cfg.n_orig}",
                f"N_train={cfg.N_train}",
                f"N_test={cfg.N_test}",
                f"seed={cfg.seed}",
                flush=True,
            )
            example.setup_probs(cfg)

        setup_main()
        return

    @hydra.main(version_base="1.1", config_path=config_path, config_name="unconstrained_qp_run.yaml")
    def train_main(cfg):
        _require_cpu()
        _disable_usetex()
        epochs = int(cfg.nn_cfg.epochs)
        epochs_jit = int(cfg.epochs_jit)
        executed = int(epochs / epochs_jit) * epochs_jit
        print(
            "RESOLVED_TRAIN",
            f"epochs={epochs}",
            f"executed_epochs={executed}",
            f"eval_unrolls={int(cfg.eval_unrolls)}",
            f"eval_every_x_epochs={int(cfg.eval_every_x_epochs)}",
            f"train_unrolls={int(cfg.train_unrolls)}",
            f"N_train={int(cfg.N_train)}",
            f"N_test={int(cfg.N_test)}",
            flush=True,
        )
        if epochs > 5 or int(cfg.eval_unrolls) > 50 or executed > 5:
            raise SystemExit("resolved config exceeds the approval cap")
        origin = hydra.utils.get_original_cwd()
        if cfg.data.datetime == "":
            cfg.data.datetime = recover_last_datetime(origin, "unconstrained_qp", "data_setup")
        print("SETUP_DATETIME", cfg.data.datetime, flush=True)
        copy_data_file("unconstrained_qp", cfg.data.datetime)
        example.run(cfg)

    train_main()


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in {"preflight", "setup", "train"}:
        raise SystemExit("usage: launch_partial.py preflight|setup|train [hydra overrides]")
    selected = sys.argv.pop(1)
    if selected == "preflight":
        preflight()
    else:
        _run_setup_or_train(selected)
