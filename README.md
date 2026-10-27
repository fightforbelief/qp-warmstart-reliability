# qp-warmstart-reliability

Independent benchmark of learned warm starts for parametric convex QPs.

The scientific question is when, if ever, a learned initialization reduces end-to-end streaming latency relative to a strong cached classical initialization at matched numerical quality.

This tree is the remote project. Experiments run here. Confirmatory test evaluation stays sealed until `protocol/draft_protocol.yaml` is replaced by an approved frozen protocol.

## Commands

From this directory, with the eval virtualenv:

```
$env:PYTHONPATH = "src"
$env:OMP_NUM_THREADS = "1"
python -m qpwr.cli env-audit --output reports/00_REMOTE_ENV_AUDIT.md
python -m qpwr.cli source-audit --output reports/01_SOURCE_DATA_AUDIT.md
python -m pytest tests -q
python -m qpwr.cli preflight --config configs/preflight.yaml --output experiments/preflight/<run_id>
python -m qpwr.cli check-protocol --config protocol/draft_protocol.yaml
python -m qpwr.cli reproduce --config configs/reproduce_anchor.yaml --output reports/03_REPRODUCTION.md
```

`evaluate --partition test` exits with a sealed-holdout error until a human sets `status: frozen` and `holdout_release_approval: true`.
