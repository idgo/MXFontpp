# Experiment directory (`exp/`)

This folder holds **fine-tuning experiment runs** for MXFont++. Each subfolder is one experiment; inside it you’ll find logs, metrics, checkpoints, and sample images.

## Layout

Two layouts are supported. The **web UI** and **migration script** use the nested form; **CLI** runs can use either.

### Nested (recommended)

Used when starting runs from the web UI or when migrating from `finetune_result/`:

```
exp/
  <exp_name>/                    # e.g. u_font_1, my_custom_run
    result/
      <run_name>/                # often same as exp_name
        log.log                  # training log (args, config, step messages)
        metrics.json             # parsed metrics per step (optional; from parse_finetune_log)
        checkpoints/             # *.pth snapshots
        images/                  # validation sample images (*.png)
```

Example: `exp/u_font_1/result/u_font_1/log.log`

### Flat

Some runs write directly under `result/` (no inner run folder):

```
exp/
  <exp_name>/
    result/
      log.log
      metrics.json
      checkpoints/
      images/
```

The web UI looks for **nested first** (`result/<exp_name>/`), then **flat** (`result/`), so both work.

## Contents

| Path | Description |
|------|-------------|
| `log.log` | Training log: startup args, config dump, and per-step messages. Used by `scripts/parse_finetune_log.py` to build `metrics.json`. |
| `metrics.json` | Array of per-step records (timestamp, step, D, G, FM, L1, AC_*, INDP_*, etc.). Can be generated from `log.log` via the web UI “Parse log” or by running the parse script. |
| `checkpoints/*.pth` | Model checkpoints at save intervals (e.g. `002000.pth`). Used to resume or for inference. |
| `images/*.png` | Validation samples written during training. |

## Creating experiments

- **Web UI**  
  Start a run from the UI; it will create `exp/<name>/result/<name>/` and pass that as `--output_path` to `finetuning.py`. Logs, checkpoints, and images are written there.

- **CLI**  
  Point `work_dir` (or `--output_path`) at the directory that should hold this run, e.g.  
  `exp/my_run/result/my_run` (nested) or `exp/my_run/result` (flat):

  ```bash
  python finetuning.py cfgs/finetune.yaml --output_path exp/my_run/result/my_run --dataset_path path/to/images
  ```

- **Migration from old layout**  
  To move experiments from `finetune_result/<exp_name>/` into `exp/<exp_name>/result/<exp_name>/`:

  ```bash
  python scripts/migrate_finetune_to_exp.py --dry-run   # preview
  python scripts/migrate_finetune_to_exp.py --execute  # run migration
  ```

## Related

- **Web UI**: `python -m webui.app` — lists experiments under `exp/`, shows log/metrics/checkpoints/images, starts new runs.
- **Parse log**: `scripts/parse_finetune_log.py` — reads `log.log` and can write `metrics.json`.
- **Config**: `cfgs/finetune.yaml` — `work_dir` is the output directory for CLI-driven runs.
