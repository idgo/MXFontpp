"""
Flask app for finetune experiments: list results, view log/checkpoints/images, start new runs.
Run with: python -m webui.app [--port 5000] [--experiments-root PATH] [--project-root PATH]
"""
import json
import os
import subprocess
import sys
from pathlib import Path

from flask import Flask, jsonify, request, send_file, send_from_directory, abort

# Add project root so we can import config
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from webui.config import (
    PROJECT_ROOT,
    EXPERIMENTS_ROOT,
    ALLOWED_DATASET_BASES,
    EXP_NAME_PATTERN,
)
from scripts.parse_finetune_log import parse_finetune_log

app = Flask(__name__, static_folder="static", static_url_path="")

# Override config from CLI (set in main())
_experiments_root = None
_project_root = None


def get_experiments_root():
    return _experiments_root if _experiments_root is not None else EXPERIMENTS_ROOT


def get_project_root():
    return _project_root if _project_root is not None else PROJECT_ROOT


# Running jobs: exp_name -> subprocess.Popen
_running_jobs = {}


def _safe_exp_name(name):
    """Return name if safe (alphanumeric, underscore, hyphen); else None."""
    if not name or not isinstance(name, str):
        return None
    name = name.strip()
    if ".." in name or "/" in name or "\\" in name:
        return None
    return name if EXP_NAME_PATTERN.match(name) else None


def _resolve_run_dir(exp_base):
    """
    Resolve the run directory from exp base.
    Structure: exp/<name>/result/<name>/ (common) or exp/<name>/result/ (flat).
    Returns Path or None.
    """
    root = get_experiments_root()
    try:
        if not str(exp_base.resolve()).startswith(str(root.resolve())):
            return None
    except Exception:
        return None
    # Try exp/<name>/result/<name>/ first (most common)
    run_dir = exp_base / "result" / exp_base.name
    if run_dir.exists() and run_dir.is_dir():
        return run_dir
    # Fallback: exp/<name>/result/ (e.g. u_img_7)
    run_dir_flat = exp_base / "result"
    if run_dir_flat.exists() and run_dir_flat.is_dir():
        return run_dir_flat
    return None


def _experiment_base(name):
    """Resolve experiment base directory (exp/<name>/). Return Path or None."""
    safe = _safe_exp_name(name)
    if safe is None:
        return None
    root = get_experiments_root()
    exp_base = (root / safe).resolve()
    try:
        if not str(exp_base).startswith(str(root.resolve())):
            return None
    except Exception:
        return None
    return exp_base


def _experiment_dir(name):
    """Resolve experiment run directory (where log.log, metrics.json, etc. live). Return Path or None."""
    exp_base = _experiment_base(name)
    if exp_base is None or not exp_base.exists() or not exp_base.is_dir():
        return None
    return _resolve_run_dir(exp_base)


def _safe_filename(filename):
    """Allow only simple filenames (no path components)."""
    if not filename or ".." in filename or "/" in filename or "\\" in filename:
        return None
    return filename


def _checkpoints_dir(name):
    """
    Checkpoint directory: exp/<name>/result/checkpoints/ (source of truth only).
    Returns Path or None. Uses path under exp_base without resolve() so symlinks
    do not cause the root check to fail.
    """
    exp_base = _experiment_base(name)
    if exp_base is None:
        return None
    return exp_base / "result" / "checkpoints"


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/experiments", methods=["GET"])
def list_experiments():
    root = get_experiments_root()
    if not root.exists():
        return jsonify([])
    result = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        name = p.name
        if name.startswith("."):
            continue
        run_dir = _resolve_run_dir(p)
        if run_dir is None:
            continue
        log_path = run_dir / "log.log"
        ckpt_dir = p / "result" / "checkpoints"
        img_dir = run_dir / "images"
        n_ckpt = (
            len([f for f in ckpt_dir.glob("*.pth") if f.name != "last.pth"])
            if ckpt_dir.exists()
            else 0
        )
        n_img = len(list(img_dir.glob("*.png"))) if img_dir.exists() else 0
        mtime = run_dir.stat().st_mtime if run_dir.exists() else 0
        dataset_dir = p / "dataset"
        dataset_preview = []
        dataset_count = 0
        if dataset_dir.exists() and dataset_dir.is_dir():
            dataset_files = sorted(dataset_dir.glob("*.png"))
            dataset_count = len(dataset_files)
            dataset_preview = [f.name for f in dataset_files[:10]]
        result.append({
            "name": name,
            "mtime": mtime,
            "has_log": log_path.exists(),
            "n_checkpoints": n_ckpt,
            "n_images": n_img,
            "dataset_preview": dataset_preview,
            "dataset_count": dataset_count,
        })
    sort_by = request.args.get("sort", "time").lower()
    if sort_by == "name":
        result.sort(key=lambda x: x["name"].lower())
    elif sort_by == "name_desc":
        result.sort(key=lambda x: x["name"].lower(), reverse=True)
    elif sort_by == "time_asc":
        result.sort(key=lambda x: x["mtime"])
    else:
        result.sort(key=lambda x: x["mtime"], reverse=True)
    resp = jsonify(result)
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    return resp


@app.route("/api/experiments/<name>/log", methods=["GET"])
def get_log(name):
    exp_dir = _experiment_dir(name)
    if exp_dir is None:
        abort(404)
    log_path = exp_dir / "log.log"
    if not log_path.exists():
        return jsonify({"content": "", "lines": 0})
    tail = request.args.get("tail", type=int)
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    if tail is not None and tail > 0:
        lines = lines[-tail:]
    content = "".join(lines)
    return jsonify({"content": content, "lines": len(lines)})


@app.route("/api/experiments/<name>/metrics", methods=["GET"])
def get_metrics_json(name):
    """Return metrics.json content. 404 if file does not exist."""
    exp_dir = _experiment_dir(name)
    if exp_dir is None:
        abort(404)
    metrics_path = exp_dir / "metrics.json"
    if not metrics_path.exists():
        return jsonify({"error": "metrics.json not found"}), 404
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/experiments/<name>/parse-log", methods=["POST", "OPTIONS"])
def parse_log_to_json(name):
    if request.method == "OPTIONS":
        return "", 204
    """Run parse_finetune_log and write result to metrics.json in experiment dir."""
    exp_dir = _experiment_dir(name)
    if exp_dir is None:
        abort(404)
    log_path = exp_dir / "log.log"
    if not log_path.exists():
        return jsonify({"error": "log.log not found"}), 404
    try:
        blocks = list(parse_finetune_log(log_path))
        out_path = exp_dir / "metrics.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(blocks, f, indent=2, ensure_ascii=False)
        return jsonify({
            "ok": True,
            "path": str(out_path),
            "blocks": len(blocks),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/experiments/<name>/checkpoints", methods=["GET"])
def list_checkpoints(name):
    ckpt_dir = _checkpoints_dir(name)
    if ckpt_dir is None or not ckpt_dir.exists():
        return jsonify([])
    result = []
    for f in sorted(ckpt_dir.glob("*.pth")):
        if f.name == "last.pth":
            continue
        st = f.stat()
        result.append({"name": f.name, "size": st.st_size, "mtime": st.st_mtime})
    return jsonify(result)


@app.route("/api/experiments/<name>/checkpoints/<filename>", methods=["GET"])
def download_checkpoint(name, filename):
    ckpt_dir = _checkpoints_dir(name)
    if ckpt_dir is None:
        abort(404)
    safe = _safe_filename(filename)
    if safe is None or not safe.endswith(".pth"):
        abort(404)
    path = (ckpt_dir / safe).resolve()
    if not path.exists() or not path.is_file():
        abort(404)
    root = get_experiments_root().resolve()
    if not str(path).startswith(str(root)):
        abort(404)
    return send_file(path, as_attachment=True, download_name=safe)


@app.route("/api/experiments/<name>/images", methods=["GET"])
def list_images(name):
    exp_dir = _experiment_dir(name)
    if exp_dir is None:
        abort(404)
    img_dir = exp_dir / "images"
    if not img_dir.exists():
        return jsonify([])
    result = []
    for f in sorted(img_dir.glob("*.png")):
        result.append({"name": f.name, "mtime": f.stat().st_mtime})
    return jsonify(result)


@app.route("/api/experiments/<name>/images/<filename>", methods=["GET"])
def serve_image(name, filename):
    exp_dir = _experiment_dir(name)
    if exp_dir is None:
        abort(404)
    safe = _safe_filename(filename)
    if safe is None:
        abort(404)
    path = (exp_dir / "images" / safe).resolve()
    if not path.exists() or not path.is_file():
        abort(404)
    if not str(path).startswith(str(exp_dir.resolve())):
        abort(404)
    return send_file(path, mimetype="image/png")


@app.route("/api/experiments/<name>/dataset", methods=["GET"])
def list_dataset(name):
    """List dataset images from exp/<name>/dataset/."""
    exp_base = _experiment_base(name)
    if exp_base is None:
        abort(404)
    dataset_dir = exp_base / "dataset"
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        return jsonify([])
    result = []
    for f in sorted(dataset_dir.glob("*.png")):
        result.append({"name": f.name, "mtime": f.stat().st_mtime})
    return jsonify(result)


@app.route("/api/experiments/<name>/dataset/<filename>", methods=["GET"])
def serve_dataset_image(name, filename):
    """Serve a dataset image from exp/<name>/dataset/."""
    exp_base = _experiment_base(name)
    if exp_base is None:
        abort(404)
    safe = _safe_filename(filename)
    if safe is None:
        abort(404)
    dataset_dir = exp_base / "dataset"
    path = (dataset_dir / safe).resolve()
    if not path.exists() or not path.is_file():
        abort(404)
    if not str(path).startswith(str(dataset_dir.resolve())):
        abort(404)
    return send_file(path, mimetype="image/png")


def _dataset_path_allowed(dataset_path):
    path = Path(dataset_path).resolve()
    if not path.exists() or not path.is_dir():
        return False
    path_str = str(path)
    for base in ALLOWED_DATASET_BASES:
        base_resolved = str(Path(base).resolve())
        if path_str == base_resolved or path_str.startswith(base_resolved + os.sep):
            return True
    return False


@app.route("/api/experiments/run", methods=["POST"])
def run_experiment():
    data = request.get_json(force=True, silent=True) or {}
    exp_name = data.get("exp_name", "").strip()
    dataset_path = data.get("dataset_path", "").strip()
    epochs = data.get("epochs")
    fixed_char_txt = (data.get("fixed_char_txt") or "").strip() or None

    safe_name = _safe_exp_name(exp_name)
    if safe_name is None:
        return jsonify({"error": "Invalid experiment name (use only letters, numbers, underscore, hyphen)"}), 400
    if not dataset_path:
        return jsonify({"error": "dataset_path is required"}), 400
    if not _dataset_path_allowed(dataset_path):
        return jsonify({"error": "Dataset path must exist and be under an allowed directory"}), 400

    root = get_experiments_root()
    project_root = get_project_root()
    # exp/<name>/result/<name>/ - matches existing structure
    out_path = root / safe_name / "result" / safe_name
    out_path.mkdir(parents=True, exist_ok=True)

    if safe_name in _running_jobs:
        proc = _running_jobs[safe_name]
        if proc.poll() is None:
            return jsonify({"error": "A run for this experiment is already in progress"}), 409
        else:
            del _running_jobs[safe_name]

    cmd = [
        sys.executable,
        str(project_root / "finetuning.py"),
        "cfgs/finetune.yaml",
        "--output_path", str(out_path),
        "--dataset_path", dataset_path,
    ]
    if epochs is not None:
        try:
            cmd.extend(["--epochs", str(int(epochs))])
        except (TypeError, ValueError):
            pass
    if fixed_char_txt:
        fp = Path(fixed_char_txt).resolve()
        if fp.exists() and fp.is_file() and _dataset_path_allowed(str(fp.parent)):
            cmd.extend(["--fixed_char_txt", str(fp)])

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        _running_jobs[safe_name] = proc
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"job_id": safe_name, "status": "started"})


@app.route("/api/experiments/<name>/status", methods=["GET"])
def job_status(name):
    safe_name = _safe_exp_name(name)
    if safe_name is None:
        abort(404)
    if safe_name in _running_jobs:
        proc = _running_jobs[safe_name]
        if proc.poll() is None:
            return jsonify({"status": "running"})
        else:
            del _running_jobs[safe_name]
    return jsonify({"status": "finished"})


def main():
    global _experiments_root, _project_root
    import argparse
    parser = argparse.ArgumentParser(description="Finetune experiments web UI")
    parser.add_argument("--port", type=int, default=5000, help="Port to listen on")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    parser.add_argument("--experiments-root", type=str, default=None,
                        help="Override experiments root directory")
    parser.add_argument("--project-root", type=str, default=None,
                        help="Override project root directory")
    args = parser.parse_args()
    if args.experiments_root:
        _experiments_root = Path(args.experiments_root).resolve()
    if args.project_root:
        _project_root = Path(args.project_root).resolve()
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
