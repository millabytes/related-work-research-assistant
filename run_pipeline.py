#!/usr/bin/env python3
import argparse
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT / "data/datasets/multixscience"
LORA_ROOT = PROJECT_ROOT / "models_cache/qwen_lora"
PREPARED_SPLITS = tuple(DATASET_ROOT / name for name in ("train.jsonl", "validation.jsonl", "test.jsonl"))
STEPS = ("tests", "prepare", "smoke", "train", "status", "eval-sample", "eval-full")

_active_process: subprocess.Popen[str] | None = None


def _timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _log(message: str) -> None:
    print(f"[{_timestamp()}] {message}", flush=True)


def _related_work_executable() -> str:
    adjacent = Path(sys.executable).with_name("related-work")
    if adjacent.is_file():
        return str(adjacent)
    executable = shutil.which("related-work")
    if executable:
        return executable
    raise RuntimeError(
        "The related-work entrypoint is unavailable. Run this script with "
        "`uv run python run_pipeline.py ...` after `uv sync --locked`."
    )


def _run(command: list[str], *, dry_run: bool) -> None:
    global _active_process
    rendered = shlex.join(command)
    _log(f"RUN {rendered}")
    if dry_run:
        return
    _active_process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env=os.environ.copy(),
        text=True,
        start_new_session=True,
    )
    return_code = _active_process.wait()
    _active_process = None
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def _latest_checkpoint() -> Path | None:
    checkpoint_root = LORA_ROOT / "checkpoints"
    candidates = []
    for path in checkpoint_root.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        try:
            step = int(path.name.rsplit("-", 1)[-1])
        except ValueError:
            continue
        candidates.append((step, path))
    return max(candidates, default=(0, None))[1]


def _adapter_completed() -> bool:
    manifest = LORA_ROOT / "manifest.json"
    if not manifest.is_file():
        return False
    try:
        return json.loads(manifest.read_text(encoding="utf-8")).get("status") == "completed"
    except (OSError, json.JSONDecodeError):
        return False


def _handle_signal(signum: int, _frame: object) -> None:
    """Terminate the active process group so the batch script can requeue safely."""
    signal_name = signal.Signals(signum).name
    _log(f"Received {signal_name}; stopping the active step")
    if _active_process is not None and _active_process.poll() is None:
        try:
            os.killpg(_active_process.pid, signal.SIGTERM)
            _active_process.wait(timeout=60)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(_active_process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    raise SystemExit(128 + signum)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run explicit Related Work Assistant preparation/training/evaluation steps. "
            "The default is `prepare train status`; evaluation is never implicit."
        )
    )
    parser.add_argument(
        "steps",
        nargs="*",
        choices=STEPS,
        default=["prepare", "train", "status"],
        help="Ordered steps to execute.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="cuda")
    parser.add_argument(
        "--mixed-precision", choices=("auto", "none", "fp16", "bf16"), default="bf16"
    )
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument(
        "--resume",
        choices=("auto", "never", "latest"),
        default="auto",
        help="Auto resumes when a checkpoint exists; latest requires one.",
    )
    parser.add_argument("--eval-limit", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.epochs <= 0 or args.batch_size < 1 or args.gradient_accumulation < 1:
        raise SystemExit("epochs, batch-size and gradient-accumulation must be positive")
    if args.learning_rate <= 0 or args.eval_limit < 1:
        raise SystemExit("learning-rate and eval-limit must be positive")

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGUSR2, _handle_signal)
    cli = _related_work_executable()
    _log(
        f"Pipeline start on host={os.uname().nodename} job={os.getenv('SLURM_JOB_ID', 'local')} "
        f"steps={','.join(args.steps)}"
    )

    for step in args.steps:
        if step == "tests":
            _run([sys.executable, "-m", "pytest", "-q", "tests/unit"], dry_run=args.dry_run)
        elif step == "prepare":
            if all(path.is_file() and path.stat().st_size for path in PREPARED_SPLITS) and not args.force_prepare:
                _log("SKIP prepare: all prepared splits already exist (use --force-prepare to rebuild)")
            else:
                _run([cli, "prepare-multixscience"], dry_run=args.dry_run)
        elif step == "smoke":
            _run(
                [
                    cli,
                    "train-qwen-lora-smoke",
                    "--device",
                    args.device,
                    "--mixed-precision",
                    args.mixed_precision,
                ],
                dry_run=args.dry_run,
            )
        elif step == "train":
            if _adapter_completed() and not args.force_train:
                _log("SKIP train: completed production adapter exists (use --force-train to retrain)")
                continue
            command = [
                cli,
                "train-qwen-lora",
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--gradient-accumulation",
                str(args.gradient_accumulation),
                "--learning-rate",
                str(args.learning_rate),
                "--device",
                args.device,
                "--mixed-precision",
                args.mixed_precision,
            ]
            checkpoint = _latest_checkpoint()
            if args.resume == "latest" and checkpoint is None:
                raise RuntimeError("--resume latest was requested but no checkpoint exists")
            if args.resume == "latest" or (args.resume == "auto" and checkpoint is not None):
                _log(f"Resuming from checkpoint {checkpoint}")
                command.extend(("--resume", "latest"))
            _run(command, dry_run=args.dry_run)
        elif step == "status":
            _run([cli, "qwen-lora-training-status"], dry_run=args.dry_run)
        elif step == "eval-sample":
            _run(
                [
                    cli,
                    "evaluate-generation-a",
                    "--limit",
                    str(args.eval_limit),
                    "--seed",
                    str(args.seed),
                    "--device",
                    args.device,
                ],
                dry_run=args.dry_run,
            )
        elif step == "eval-full":
            _run(
                [
                    cli,
                    "evaluate-generation-a",
                    "--full",
                    "--seed",
                    str(args.seed),
                    "--device",
                    args.device,
                ],
                dry_run=args.dry_run,
            )

    _log("Pipeline completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
