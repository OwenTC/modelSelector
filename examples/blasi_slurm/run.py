"""Run jumping-backwards model selection for the Blasi model."""

import argparse
import fcntl
from hashlib import sha256
from importlib.metadata import version
import json
import logging
import math
import os
from pathlib import Path
import signal

import numpy as np
from pypesto.engine import MultiProcessEngine, SingleCoreEngine

import blasi_model
from blasi_model import BlasiData, BlasiModel, PARAMETERS
from modelSelector.criterion import AIC_c
from modelSelector.result import ModelSelectionResult
from modelSelector.selection_aglorithm.jumping_backwards_exhaustive_search import (
    jumping_backwards_exhaustive_search,
)


DEFAULT_PETAB = Path(__file__).resolve().parent / "inputs/petab/petab_problem.yaml"


class Paused(Exception):
    """Stop the search between fits."""


class Checkpoint:
    """Append scores and fitted parameters to the checkpoint."""

    header = b"model_id\tobj\tlog10_parameters\n"

    def __init__(self, stream, supermodel, criterion, max_new_fits):
        self.stream = stream
        self.criterion = criterion
        self.max_new_fits = max_new_fits
        self.new_fits = 0
        self.stop_requested = False
        self.scores = ModelSelectionResult()

        stream.seek(0)
        header = stream.readline()
        if header != self.header:
            if not self.header.startswith(header):
                raise ValueError("Expected model_id, obj, and log10_parameters columns.")
            stream.seek(0)
            stream.truncate()
            stream.write(self.header)
            stream.flush()
            os.fsync(stream.fileno())

        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                # A crashed writer may leave an unfinished final record.
                stream.truncate(offset)
                stream.seek(offset)
                break
            model_id, value, _ = line.decode("ascii").rstrip("\n").split("\t")
            score = float(value)
            if (not model_id.startswith("M_") or len(model_id) != len(supermodel.id)
                    or set(model_id[2:]) - {"0", "1"} or not supermodel.is_submodel(model_id)
                    or not math.isfinite(score) or model_id in self.scores):
                raise ValueError(f"Invalid checkpoint record for {model_id}.")
            self.scores[model_id] = score

    def request_stop(self, _signum, _frame):
        """Let the current fit finish before stopping."""
        self.stop_requested = True

    def before_fit(self, model_id):
        """Check whether another calibration can start."""
        if self.stop_requested:
            raise Paused("received a stop signal")
        if (model_id not in self.scores and self.max_new_fits is not None
                and self.new_fits >= self.max_new_fits):
            raise Paused("reached --max-new-fits")

    def save(self, model):
        """Commit a newly scored model before the next calibration."""
        if model.id in self.scores:
            return
        score = self.criterion(model)
        if not math.isfinite(score):
            raise RuntimeError(f"Nonfinite objective score for {model.id}.")
        names = [PARAMETERS[index] for index in model.parameter_indices_free] + [PARAMETERS[-1]]
        values = dict(zip(names, model.parameters.tolist(), strict=True))
        parameters = json.dumps(values, allow_nan=False)
        self.stream.write(f"{model.id}\t{score!r}\t{parameters}\n".encode("ascii"))
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.scores[model.id] = score
        self.new_fits += 1
        if self.new_fits <= 3 or self.new_fits % 100 == 0:
            logging.info("%s obj=%.6f, %d new fits", model.id, score, self.new_fits)


def parse_args():
    """Read the dataset and run settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--petab-yaml", type=Path, default=DEFAULT_PETAB)
    parser.add_argument("--data-mat", type=Path)
    parser.add_argument("--check-inputs", action="store_true")
    parser.add_argument("--supermodel", default="1" * 32,
                        help="32 binary digits (optionally prefixed with M_)")
    parser.add_argument("--n-starts", type=int, default=200)
    parser.add_argument("--workers", type=int, default=int(os.getenv("SLURM_CPUS_PER_TASK", "1")))
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--saliency", choices=("hessian", "coefficient"), default="hessian")
    parser.add_argument("--hessian-step", type=float, default=1e-3)
    parser.add_argument("--max-new-fits", type=int)
    args = parser.parse_args()
    bits = args.supermodel.removeprefix("M_")
    if len(bits) != 32 or set(bits) - {"0", "1"}:
        parser.error("--supermodel must contain exactly 32 binary digits")
    args.supermodel = "M_" + bits
    if min(args.workers, args.n_starts) < 1 or args.seed < 0:
        parser.error("--workers and --n-starts must be positive and --seed nonnegative")
    if not math.isfinite(args.hessian_step) or args.hessian_step <= 0:
        parser.error("--hessian-step must be positive and finite")
    if args.max_new_fits is not None and args.max_new_fits < 1:
        parser.error("--max-new-fits must be positive")
    if args.output is None and not args.check_inputs:
        parser.error("--output is required for a search")
    return args


def check_gradient(model, data):
    """Compare the analytic gradient with central differences."""
    problem = model.problem(data)
    x = 0.46 * problem.lb + 0.54 * problem.ub
    value, gradient = problem.objective(x, (0, 1))
    step = 1e-5
    numeric = np.array([
        (problem.objective(x + direction) - problem.objective(x - direction)) / (2 * step)
        for direction in step * np.eye(problem.dim)
    ])
    if not np.isfinite(value) or not np.isfinite(numeric).all():
        raise RuntimeError("Nonfinite objective or finite-difference gradient at the check point.")
    np.testing.assert_allclose(gradient, numeric, rtol=1e-4, atol=1e-5)
    logging.info("Validated %s with %d fitted parameters", model.id, problem.dim)


def run_search(args, model, data):
    """Restore saved scores and continue the search from the supermodel."""
    digest = sha256(Path(blasi_model.__file__).read_bytes())
    for array in (data.log_data, data.lower, data.upper):
        digest.update(array.tobytes())
    config = {
        "version": 10,
        "input_digest": digest.hexdigest(),
        "dependencies": {name: version(name) for name in ("pypesto", "petab", "fides", "numpy", "scipy", "pandas")},
        "n_data": data.n_observations,
        **{name: getattr(args, name) for name in ("supermodel", "saliency", "hessian_step", "n_starts", "seed")},
    }
    settings = args.output / "settings.json"
    scores_path = args.output / "fits.tsv"
    if settings.exists():
        if json.loads(settings.read_text()) != config:
            raise ValueError("Search settings differ. Use the same settings or a new output directory.")
    else:
        if scores_path.exists() or (args.output / "summary.json").exists() or (args.output / "checkpoint.sqlite3").exists():
            raise ValueError("This directory contains an older run. Use a new output directory.")
        temporary = settings.with_suffix(".tmp")
        temporary.write_text(json.dumps(config, indent=2) + "\n")
        temporary.replace(settings)

    criterion = AIC_c()
    engine = MultiProcessEngine(args.workers) if args.workers > 1 else SingleCoreEngine()
    jumps, status = None, "complete"
    with scores_path.open("a+b") as stream:
        checkpoint = Checkpoint(stream, model, criterion, args.max_new_fits)
        for signum in (signal.SIGUSR1, signal.SIGTERM):
            signal.signal(signum, checkpoint.request_stop)
        fit_options = dict(data=data, checkpoint=checkpoint, engine=engine,
                           n_starts=args.n_starts, seed=args.seed,
                           saliency_method=args.saliency, hessian_step=args.hessian_step)
        try:
            # The supermodel's fit supplies saliencies that a score file cannot store.
            model.fit(**fit_options)
            _, jumps = jumping_backwards_exhaustive_search(
                model, criterion=criterion, result=checkpoint.scores,
                record_exclusions=False, **fit_options,
            )
        except Paused as exc:
            status = "paused"
            logging.info("Paused: %s", exc)

    scores = checkpoint.scores
    best_id, best_score = scores.get_best_model() if scores else (None, None)
    completed = len(scores)
    space_size = 1 << model.id.count("1")
    summary = dict(config, status=status, best_model=best_id, best_obj=best_score,
                   space_size=space_size, completed_fits=completed, new_fits=checkpoint.new_fits,
                   excluded_models=space_size - completed if status == "complete" else None,
                   jumps=jumps)
    temporary = args.output / "summary.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2) + "\n")
    temporary.replace(args.output / "summary.json")
    logging.info("%s. Best model: %s, obj: %s, completed fits: %d",
                 status.capitalize(), best_id, best_score, completed)


def main():
    """Load the Blasi inputs and launch a single search."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    model = BlasiModel(args.supermodel)
    data = BlasiData.load(args.petab_yaml, args.data_mat)
    if data.n_observations <= model.n_params + 1:
        raise ValueError("Too few observations for AICc at the requested model size.")
    logging.info("Loaded %d observations", data.n_observations)
    if args.check_inputs:
        check_gradient(model, data)
        return

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another job already owns this output directory.") from exc
        run_search(args, model, data)


if __name__ == "__main__":
    main()
