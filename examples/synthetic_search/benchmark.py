"""Plan and run paired synthetic benchmarks for exhaustive search heuristics."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from itertools import combinations, product
import json
import math
from pathlib import Path
from tempfile import NamedTemporaryFile
from types import SimpleNamespace

import numpy as np

from modelSelector.criterion import AIC_c
from modelSelector.model import LinearRegressionModel


DESIGNS = ("identifiable", "pairwise-sums")
BASELINE = "backward-only"


@dataclass(frozen=True)
class Method:
    name: str
    backward: bool
    forward_search: bool = True
    pruning: str = "off"
    larger_only: bool = False


METHODS = [Method(BASELINE, backward=True, forward_search=False)]
for backward, pruning in product((True, False), ("off", "reverse", "saliency", "one-step")):
    for larger_only in ((False, True) if pruning != "off" else (False,)):
        name = f"{'B' if backward else 'N'}-{pruning}"
        METHODS.append(Method(name + ("-larger" if larger_only else ""), backward,
                              pruning=pruning, larger_only=larger_only))


def size(mask: int) -> int:
    return bin(mask).count("1")


def tolerance(score: float) -> float:
    return max(1e-8, 1e-12 * abs(score))


def generate_data(case: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Adapt the notebook's orthogonal and pairwise-sum generators."""
    n, p, support_size = (case[name] for name in ("observations", "features", "support"))
    dependent = case["design"] == "pairwise-sums"
    rng = np.random.default_rng([case["seed"], int(dependent), p, support_size, case["replicate"]])
    # Keep the number of columns matched across designs and force a rank deficit.
    rank = max(2, (p + 1) // 2) if dependent else p
    if dependent and rank * (rank + 1) // 2 < p:
        rank += 1
    draws = rng.normal(size=(n, rank))
    basis, _ = np.linalg.qr(draws - draws.mean(axis=0))
    base = np.sqrt(n) * basis
    x = base
    if dependent:
        sums = [base[:, i] + base[:, j] for i, j in combinations(range(rank), 2)]
        pool = np.column_stack([base, *sums])
        x = pool[:, rng.choice(pool.shape[1], p, replace=False)]
    support = np.sort(rng.choice(p, support_size, replace=False))
    signal = x[:, support].sum(axis=1)
    noise = rng.normal(size=n)
    noise -= noise.mean()
    noise -= basis @ (basis.T @ noise)
    noise *= signal.std() / (np.sqrt(case["snr"]) * noise.std())
    return x, signal + noise, support


@dataclass(frozen=True)
class Fit:
    score: float
    twice_nll: float
    ranking: tuple[int, ...]


class Calibrations:
    """Cache deterministic fits across methods, revealing each only when requested."""

    def __init__(self, x: np.ndarray, y: np.ndarray, saliency: str):
        self.x, self.y, self.saliency = x, y, saliency
        self.n, self.p = x.shape
        self.full = (1 << self.p) - 1
        self.bits = tuple(1 << index for index in reversed(range(self.p)))
        self.order = sorted(range(self.full + 1), key=lambda mask: (size(mask), mask), reverse=True)
        self.criterion = AIC_c()
        self.penalties = [self.criterion(SimpleNamespace(nll=0.0, n_params=k, n_data=self.n))
                          for k in range(self.p + 1)]
        self.cache: dict[int, Fit] = {}

    def model_id(self, mask: int) -> str:
        return f"M_{mask:0{self.p}b}"

    def __getitem__(self, mask: int) -> Fit:
        if mask not in self.cache:
            model = LinearRegressionModel(self.model_id(mask))
            model.fit(data_x=self.x, data_y=self.y)
            ranked = sorted(model.get_submodel_saliencies(saliency=self.saliency),
                            key=lambda item: item[1], reverse=True)
            fit = Fit(float(self.criterion(model)), float(2 * model.nll),
                      tuple(self.bits[index] for index, _ in ranked))
            if not math.isfinite(fit.score):
                raise ValueError(f"Nonfinite criterion for {model.id}")
            self.cache[mask] = fit
        return self.cache[mask]


class Search:
    """Run one heuristic with its own fitted and remaining candidates."""

    def __init__(self, calibrations: Calibrations, method: Method):
        self.calibrations = calibrations
        self.method = method
        self.pending = set(calibrations.order)
        self.fitted: dict[int, Fit] = {}
        self.depth: Counter[int] = Counter()
        self.counts: Counter[str] = Counter()
        self.best = calibrations.full
        self.best_score = math.inf

    def excludes_size(self, mask: int, target_size: int) -> bool:
        lower_bound = self.fitted[mask].twice_nll + self.calibrations.penalties[target_size]
        return lower_bound > self.best_score + tolerance(self.best_score)

    def exclude(self, mask: int) -> None:
        count = size(mask)
        bits = self.fitted[mask].ranking
        for dropped in range(self.depth[mask] + 1, count + 1):
            if not self.excludes_size(mask, count - dropped):
                break
            self.pending.difference_update(mask ^ sum(removed) for removed in combinations(bits, dropped))
            self.depth[mask] = dropped

    def fit(self, mask: int, phase: str) -> None:
        fit = self.calibrations[mask]
        self.fitted[mask] = fit
        self.pending.discard(mask)
        self.counts[phase] += 1
        if (fit.score, size(mask), mask) < (self.best_score, size(self.best), self.best):
            self.best, self.best_score = mask, fit.score
            for fitted_mask in self.fitted:
                self.exclude(fitted_mask)
        else:
            self.exclude(mask)

    def has_pending_submodel(self, mask: int) -> bool:
        if 0 in self.pending:
            return True
        subset = (mask - 1) & mask
        while subset:
            if subset in self.pending:
                return True
            subset = (subset - 1) & mask
        return False

    def forward_prune(self, current: int) -> None:
        full = self.calibrations.full
        target_size = size(current)
        if current == full or (self.method.larger_only and target_size <= size(self.best)):
            return
        if self.method.pruning == "one-step":
            for bit in self.calibrations.bits:
                extension = current | bit
                if extension not in self.fitted:
                    self.fit(extension, "forward_pruning")
            return

        closest = min((mask for mask in self.fitted if mask != current and mask & current == current), key=size)
        local = [bit for bit in self.fitted[closest].ranking if not current & bit]
        other = [bit for bit in self.fitted[full].ranking if not closest & bit]
        if self.method.pruning == "reverse":
            local.reverse()
            other.reverse()
        extension = current
        for bit in local + other:
            extension |= bit
            if extension not in self.fitted:
                if not self.has_pending_submodel(extension):
                    continue
                self.fit(extension, "forward_pruning")
            if not self.excludes_size(extension, target_size):
                break

    def run(self) -> dict:
        current = self.calibrations.full
        self.fit(current, "initial")
        remaining = iter(self.calibrations.order)
        while self.pending:
            children = []
            if self.method.backward:
                children = [current ^ bit for bit in self.fitted[current].ranking
                            if current ^ bit in self.pending]
                for child in children:
                    self.fit(child, "backward")
            if self.method.forward_search:
                prefix = 0
                for bit in self.fitted[current].ranking:
                    prefix |= bit
                    if prefix in self.pending:
                        self.fit(prefix, "forward_search")
            if self.method.pruning != "off" and self.pending:
                self.forward_prune(current)
            if children:
                current = min(children, key=lambda mask: (self.fitted[mask].score, mask))
            elif self.pending:
                current = next(mask for mask in remaining if mask in self.pending)
                self.fit(current, "exhaustive")
        return {
            "method": self.method.name,
            "fits": len(self.fitted),
            "best_model": self.calibrations.model_id(self.best),
            "best_aicc": self.best_score,
            **{phase: self.counts[phase] for phase in
               ("initial", "backward", "forward_search", "forward_pruning", "exhaustive")},
        }


def atomic_json(path: Path, value: dict) -> None:
    with NamedTemporaryFile(mode="w", dir=path.parent, encoding="utf-8", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
        except Exception:
            temporary.unlink()
            raise
    temporary.replace(path)


def plan(args: argparse.Namespace) -> None:
    if args.seed < 0 or args.replicates < 1 or args.verify_up_to < 0:
        raise ValueError("Use a nonnegative seed and verification limit, and at least one replicate")
    if min(args.features) < 3 or max(args.features) + 1 >= args.observations:
        raise ValueError("Each feature count must be at least 3 and below observations minus one")
    if min(args.support) < 1 or max(args.support) > min(args.features):
        raise ValueError("Support sizes must be positive and fit every feature count")
    if any(not math.isfinite(snr) or snr <= 0 for snr in args.snr):
        raise ValueError("SNR values must be positive and finite")
    cases = []
    for design, features, support, snr, replicate in product(
        DESIGNS, sorted(set(args.features)), sorted(set(args.support)), sorted(set(args.snr)), range(args.replicates)
    ):
        cases.append(dict(index=len(cases), design=design, features=features, support=support, snr=snr,
                          observations=args.observations, seed=args.seed, replicate=replicate,
                          verify=features <= args.verify_up_to, saliency=args.saliency))
    manifest = {"version": 1, "methods": [asdict(method) for method in METHODS], "cases": cases}
    args.directory.mkdir(parents=True, exist_ok=True)
    (args.directory / "cases").mkdir(exist_ok=True)
    path = args.directory / "plan.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("This directory already has a different plan. Choose a new directory")
    atomic_json(path, manifest)
    print(f"{len(cases)} paired cases. Slurm array bounds: 0-{len(cases) - 1}")


def run_case(case: dict, methods: list[Method]) -> dict:
    x, y, support = generate_data(case)
    rank = int(np.linalg.matrix_rank(x))
    if (rank == case["features"]) != (case["design"] == "identifiable"):
        raise RuntimeError(f"Unexpected design rank for case {case['index']}: {rank}")
    fits = Calibrations(x, y, case["saliency"])
    results = [Search(fits, method).run() for method in methods]
    oracle = None
    if case["verify"]:
        scores = [fits[mask].score for mask in range(fits.full + 1)]
        best = min(scores)
        oracle = {"best_aicc": best, "tied_models": sum(abs(score - best) <= tolerance(best) for score in scores)}
    reference = oracle["best_aicc"] if oracle else min(row["best_aicc"] for row in results)
    for row in results:
        if abs(row["best_aicc"] - reference) > tolerance(reference):
            raise RuntimeError(f"Case {case['index']} method {row['method']} missed best AICc {reference:.12g}")
    return {"case": case, "rank": rank, "true_support": support.tolist(), "oracle": oracle, "results": results}


def run(args: argparse.Namespace) -> None:
    manifest = json.loads((args.directory / "plan.json").read_text())
    methods = [Method(**entry) for entry in manifest["methods"]]
    cases = manifest["cases"]
    if args.case is not None:
        if not 0 <= args.case < len(cases):
            raise ValueError(f"Case index must be between 0 and {len(cases) - 1}")
        cases = [cases[args.case]]
    for case in cases:
        path = args.directory / "cases" / f"{case['index']:06d}.json"
        if path.exists():
            print(f"Case {case['index']} already complete", flush=True)
            continue
        atomic_json(path, run_case(case, methods))
        print(f"Case {case['index']} complete", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True)
    create = commands.add_parser("plan", help="write the experiment plan")
    create.add_argument("directory", type=Path)
    create.add_argument("--features", nargs="+", type=int, default=[8, 12, 16])
    create.add_argument("--support", nargs="+", type=int, default=[2, 4, 6])
    create.add_argument("--snr", nargs="+", type=float, default=[5, 30])
    create.add_argument("--replicates", type=int, default=8)
    create.add_argument("--observations", type=int, default=100)
    create.add_argument("--seed", type=int, default=2024)
    create.add_argument("--saliency", choices=("hessian", "coefficient"), default="hessian")
    create.add_argument("--verify-up-to", type=int, default=12)
    create.set_defaults(command=plan)
    worker = commands.add_parser("run", help="run a case or all cases from the plan")
    worker.add_argument("directory", type=Path)
    worker.add_argument("--case", type=int)
    worker.set_defaults(command=run)
    args = parser.parse_args()
    try:
        args.command(args)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
