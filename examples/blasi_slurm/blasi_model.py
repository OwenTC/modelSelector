"""Fit Blasi steady-state models and rank motif multipliers by saliency."""

from dataclasses import dataclass
from functools import partial
from logging import ERROR

import fides
import numpy as np
import petab.v1 as petab
from pypesto.objective import Objective
from pypesto.optimize import FidesOptimizer, minimize
from pypesto.problem import Problem
from scipy.io import loadmat

from modelSelector.model import SalientModelBase


# Model ID bits follow the ordering of the acetylation rates in the matrix.
MOTIF_PARAMETERS = (
    "a_0ac_k05", "a_0ac_k08", "a_0ac_k12", "a_0ac_k16",
    "a_k05_k05k08", "a_k05_k05k12", "a_k05_k05k16",
    "a_k08_k05k08", "a_k08_k08k12", "a_k08_k08k16",
    "a_k12_k05k12", "a_k12_k08k12", "a_k12_k12k16",
    "a_k16_k05k16", "a_k16_k08k16", "a_k16_k12k16",
    "a_k05k08_k05k08k12", "a_k05k08_k05k08k16",
    "a_k05k12_k05k08k12", "a_k05k12_k05k12k16",
    "a_k05k16_k05k08k16", "a_k05k16_k05k12k16",
    "a_k08k12_k05k08k12", "a_k08k12_k08k12k16",
    "a_k08k16_k05k08k16", "a_k08k16_k08k12k16",
    "a_k12k16_k05k12k16", "a_k12k16_k08k12k16",
    "a_k05k08k12_4ac", "a_k05k08k16_4ac", "a_k05k12k16_4ac", "a_k08k12k16_4ac",
)
PARAMETERS = (*MOTIF_PARAMETERS, "a_b")
STATES = (
    "0ac", "k05", "k08", "k12", "k16", "k05k08", "k05k12", "k05k16",
    "k08k12", "k08k16", "k12k16", "k05k08k12", "k05k08k16", "k05k12k16",
    "k08k12k16", "4ac",
)


def _reaction_matrix(theta):
    """Build the reaction matrix from the one-indexed rate vector."""
    matrix = [
        [-theta[1]-theta[2]-theta[3]-theta[4], theta[33], theta[34], theta[35], theta[36], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [theta[1], -theta[5]-theta[6]-theta[7]-theta[33], 0, 0, 0, theta[37], theta[39], theta[41], 0, 0, 0, 0, 0, 0, 0, 0],
        [theta[2], 0, -theta[8]-theta[9]-theta[10]-theta[34], 0, 0, theta[38], 0, 0, theta[43], theta[45], 0, 0, 0, 0, 0, 0],
        [theta[3], 0, 0, -theta[11]-theta[12]-theta[13]-theta[35], 0, 0, theta[40], 0, theta[44], 0, theta[47], 0, 0, 0, 0, 0],
        [theta[4], 0, 0, 0, -theta[14]-theta[15]-theta[16]-theta[36], 0, 0, theta[42], 0, theta[46], theta[48], 0, 0, 0, 0, 0],
        [0, theta[5], theta[8], 0, 0, -theta[17]-theta[18]-theta[37]-theta[38], 0, 0, 0, 0, 0, theta[49], theta[52], 0, 0, 0],
        [0, theta[6], 0, theta[11], 0, 0, -theta[19]-theta[20]-theta[39]-theta[40], 0, 0, 0, 0, theta[50], 0, theta[55], 0, 0],
        [0, theta[7], 0, 0, theta[14], 0, 0, -theta[21]-theta[22]-theta[41]-theta[42], 0, 0, 0, 0, theta[53], theta[56], 0, 0],
        [0, 0, theta[9], theta[12], 0, 0, 0, 0, -theta[23]-theta[24]-theta[43]-theta[44], 0, 0, theta[51], 0, 0, theta[58], 0],
        [0, 0, theta[10], 0, theta[15], 0, 0, 0, 0, -theta[25]-theta[26]-theta[45]-theta[46], 0, 0, theta[54], 0, theta[59], 0],
        [0, 0, 0, theta[13], theta[16], 0, 0, 0, 0, 0, -theta[27]-theta[28]-theta[47]-theta[48], 0, 0, theta[57], theta[60], 0],
        [0, 0, 0, 0, 0, theta[17], theta[19], 0, theta[23], 0, 0, -theta[29]-theta[49]-theta[50]-theta[51], 0, 0, 0, theta[61]],
        [0, 0, 0, 0, 0, theta[18], 0, theta[21], 0, theta[25], 0, 0, -theta[30]-theta[52]-theta[53]-theta[54], 0, 0, theta[62]],
        [0, 0, 0, 0, 0, 0, theta[20], theta[22], 0, 0, theta[27], 0, 0, -theta[31]-theta[55]-theta[56]-theta[57], 0, theta[63]],
        [0, 0, 0, 0, 0, 0, 0, 0, theta[24], theta[26], theta[28], 0, 0, 0, -theta[32]-theta[58]-theta[59]-theta[60], theta[64]],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, theta[29], theta[30], theta[31], theta[32], -theta[61]-theta[62]-theta[63]-theta[64]],
    ]
    return np.array(matrix)


# The last state equals one minus the sum of the first 15 states.
_CONSERVATION_BASIS = np.vstack((np.eye(15), -np.ones(15)))
_ACETYLATION_DERIVATIVES = np.stack([
    _reaction_matrix(unit_rate)[:15] for unit_rate in np.eye(65)[1:33]
])


def value_and_gradient(x, active, log_data):
    """Evaluate the sum of squared log residuals and its gradient."""
    basal_rate = 10**x[-1]
    acetylation_rates = np.full(32, basal_rate)
    acetylation_rates[active] *= 10**x[:-1]
    # Index zero is unused. Each deacetylation rate is fixed at one.
    reaction = _reaction_matrix(np.r_[0.0, acetylation_rates, np.ones(32)])
    reduced = reaction[:15] @ _CONSERVATION_BASIS

    try:
        first_states = -np.linalg.solve(reduced, reaction[:15, 15])
        state = np.r_[first_states, 1 - first_states.sum()]
        if not np.isfinite(state).all() or np.any(state <= 0):
            return np.inf, np.full_like(x, np.nan)

        # Each motif multiplier changes one rate, and the basal rate changes all 32.
        rhs = (_ACETYLATION_DERIVATIVES @ state).T * (np.log(10.0) * acetylation_rates)
        rhs = np.column_stack((rhs[:, active], rhs.sum(axis=1)))
        state_gradient = -_CONSERVATION_BASIS @ np.linalg.solve(reduced, rhs)
    except np.linalg.LinAlgError:
        return np.inf, np.full_like(x, np.nan)

    residuals = np.log(state[:, None]) - log_data
    observed = np.isfinite(log_data)
    residual_sum = np.sum(residuals, axis=1, where=observed)
    gradient = (2 * residual_sum / state) @ state_gradient
    return np.sum(residuals**2, where=observed), gradient


@dataclass(frozen=True)
class BlasiData:
    """Log measurements and log10 parameter bounds shared by all fits."""

    log_data: np.ndarray
    lower: np.ndarray
    upper: np.ndarray

    @property
    def n_observations(self) -> int:
        """Count observed values, ignoring missing measurements."""
        return int(np.isfinite(self.log_data).sum())

    @classmethod
    def load(cls, petab_yaml, data_mat=None):
        """Read PEtab bounds and measurements, with an optional MATLAB data override."""
        problem = petab.Problem.from_yaml(str(petab_yaml))
        if data_mat is None:
            measurements = problem.measurement_df
            if (not np.isposinf(measurements["time"].to_numpy(dtype=float)).all()
                    or measurements["simulationConditionId"].nunique() != 1):
                raise ValueError("Expected steady-state measurements for one condition.")

            data = np.full((len(STATES), 18), np.nan)
            formulas = problem.observable_df["observableFormula"]
            seen = set()
            for observable, rows in measurements.groupby("observableId"):
                formula = formulas.loc[observable]
                state = formula.removeprefix("x_")
                if not formula.startswith("x_") or state not in STATES or state in seen:
                    raise ValueError(f"Unexpected observable formula for {observable}: {formula}")
                values = rows["measurement"].to_numpy(dtype=float)
                if len(values) > data.shape[1]:
                    raise ValueError(f"More than 18 measurements for {observable}.")
                data[STATES.index(state), :len(values)] = values
                seen.add(state)
        else:
            data = np.asarray(loadmat(data_mat)["data"], dtype=float)

        if (data.shape != (16, 18) or np.isinf(data).any()
                or np.any(data[np.isfinite(data)] <= 0)):
            raise ValueError("Expected a 16 x 18 matrix with positive measurements or NaN.")

        table = problem.parameter_df
        if not table.index.is_unique or not set(PARAMETERS).issubset(table.index):
            raise ValueError("Expected one parameter row for each motif and basal rate.")
        table = table.loc[list(PARAMETERS)]
        if not (table["parameterScale"] == "log10").all():
            raise ValueError("Expected log10 scaling for the motif and basal rates.")
        lower, upper = table[["lowerBound", "upperBound"]].to_numpy(dtype=float).T
        if (not np.isfinite(lower).all() or not np.isfinite(upper).all()
                or np.any((lower <= 0) | (lower >= upper))
                or np.any((lower[:-1] > 1) | (upper[:-1] < 1))):
            raise ValueError("Expected finite positive bounds that include 1 for each motif.")
        return cls(np.log(data), np.log10(lower), np.log10(upper))


class BlasiModel(SalientModelBase):
    """Fit selected motif multipliers and the shared basal rate."""

    def __init__(self, model_id: str):
        super().__init__(model_id)
        self.n_params += 1  # The basal rate is estimated for every model.

    def problem(self, data: BlasiData) -> Problem:
        """Build the pyPESTO problem from the active bits in the model ID."""
        active = self.parameter_indices_free
        indices = [*active, len(PARAMETERS) - 1]
        objective = partial(value_and_gradient, active=active, log_data=data.log_data)
        return Problem(
            objective=Objective(fun=objective, grad=True),
            lb=data.lower[indices],
            ub=data.upper[indices],
            x_names=[PARAMETERS[index] for index in indices],
            x_scales=["log10"] * self.n_params,
        )

    def fit(self, *, data: BlasiData, engine, n_starts: int, seed: int,
            saliency_method: str = "hessian", hessian_step: float = 1e-3,
            checkpoint=None):
        """Run multistart optimization and compute saliencies."""
        if checkpoint is not None:
            checkpoint.before_fit(self.id)
        self.n_data = data.n_observations
        problem = self.problem(data)
        mask = int(self.id[2:], 2)
        np.random.seed(np.random.SeedSequence([seed, mask]).generate_state(1))
        optimizer = FidesOptimizer(hessian_update=fides.BFGS(), verbose=ERROR)
        result = minimize(problem, optimizer=optimizer, engine=engine,
                          n_starts=n_starts, progress_bar=False)
        finite = [
            start for start in result.optimize_result.list
            if np.isfinite(start.fval) and start.x is not None and np.isfinite(start.x).all()
        ]
        if not finite:
            raise RuntimeError(f"All optimization starts failed for {self.id}.")
        x = self.parameters = np.asarray(min(finite, key=lambda start: start.fval).x)
        objective = problem.objective.get_fval
        self.nll = float(objective(x))
        if not np.isfinite(self.nll):
            raise RuntimeError(f"Nonfinite fitted objective for {self.id}.")

        self.hessian_diagonal = None
        if saliency_method == "hessian":
            diagonal = np.empty(self.n_params - 1)
            step = np.zeros_like(x)
            for index, parameter in enumerate(problem.x_names[:-1]):
                step[index] = hessian_step
                curvature = (objective(x + step) + objective(x - step) - 2 * self.nll) / hessian_step**2
                step[index] = 0
                if not np.isfinite(curvature):
                    raise RuntimeError(f"Nonfinite Hessian entry for {self.id}, {parameter}.")
                diagonal[index] = curvature
            self.hessian_diagonal = diagonal
        if checkpoint is not None:
            checkpoint.save(self)

    def get_submodel_saliencies(self):
        """Return normalized motif scores in the original parameter order."""
        active = self.parameter_indices_free
        if not active:
            return []
        scores = np.abs(self.parameters[:-1])
        if self.hessian_diagonal is not None:
            weighted = scores * np.maximum(self.hessian_diagonal, 0)
            if weighted.sum() > 0:
                scores = weighted
        total = scores.sum()
        scores = scores / total if total else np.full(len(active), 1 / len(active))
        return list(zip(active, scores.tolist()))
