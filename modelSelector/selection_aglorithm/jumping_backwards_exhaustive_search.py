"""Jumping backwards exhaustive model selection."""

from itertools import combinations
from math import comb
from typing import Optional

from ..criterion import AIC, CriterionBase
from ..model import SalientModelBase
from ..result import ModelSelectionResult
from .utils import _get_n_parameters_to_delete, _get_n_params_from_id


def _size(mask: int) -> int:
    """Count selected parameters"""
    return bin(mask).count("1")


class _CandidateSpace:
    """Store one pending flag per submodel, indexed by its integer mask."""

    def __init__(self, n_parameters: int):
        self.bits = tuple(1 << index for index in range(n_parameters - 1, -1, -1))
        self.n_models = 1 << n_parameters
        self.pending = bytearray(b"\xff") * ((self.n_models + 7) // 8)
        self.counts = [comb(n_parameters, size) for size in range(n_parameters + 1)]
        self.bulk_exclusions = [[] for _ in self.counts]
        self.max_size = n_parameters

    def __contains__(self, mask: int) -> bool:
        if not 0 <= mask < self.n_models or not self.pending[mask >> 3] & (1 << (mask & 7)):
            return False
        size = _size(mask)
        if size > self.max_size:
            return False
        if self._covered(mask, size):
            self.discard(mask)
            return False
        return True

    def _covered(self, mask: int, size: int) -> bool:
        witnesses = self.bulk_exclusions[size]
        return bool(witnesses) and any(mask & witness == mask for witness in witnesses)

    def discard(self, mask: int) -> None:
        """Clear a pending flag and update the count for its size."""
        byte, bit = mask >> 3, 1 << (mask & 7)
        if self.pending[byte] & bit:
            self.pending[byte] ^= bit
            self.counts[_size(mask)] -= 1

    def remaining(self):
        """Visit pending masks in descending size and mask order."""
        for size in range(len(self.bits), -1, -1):
            yield from self._remaining_at_size(0, self.n_models - 1, size, size)

    def _remaining_at_size(self, selected: int, free: int, needed: int, size: int):
        if size > self.max_size or self.counts[size] == 0 or self._covered(selected | free, size):
            return
        if needed == 0:
            if selected in self:
                yield selected
            return

        # Check the union of the selected and still-free bits so that one
        # broad exclusion can skip every completion of a partial mask.
        while _size(free) >= needed:
            bit = 1 << (free.bit_length() - 1)
            free ^= bit
            yield from self._remaining_at_size(selected | bit, free, needed - 1, size)
            if size > self.max_size or self.counts[size] == 0 or self._covered(selected | free, size):
                break

    def exclude_submodels(self, mask: int, previous: int, depth: int) -> None:
        """Clear newly excluded submodels at sizes still in the search."""
        size = _size(mask)
        first = max(previous + 1, size - self.max_size)
        # Always let the zero-parameter model be fitted. The AICc bound helper
        # only compares positive model sizes before returning a deletion depth.
        last = min(depth, size - 1)
        if first > last:
            return
        bits = [bit for bit in self.bits if mask & bit]
        for deletions in range(first, last + 1):
            child_size = size - deletions
            if self.counts[child_size] == 0:
                continue
            # Store wide exclusions as a superset mask for this size. This
            # avoids visiting millions of IDs just to clear their flags.
            if comb(size, deletions) > 65536:
                self.bulk_exclusions[child_size].append(mask)
                continue
            for removed in combinations(bits, deletions):
                self.discard(mask ^ sum(removed))


def jumping_backwards_exhaustive_search(
    super_model: SalientModelBase,
    criterion: CriterionBase = AIC(),
    result: Optional[ModelSelectionResult] = None,
    *,
    saliency: Optional[str] = None,
    record_exclusions: bool = True,
    **kwargs,
) -> tuple[ModelSelectionResult, int]:
    """
    Search submodels and return the selection result and jump count.

    Each iteration fits the one-parameter deletions available at the start
    of the level, then tries a jump by adding ranked parameters to the empty model.
    Continue from an improving deletion or the largest remaining candidate.
    Supply a fitted supermodel and its score when continuing an existing result.
    Set saliency to a method supported by the model.
    The model's n_params includes any parameters fitted in every candidate.

    Candidate flags use one bit per submodel. Set record_exclusions to False
    for large spaces to keep only fitted scores and existing entries in the
    result. The default records excluded IDs with None after the search.
    Other keyword arguments go to model fitting.
    """
    if result is None:
        result = ModelSelectionResult()
        super_model.fit(**kwargs)
        result[super_model.id] = criterion(super_model)

    super_bits = super_model.id[2:]
    active = [index for index, bit in enumerate(super_bits) if bit == "1"]
    always_fitted = super_model.n_params - len(active)
    candidates = _CandidateSpace(len(active))
    full_mask = candidates.n_models - 1
    parameter_bits = dict(zip(active, candidates.bits))
    n_data = super_model.n_data
    current_mask = full_mask
    current_model = super_model
    current_score = result[super_model.id]
    number_of_jumps = 0
    saliency_options = {} if saliency is None else {"saliency": saliency}

    def model_id(mask: int) -> str:
        if len(active) == len(super_bits) and super_bits:
            return f"M_{mask:0{len(super_bits)}b}"
        bits = ["0"] * len(super_bits)
        for index, bit in parameter_bits.items():
            if mask & bit:
                bits[index] = "1"
        return "M_" + "".join(bits)

    fitted = {}
    for recorded_id, score in result.items():
        if len(recorded_id) != len(super_model.id) or not super_model.is_submodel(recorded_id):
            continue
        mask = sum(bit for index, bit in parameter_bits.items() if recorded_id[index + 2] == "1")
        candidates.discard(mask)
        if score is not None:
            fitted[mask] = (score, 0)

    _, best_score = result.get_best_model()

    def exclude(mask: int) -> None:
        score, previous = fitted[mask]
        size = _size(mask)
        if size < 2:
            return
        # The criterion includes parameters fitted in every model. Mask sizes
        # count only the parameters whose inclusion can change.
        depth = _get_n_parameters_to_delete(best_score, score, size + always_fitted, n_data, criterion)
        if depth <= previous:
            return
        fitted[mask] = (score, depth)
        if mask == full_mask:
            candidates.max_size = max(0, size - depth - 1)
        else:
            candidates.exclude_submodels(mask, previous, depth)

    def refresh_exclusions() -> None:
        # Apply the full-model size limit before marking subsets of other fits.
        exclude(full_mask)
        for mask in fitted:
            if mask != full_mask:
                exclude(mask)

    refresh_exclusions()

    def fit_candidate(mask: int) -> tuple[SalientModelBase, float]:
        nonlocal best_score
        model = type(super_model)(model_id(mask))
        model.fit(**kwargs)
        score = criterion(model)
        result[model.id] = score
        candidates.discard(mask)
        fitted[mask] = (score, 0)
        if score < best_score:
            best_score = score
            refresh_exclusions()
        else:
            exclude(mask)
        return model, score

    remaining = candidates.remaining()
    while True:
        saliencies = sorted(
            current_model.get_submodel_saliencies(**saliency_options),
            key=lambda entry: entry[1],
            reverse=True,
        )
        ranked_bits = [parameter_bits[index] for index, _ in saliencies]
        improvement = None

        # A better fit can tighten the bounds. Keep the already selected
        # deletions so those updates apply to the jump and later levels.
        children = (current_mask & ~bit for bit in ranked_bits)
        available_children = [child for child in children if child in candidates]
        for child in available_children:
            model, score = fit_candidate(child)
            if score < current_score:
                improvement = child, model, score

        # Jump after calibrating the deletions.
        prefix = 0
        for bit in ranked_bits:
            prefix |= bit
            if prefix in candidates:
                fit_candidate(prefix)
                number_of_jumps += 1
                break

        if improvement is not None:
            current_mask, current_model, current_score = improvement
            continue

        next_mask = next(remaining, None)
        if next_mask is None:
            break
        current_mask = next_mask
        current_model, current_score = fit_candidate(next_mask)

    if record_exclusions:
        for mask in range(candidates.n_models):
            if mask not in fitted:
                result[model_id(mask)] = None

    return result, number_of_jumps
