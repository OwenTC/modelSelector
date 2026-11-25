"""Regression checks for the packed jumping-backwards search."""

import unittest
from unittest.mock import patch

from modelSelector.criterion import AIC_c
from modelSelector.model import SalientModelBase
from modelSelector.result import ModelSelectionResult
from modelSelector.selection_aglorithm.jumping_backwards_exhaustive_search import (
    _CandidateSpace,
    jumping_backwards_exhaustive_search,
)


class ScoredModel(SalientModelBase):
    """A nested objective whose optimum is easy to compute exhaustively."""

    def fit(self, *, scores, visits):
        self.nll = scores[self.id]
        visits.append(self.id)

    def get_submodel_saliencies(self):
        return [(index, 1 / self.n_params) for index in self.parameter_indices_free]


class PackedSearchTests(unittest.TestCase):
    def test_aicc_bounds_count_an_always_fitted_parameter(self):
        class ModelWithBasalRate(ScoredModel):
            def __init__(self, model_id):
                super().__init__(model_id)
                self.n_params += 1

            def fit(self, **kwargs):
                super().fit(**kwargs)
                self.n_data = 7

        scores = {f"M_{mask:03b}": 8.0 * (not mask & 4) + 5.5 * (not mask & 2)
                  for mask in range(8)}
        visits = []
        criterion = AIC_c()
        full = ModelWithBasalRate("M_111")
        full.fit(scores=scores, visits=visits)
        initial = ModelSelectionResult()
        initial[full.id] = criterion(full)
        initial["M_100"] = 18.0

        result, _ = jumping_backwards_exhaustive_search(
            full, criterion=criterion, result=initial, scores=scores, visits=visits,
        )
        self.assertEqual(result.get_best_model(), ("M_110", 14.0))

    def test_candidate_membership_and_size_limit(self):
        space = _CandidateSpace(5)
        self.assertEqual(len(space.pending), 4)
        space.exclude_submodels(0b10111, 0, 2)
        self.assertNotIn(0b10110, space)
        self.assertNotIn(0b00110, space)
        self.assertIn(0b01010, space)
        self.assertIn(0b10000, space)
        space.max_size = 1
        self.assertNotIn(0b01010, space)
        self.assertEqual(list(space.remaining()), [16, 8, 4, 2, 1, 0])

    def test_bulk_exclusion_skips_all_completions_of_a_branch(self):
        space = _CandidateSpace(20)
        witness = (1 << 19) - 1
        space.exclude_submodels(witness, 9, 10)
        self.assertEqual(space.bulk_exclusions[9], [witness])
        child = (1 << 9) - 1
        self.assertNotIn(child, space)
        self.assertFalse(space.pending[child >> 3] & (1 << (child & 7)))
        self.assertEqual(list(space._remaining_at_size(0, witness, 9, 9)), [])
        self.assertIn((1 << 19) | ((1 << 8) - 1), space)

    def test_inverse_jump_requires_a_pending_proper_submodel(self):
        scores = {f"M_{mask:04b}": 1.5 * (4 - bin(mask).count("1")) for mask in range(16)}
        for has_pending_child in (True, False):
            with self.subTest(has_pending_child=has_pending_child):
                visits = []
                full = ScoredModel("M_1111")
                full.fit(scores=scores, visits=visits)
                result = ModelSelectionResult()
                result["M_1111"] = 8.0
                result["M_0000"] = 12.0
                result["M_0010"] = None
                if not has_pending_child:
                    result["M_0001"] = None

                result, _ = jumping_backwards_exhaustive_search(
                    full, result=result, scores=scores, visits=visits,
                )
                after_jump = visits[visits.index("M_1100") + 1]
                self.assertEqual(after_jump == "M_0011", has_pending_child)
                self.assertIsNotNone(result["M_0011"])

    def test_inverse_can_have_a_pending_descendant_below_its_children(self):
        space = _CandidateSpace(3)
        space.discard(0b100)
        space.discard(0b010)
        self.assertTrue(space.has_pending_submodels(0b110))
        space.discard(0)
        self.assertFalse(space.has_pending_submodels(0b110))

    def test_inverse_checks_small_submodels_first(self):
        space = _CandidateSpace(5)
        inverse = 0b11110
        with patch.object(space, "_remaining_at_size", wraps=space._remaining_at_size) as visit:
            self.assertTrue(space.has_pending_submodels(inverse))
        visit.assert_called_once_with(0, inverse, 0, 0)

    def test_inverse_jump_uses_the_best_fit_after_all_jump_children(self):
        weights = (3.0, 3.0, 0.2, 1.1, 1.1)
        scores = {
            f"M_{mask:05b}": sum(
                (weight for index, weight in enumerate(weights) if not mask & (16 >> index)), 0.0
            )
            for mask in range(32)
        }
        visits = []
        result, _ = jumping_backwards_exhaustive_search(
            ScoredModel("M_11111"), scores=scores, visits=visits,
        )
        self.assertEqual(visits[6:10], ["M_10000", "M_11000", "M_11100", "M_00111"])
        self.assertLess(result["M_11000"], result["M_10000"])
        self.assertLess(result["M_11000"], result["M_11100"])

    def test_sparse_search_matches_brute_force_without_duplicate_fits(self):
        weights = {0: 3.0, 2: 0.2, 3: 0.3, 5: 4.0}
        scores = {
            f"M_{mask:06b}": sum(
                (weight for index, weight in weights.items() if not mask & (32 >> index)), 0.0
            )
            for mask in range(64) if mask & ~0b101101 == 0
        }
        expected_id = min(scores, key=lambda model_id: scores[model_id] + model_id.count("1"))

        for record_exclusions in (False, True):
            visits = []
            result, _ = jumping_backwards_exhaustive_search(
                ScoredModel("M_101101"), scores=scores, visits=visits,
                record_exclusions=record_exclusions,
            )
            self.assertEqual(result.get_best_model()[0], expected_id)
            self.assertEqual(len(visits), len(set(visits)))
            self.assertEqual(result.n_fitted_models, len(visits))
            if record_exclusions:
                self.assertEqual(result.keys(), scores.keys())
                self.assertIn(None, result.values())
            else:
                self.assertEqual(set(result), set(visits))

    def test_jump_skips_an_existing_exclusion(self):
        scores = {"M_11": 0.0, "M_01": 0.5, "M_10": 3.0, "M_00": 3.5}
        visits = []
        full = ScoredModel("M_11")
        full.fit(scores=scores, visits=visits)
        result = ModelSelectionResult()
        result["M_11"] = 4.0
        result["M_10"] = None
        result, _ = jumping_backwards_exhaustive_search(
            full, result=result, scores=scores, visits=visits,
        )
        self.assertEqual(result.get_best_model()[0], "M_01")
        self.assertIsNone(result["M_10"])
        self.assertNotIn("M_10", visits)

    def test_new_exclusions_do_not_skip_siblings_but_still_apply_to_the_jump(self):
        class RankedModel(ScoredModel):
            def get_submodel_saliencies(self):
                return [(index, 0.5 if index < 2 else 0.0) for index in self.parameter_indices_free]

        scores = {f"M_{mask:04b}": float(5 * (not mask & 8) + 6 * (not mask & 4)) for mask in range(16)}
        visits = []
        result, _ = jumping_backwards_exhaustive_search(
            RankedModel("M_1111"), scores=scores, visits=visits,
        )
        self.assertEqual(visits[:6], ["M_1111", "M_0111", "M_1011", "M_1101", "M_1110", "M_1100"])
        self.assertIsNone(result["M_1000"])
        self.assertEqual(result.get_best_model(), ("M_1100", 4.0))

    def test_every_deletion_precedes_the_jump(self):
        scores = {f"M_{mask:04b}": 1.5 * (4 - bin(mask).count("1")) for mask in range(16)}
        visits = []
        result, _ = jumping_backwards_exhaustive_search(
            ScoredModel("M_1111"), scores=scores, visits=visits,
        )
        self.assertEqual(visits[:6], ["M_1111", "M_0111", "M_1011", "M_1101", "M_1110", "M_1000"])
        self.assertEqual(result.get_best_model(), ("M_1111", 8.0))
