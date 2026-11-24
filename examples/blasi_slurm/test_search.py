"""Check the Blasi model's parameter count and saliency."""

import json
import unittest
from tempfile import TemporaryFile

import numpy as np

from blasi_model import BlasiModel
from modelSelector.criterion import AIC_c
from run import Checkpoint


class BlasiModelTests(unittest.TestCase):
    def test_aicc_counts_the_basal_rate(self):
        model = BlasiModel("M_" + "1" * 7 + "0" * 25)
        model.nll = 16.1570009618775
        model.n_data = 252
        self.assertEqual(model.n_params, 8)
        self.assertAlmostEqual(AIC_c()(model), 48.90659451634759)

    def test_saliency_uses_motif_indices_and_excludes_the_basal_rate(self):
        model = BlasiModel("M_101" + "0" * 29)
        model.parameters = np.array([1.0, -3.0, 7.0])
        model.hessian_diagonal = np.array([2.0, 2.0])
        self.assertEqual(model.get_submodel_saliencies(), [(0, 0.25), (2, 0.75)])

    def test_checkpoint_recovers_scores_and_removes_a_torn_record(self):
        full = BlasiModel("M_11")
        candidate = BlasiModel("M_01")
        candidate.nll = 1.0
        candidate.n_data = 20
        candidate.parameters = np.array([0.25, -1.5])
        with TemporaryFile(mode="a+b") as stream:
            stream.write(Checkpoint.header + b'M_11\t10.0\t{"a_b": -1.0}\nM_01\t1\t{')
            checkpoint = Checkpoint(stream, full, AIC_c(), max_new_fits=1)
            self.assertEqual(checkpoint.scores, {"M_11": 10.0})
            checkpoint.save(candidate)
            checkpoint = Checkpoint(stream, full, AIC_c(), max_new_fits=1)
            self.assertEqual(checkpoint.scores, {"M_11": 10.0, "M_01": AIC_c()(candidate)})
            stream.seek(0)
            _, _, row = stream.readlines()
            self.assertEqual(json.loads(row.split(b"\t")[2]), {"a_0ac_k08": 0.25, "a_b": -1.5})
