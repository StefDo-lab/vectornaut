# -*- coding: utf-8 -*-
import json
import os
import tempfile
import unittest
from unittest import mock

import torch

from vectornaut.solvers import model_cache
from vectornaut.solvers.model_cache import CACHE_VERSION, load_cached_pinn, save_pinn_model
from vectornaut.solvers.pinn_model import GenericPINN

GOV_EQ = "d2u_dy2 = -G / mu"
BCS = ["u(0) = 0", "u(h) = 0"]


class ModelCacheTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._env = mock.patch.dict(os.environ, {"VECTORNAUT_DATA_DIR": self._tmp.name})
        self._env.start()

    def tearDown(self):
        self._env.stop()
        self._tmp.cleanup()

    def _save(self, design_name, params, domain=(0.0, 1e-6), epochs=50):
        with mock.patch("builtins.print"):
            save_pinn_model(
                GenericPINN(), prefix="pinn_", label="", design_name=design_name,
                gov_eq=GOV_EQ, bcs=BCS, params=params, final_loss=0.1,
                loss_history=[0.1] * epochs, domain=domain, epochs=epochs,
            )

    def _load(self, params, domain=(0.0, 1e-6), epochs=50):
        loaded = []
        with mock.patch("builtins.print"):
            hit = load_cached_pinn(
                prefix="pinn_", label="", loaded_message="loaded",
                gov_eq=GOV_EQ, bcs=BCS, params=params, build_model=GenericPINN,
                on_loaded=lambda model, meta: loaded.append(meta),
                domain=domain, epochs=epochs,
            )
        return hit, loaded

    def test_identical_problem_is_reused(self):
        self._save("film", {"h": 1e-6, "G": 1e4, "mu": 1e-3})

        hit, loaded = self._load({"h": 1e-6, "G": 1e4, "mu": 1e-3})

        self.assertTrue(hit)
        self.assertEqual(loaded[0]["cache_version"], CACHE_VERSION)

    def test_micro_scale_parameters_are_compared_relatively(self):
        # 1e-6 m and 5e-6 m differ by less than the old absolute tolerance of 1e-5.
        self._save("film", {"h": 1e-6, "G": 1e4, "mu": 1e-3})

        hit, _ = self._load({"h": 5e-6, "G": 1e4, "mu": 1e-3})

        self.assertFalse(hit)

    def test_micro_scale_domain_is_compared_relatively(self):
        self._save("film", {"h": 1e-6}, domain=(0.0, 1e-6))

        hit, _ = self._load({"h": 1e-6}, domain=(0.0, 5e-6))

        self.assertFalse(hit)

    def test_models_from_older_cache_versions_are_not_reused(self):
        self._save("film", {"h": 1e-6})
        cache_dir = model_cache.models_dir()
        for name in os.listdir(cache_dir):
            if name.endswith(".json"):
                path = os.path.join(cache_dir, name)
                with open(path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                meta.pop("cache_version")
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(meta, f)

        hit, _ = self._load({"h": 1e-6})

        self.assertFalse(hit)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
