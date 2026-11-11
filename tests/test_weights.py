"""Regression checks for the previous analysis normalization and ROOT output."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import uproot

spec = importlib.util.spec_from_file_location(
    "hhh_weights", Path(__file__).resolve().parents[1] / "FCCAnalyses/add_weights.py")
weights = importlib.util.module_from_spec(spec)
spec.loader.exec_module(weights)


class WeightingTests(unittest.TestCase):
    def make_input(self, directory, process, values):
        path = Path(directory) / f"{process}.root"
        with uproot.recreate(path) as output:
            tree = output.mktree("events", {key: value.dtype for key, value in values.items()})
            tree.extend(values)
        return path

    def read(self, path):
        return uproot.open(path, handler=uproot.source.file.MemmapSource)

    def test_signal_total_reweights_and_null_kinematics(self):
        with tempfile.TemporaryDirectory() as tmp:
            values = {"weight": np.array([2., 4., 8.]), "mass": np.array([np.nan, 125., 130.])}
            values.update({f"weight_k3_point_{i}": np.array([i + 1., i + 2., i + 3.]) for i in range(9)})
            path = self.make_input(tmp, weights.SIGNAL_KEY, values)
            weights.process_file(path, Path(tmp))
            with self.read(Path(tmp) / "weighted" / path.name) as source:
                self.assertEqual(source['events'].classname, 'TTree')
                result = source['events'].arrays(library='np')
                self.assertEqual(len(result['weight_xsec']), 3)
                self.assertAlmostEqual(result['weight_xsec'].sum(), 0.00156 * 2.25 * 3 * 0.5792**2 * 0.0624)
                for key in values:
                    np.testing.assert_array_equal(result[key], values[key])

    def test_tau_only_ttbb_and_negative_generator_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.make_input(tmp, 'mgp8_pp_ttbb_4f_84TeV', {'weight': np.array([100., -20., 50.])})
            weights.process_file(path, Path(tmp))
            with self.read(Path(tmp) / 'weighted' / path.name) as source:
                result = source['events']['weight_xsec'].array(library='np')
            np.testing.assert_allclose(result, np.array([100., -20., 50.]) * 0.1138**2 * 1.41 / 3)
            self.assertLess(result[1], 0)

    def test_invalid_weights_fail_without_output(self):
        for process, values in [(weights.SIGNAL_KEY, {'weight': np.array([np.nan])}),
                                ('mgp8_pp_tth_5f_84TeV', {'weight': np.array([np.inf])}),
                                ('mgp8_pp_tth_5f_84TeV', {'mass': np.array([125.])})]:
            with self.subTest(process=process), tempfile.TemporaryDirectory() as tmp:
                path = self.make_input(tmp, process, values)
                with self.assertRaises(RuntimeError):
                    weights.process_file(path, Path(tmp))
                self.assertFalse((Path(tmp) / 'weighted' / path.name).exists())

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.make_input(tmp, weights.SIGNAL_KEY, {'weight': np.ones(2)})
            weights.process_file(path, Path(tmp))
            output = Path(tmp) / 'weighted' / path.name
            original = output.read_bytes()
            with self.assertRaises(RuntimeError):
                weights.process_file(path, Path(tmp))
            self.assertEqual(original, output.read_bytes())


if __name__ == '__main__':
    unittest.main()
