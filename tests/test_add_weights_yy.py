"""Run from repo: python3 -m unittest discover -s tests -p test_add_weights_yy.py."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import awkward as ak
import numpy as np
import uproot

path = Path(__file__).resolve().parents[1] / 'FCCAnalyses/add_weights_yy.py'
spec = importlib.util.spec_from_file_location('weights_yy', path)
weights_yy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(weights_yy)


class WeightFormulaTests(unittest.TestCase):
    def test_signed_cross_section_weights_keep_signs_and_magnitudes(self):
        result = weights_yy.cross_section_weights(ak.Array([[2.], [-4.], [6.]]), 100, 0.1, 2.)
        np.testing.assert_allclose(result, [0.004, -0.008, 0.012])

    def test_scalar_nominal_branch_supported(self):
        np.testing.assert_allclose(weights_yy.generator_weights(ak.Array([2., -3.])), [2., -3.])

    def test_missing_or_ambiguous_weight_rejected(self):
        for values in [[[1.], []], [[1., 2.]], [[np.nan]], [None, 1.]]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                weights_yy.generator_weights(ak.Array(values))

    def test_mixed_higgs_decays_get_factor_four(self):
        factor = weights_yy.DECAY_FACTORS['pwp8_pp_hh_k3_1_k4_1_84TeV']
        self.assertAlmostEqual(0.5 * factor, 2 * weights_yy.BR_BB * weights_yy.BR_YY)

    def test_no_normalisation_to_selected_event_count(self):
        result = weights_yy.cross_section_weights(ak.Array([[10.], [-10.]]), 100, 1.)
        self.assertAlmostEqual(result[0], 0.1)
        self.assertEqual(result.sum(), 0.)

    def test_no_fallback_without_metadata(self):
        with self.assertRaisesRegex(ValueError, 'missing eventsProcessed'):
            weights_yy.read_events_processed({})


class RootFileTests(unittest.TestCase):
    def setUp(self):
        try:
            import ROOT
        except ImportError:
            self.skipTest('PyROOT is needed to construct a TParameter fixture')
        self.ROOT = ROOT
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.input = self.directory / 'pwp8_pp_hh_k3_1_k4_1_84TeV.root'

    def make_input(self, size=3):
        with uproot.recreate(self.input) as f:
            tree = f.mktree('events', {'weight': 'var * float64', 'x': 'float32'})
            if size:
                tree.extend({'weight': ak.Array([[2.], [-4.], [6.]]), 'x': np.array([10,20,30],dtype='f4')})
            f['note'] = 'Original generation metadata'
        f = self.ROOT.TFile.Open(str(self.input), 'UPDATE')
        self.ROOT.TParameter(int)('eventsProcessed', 100).Write()
        self.ROOT.TParameter(int)('eventsSelected', size).Write()
        f.Close()

    def test_ttree_metadata_original_branches_and_negative_weights_preserved(self):
        self.make_input()
        out = weights_yy.process_file(self.input, self.directory / 'weighted')
        with uproot.open(out, handler=uproot.source.file.MemmapSource) as f:
            self.assertEqual(f['events'].classname, 'TTree')
            self.assertEqual(f['eventsProcessed'].member('fVal'), 100)
            self.assertEqual(f['eventsSelected'].member('fVal'), 3)
            self.assertEqual(str(f['note']), 'Original generation metadata')
            self.assertEqual(ak.to_list(f['events']['weight'].array()), [[2.],[-4.],[6.]])
            np.testing.assert_array_equal(f['events']['x'].array(library='np'), [10,20,30])
            factor = weights_yy.DECAY_FACTORS[self.input.stem]
            np.testing.assert_allclose(f['events']['weight_xsec'].array(library='np'), np.array([2,-4,6])*factor/100)
        with self.assertRaises(FileExistsError):
            weights_yy.process_file(self.input, self.directory / 'weighted')

    def test_empty_selected_sample_keeps_tree_and_processed_count(self):
        self.make_input(size=0)
        out = weights_yy.process_file(self.input, self.directory / 'weighted')
        with uproot.open(out, handler=uproot.source.file.MemmapSource) as f:
            self.assertEqual(f['events'].num_entries, 0)
            self.assertIn('weight_xsec', f['events'])
            self.assertEqual(f['eventsProcessed'].member('fVal'), 100)


if __name__ == '__main__':
    unittest.main()
