"""Run: python3 -m unittest discover -s tests -p test_hh_quartic_xsec.py"""
import gzip
import importlib.util
from pathlib import Path
import tempfile
import unittest

script = Path(__file__).resolve().parents[1] / 'kinematics/plot_hh_quartic_xsec.py'
spec = importlib.util.spec_from_file_location('quartic_xsec', script)
xsec = importlib.util.module_from_spec(spec)
spec.loader.exec_module(xsec)


def lhe(k3='1D0', k4='-2', value='2D0', error='0.1', beam='42000', closed=True):
    return f'''<LesHouchesEvents version="3.0">
<!--
chhh {k3}
ch4 {k4}
numevts 1
-->
<init>
2212 2212 {beam} 42000 -1 -1 -1 -1 -4 1
{value} {error} 1 9999
</init>
<event>
2 9999 2 125 0 0.1
</event>
''' + ('</LesHouchesEvents>\n' if closed else '')


class QuarticCrossSections(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def write(self, name, content):
        path = self.directory / name
        path.parent.mkdir(exist_ok=True)
        path.write_text(content)
        return path

    def test_couplings_from_header_not_filename(self):
        data = xsec.read_lhe(self.write('misleading_name.lhe', lhe()))
        self.assertEqual((data['k3'], data['k4'], data['xsec_pb']), (1, -2, 2))

    def test_seed_cross_sections_are_averaged_and_error_not_shrunk(self):
        self.write('seed1.lhe', lhe(value='2'))
        self.write('seed2.lhe', lhe(value='4'))
        row, = xsec.collect(self.directory)
        self.assertEqual(row['xsec_pb'], 3)
        self.assertEqual(row['error_pb'], 0.1)
        self.assertEqual((row['files'], row['events']), (2, 2))

    def test_gzip_and_recursive_search(self):
        path = self.directory / 'point'
        path.mkdir()
        with gzip.open(path / 'seed.lhe.gz', 'wt') as stream:
            stream.write(lhe())
        self.assertEqual(xsec.collect(self.directory)[0]['xsec_pb'], 2)

    def test_partial_file_skipped(self):
        self.write('good.lhe', lhe())
        self.write('partial.lhe', lhe(closed=False))
        self.assertEqual(xsec.collect(self.directory)[0]['files'], 1)

    def test_short_generation_rejected(self):
        path = self.write('short.lhe', lhe().replace('numevts 1', 'numevts 125'))
        with self.assertRaisesRegex(ValueError, 'expected 125'):
            xsec.read_lhe(path)

    def test_other_run_settings_are_ignored(self):
        self.write('one.lhe', lhe())
        self.write('two.lhe', lhe(beam='7000'))
        row, = xsec.collect(self.directory)
        self.assertEqual(row['files'], 2)
        self.assertEqual(row['xsec_pb'], 2)

    def test_missing_coupling_rejected(self):
        path = self.write('missing.lhe', lhe().replace('ch4 -2', '#ch4 -2'))
        with self.assertRaisesRegex(ValueError, 'missing chhh/ch4'):
            xsec.read_lhe(path)

    def test_no_usable_points(self):
        self.write('one.lhe', lhe())
        with self.assertRaisesRegex(ValueError, 'no complete'):
            xsec.collect(self.directory, min_events=1000)

    def test_subprocess_rows_are_summed(self):
        content = lhe().replace('-4 1', '-4 2').replace('2D0 0.1 1 9999',
                          '2D0 0.1 1 9999\n3 0.2 1 10000')
        data = xsec.read_lhe(self.write('processes.lhe', content))
        self.assertEqual(data['xsec_pb'], 5)
        self.assertAlmostEqual(data['error_pb'], (0.1**2 + 0.2**2)**0.5)


if __name__ == '__main__':
    unittest.main()
