import base64
import copy
import json
import sys
import shutil
import threading
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from core import MODEL_IDS, ROOT, analyze_waveforms, load_config
from training_service import TrainingService, parameters, training_rows
from model_registry import registry, valid_id


@contextmanager
def temporary_workspace():
    parent = (ROOT / 'runs').resolve()
    parent.mkdir(exist_ok=True)
    folder = parent / ('test_' + uuid.uuid4().hex)
    folder.mkdir()
    try:
        yield str(folder)
    finally:
        assert folder.resolve().is_relative_to(parent) and folder.name.startswith('test_')
        shutil.rmtree(folder)


class TrainingTests(unittest.TestCase):
    def payload(self):
        text = 'mag0,meg0,L1,L2,match_mag,match_deg\n'
        text += ''.join(f'-6,-{100+i},{40+i},{30+i},.1,30\n' for i in range(30))
        return {'filename': 'training.csv', 'content_base64': base64.b64encode(text.encode()).decode(),
                'magnitude_unit': 'db', 'matched_magnitude_unit': 'linear'}

    def test_training_has_independent_before_after_magnitude_units(self):
        rows = training_rows(self.payload())
        self.assertAlmostEqual(rows[0][0], 10**(-6/20))
        self.assertEqual(rows[0][4], .1)
        invalid = dict(self.payload(), matched_magnitude_unit='db')
        with self.assertRaisesRegex(ValueError, '幅值'):
            training_rows(invalid)

    def test_invalid_parameters_and_model_ids_rejected(self):
        for value in (0, 1.5, float('nan')):
            with self.assertRaises(ValueError):
                parameters({'parameters': {'epochs': value}}, {})
        self.assertFalse(valid_id('../../bad'))
        self.assertTrue(valid_id('mlp_copy_2'))
        with self.assertRaises(ValueError):
            registry({'model_types': [{'id': 'residual_mlp', 'runtime_dir': '.'}]})

    def test_subset_config_and_overlay_are_portable(self):
        with temporary_workspace() as temp:
            path = Path(temp) / 'models.json'
            value = {'models': [{'id': 'mlp_custom_name', 'type': 'residual_mlp', 'checkpoint': 'runs/custom/best.pt'}],
                     'evaluation_model': 'mlp_custom_name'}
            path.write_text(json.dumps(value))
            config = load_config(path)
            self.assertEqual(len(config['models']), 1)
            self.assertEqual(config['models'][0]['id'], 'mlp_custom_name')
            value['models'][0]['id'] = 'replacement'
            value['evaluation_model'] = 'replacement'
            path.with_name('models.active.local.json').write_text(json.dumps(value))
            self.assertEqual(load_config(path)['models'][0]['id'], 'replacement')

    def test_restart_marks_unfinished_job_interrupted(self):
        with temporary_workspace() as temp:
            root = Path(temp)
            job = root / 'job1';job.mkdir()
            (job / 'job.json').write_text(json.dumps({'id': 'job1', 'status': 'training', 'models': []}))
            (root / 'latest.json').write_text(json.dumps({'id': 'job1'}))
            service = TrainingService(ROOT / 'config/models.json', threading.Lock(), root)
            self.assertEqual(service.snapshot()['status'], 'interrupted')
            with self.assertRaises(ValueError):
                service.activate('job1')

    def test_one_and_two_model_waveforms_require_selected_members(self):
        rows = [{'run_id': 'r', 'sample_id': '1', 'scenario_id': 's', 'model_id': mid,
                 'time_s': i/100, 'power_w': 1000, 'target_w': 1000}
                for mid in ('copy_a', 'copy_b') for i in range(101)]
        result = analyze_waveforms({'rows': rows, 'model_ids': ['copy_a', 'copy_b']})
        self.assertEqual(len(result['metrics']), 2)
        with self.assertRaisesRegex(ValueError, '全部'):
            analyze_waveforms({'rows': rows[:101], 'model_ids': ['copy_a', 'copy_b']})
        self.assertEqual(len(analyze_waveforms({'rows': rows[:101], 'model_ids': ['copy_a']})['metrics']), 1)


if __name__ == '__main__':
    unittest.main()
