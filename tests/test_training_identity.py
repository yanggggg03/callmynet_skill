import copy
import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from training_identity import checkpoint_identity, content_digest, file_digest, validate_comparison


class TrainingIdentityTests(unittest.TestCase):
    def setUp(self):
        import io
        # File-like inputs let array identity tests stay independent of local paths.
        self.io = io
        self.values = {"x": np.array([[.2, .3], [.4, .5], [.6, .7]], dtype=np.float32),
                       "y": np.ones((3, 2), dtype=np.float32),
                       "after": np.zeros((3, 2), dtype=np.float32),
                       "groups": np.arange(3), "row_ids": np.arange(2, 5),
                       "train_idx": np.array([0]), "val_idx": np.array([1]), "test_idx": np.array([2])}
        self.meta = {"schema": "measured-double-stub-v1", "config": {"magnitude_unit": "linear"},
                     "length_bounds": [[0, 90], [0, 90]], "bounds_source": "user_device_limits",
                     "source": "D:/project1/DATA/input.xlsx"}

    def archive(self, metadata=None, values=None):
        stream = self.io.BytesIO()
        np.savez_compressed(stream, **(values or self.values), metadata=np.array(json.dumps(metadata or self.meta)))
        stream.seek(0)
        return stream

    def test_identical_values_with_different_source_paths_match(self):
        second = dict(self.meta, source="D:/project2/DATA/input.xlsx")
        self.assertEqual(content_digest(self.archive()), content_digest(self.archive(second)))

    def test_changed_split_is_rejected(self):
        changed = dict(self.values, train_idx=np.array([1]), val_idx=np.array([0]))
        self.assertNotEqual(content_digest(self.archive()), content_digest(self.archive(values=changed)))

    def test_changed_targets_or_units_change_identity(self):
        changed = copy.deepcopy(self.values); changed["after"][0, 0] = .1
        self.assertNotEqual(content_digest(self.archive()), content_digest(self.archive(values=changed)))
        meta = dict(self.meta, config={"magnitude_unit": "db"})
        self.assertNotEqual(content_digest(self.archive()), content_digest(self.archive(meta)))

    def test_differing_archives_require_matching_verified_contents(self):
        meta = [{"data_sha256": str(i), "training_identity": {"verified": True, "sha256": "same-content"}} for i in range(3)]
        self.assertEqual(validate_comparison(meta), "identical_verified_prepared_content")
        meta[1]["training_identity"]["sha256"] = "other-split"
        with self.assertRaisesRegex(ValueError, "实际不一致"):
            validate_comparison(meta)

    def test_missing_prepared_file_does_not_bypass_comparison(self):
        meta = [{"data_sha256": str(i), "training_identity": {"verified": False}} for i in range(3)]
        with self.assertRaisesRegex(ValueError, "无法核验"):
            validate_comparison(meta)

    def test_identical_checkpoint_archives_work_without_prepared_files(self):
        self.assertEqual(validate_comparison([{"data_sha256": "same"}] * 3), "identical_prepared_archive")


if __name__ == "__main__":
    unittest.main()
