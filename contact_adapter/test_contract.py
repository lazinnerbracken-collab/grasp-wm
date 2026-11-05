"""Local contract tests against the common.py supplied by the user.

The unavailable project path/transform utilities are represented by test stubs.
These are not TensorFlow or full-project integration tests.
"""
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
for name in ('grasp', 'grasp.adapters', 'grasp.preprocessing'):
    sys.modules[name] = types.ModuleType(name)
sys.modules['grasp'].PROJECT_ROOT = ROOT
sys.modules['grasp'].__version__ = 'test'

def validate_T(value):
    value = np.asarray(value)
    if value.shape != (4,4) or not np.isfinite(value).all():
        raise ValueError('bad transform')
    if not np.allclose(value[3], [0,0,0,1]):
        raise ValueError('bad bottom row')
    if not np.allclose(value[:3,:3].T@value[:3,:3], np.eye(3), atol=1e-5):
        raise ValueError('bad rotation')
    return value

pre = sys.modules['grasp.preprocessing']
pre.validate_T = validate_T
pre.output_path = lambda p: Path(p)
pre.write_json = lambda *args: None
common = types.ModuleType('grasp.adapters.common')
sys.modules[common.__name__] = common
exec(compile((ROOT/'common_reference.txt').read_text(encoding='utf-8-sig'), 'common_reference', 'exec'), common.__dict__)
spec = importlib.util.spec_from_file_location('grasp.adapters.contact_graspnet', ROOT/'grasp/adapters/contact_graspnet.py')
adapter = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = adapter
spec.loader.exec_module(adapter)


class Contract(unittest.TestCase):
    def test_rank_and_frame_and_widths(self):
        poses = np.tile(np.eye(4), (3,1,1))
        poses[:, :3,3] = [[0,0,.6],[.1,0,.7],[.2,0,.8]]
        poses[1,:3,:3] = [[1,0,0],[0,0,-1],[0,1,0]]
        raw = dict(poses=poses, scores=[.2,.9,.5], widths=[.03,.06,.04], contacts=np.zeros((3,3)))
        result = adapter.normalize(raw,2)
        np.testing.assert_array_equal(result['native_indices'], [1,2])
        np.testing.assert_allclose(result['widths'], [.06,.04])
        np.testing.assert_allclose(result['poses'][0,:3,3], [.1,-.1034,.7], atol=1e-6)
        self.assertEqual(result['native_poses'].shape, (3,4,4))

    def test_empty_and_invalid(self):
        raw = dict(poses=[], scores=[], widths=[], contacts=[])
        self.assertEqual(adapter.normalize(raw,10)['poses'].shape,(0,4,4))
        raw.update(poses=[np.eye(4)], scores=[.8], widths=[.12], contacts=[[0,0,.5]])
        with self.assertRaises(ValueError):
            adapter.normalize(raw,10)

    def test_demo_conversion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.savez(root/'demo.npz',depth=np.array([[1.,0],[.5,2.]]),
                     K=np.array([[2,0,0],[0,2,0],[0,0,1]]),segmap=np.array([[0,0],[1,1]]))
            path = adapter.convert_demo(root/'demo.npz',root/'observation')
            points, metadata = common.load_observation(path,1)
            np.testing.assert_allclose(points, [[0,.25,.5]])
            self.assertEqual(metadata['length_unit'],'meter')
            with self.assertRaises(FileExistsError):
                adapter.convert_demo(root/'demo.npz',root/'observation')


if __name__ == '__main__':
    unittest.main()
