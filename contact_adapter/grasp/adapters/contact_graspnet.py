"""Contact-GraspNet adapter; separate TensorFlow process, common result schema.

Run from grasp_project: python -m grasp.adapters.contact_graspnet --help
"""
from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import os
import subprocess
import tempfile
import numpy as np
from .common import canonicalize, load_observation, sha256

# Standard Panda: native X closing, Z approach -> hand Y closing, TCP at 0.1034m.
T_NATIVE_TCP = np.array([[0,-1,0,0],[1,0,0,0],[0,0,1,.1034],[0,0,0,1]], float)


def normalize(raw, limit):
    arrays = canonicalize(raw['poses'], raw['scores'], T_NATIVE_TCP, .08,
                          limit, method='contact_graspnet')
    n = len(arrays['native_scores'])
    widths = np.asarray(raw['widths'], dtype=np.float32).reshape(-1)
    contacts = np.asarray(raw['contacts'], dtype=np.float32).reshape(-1, 3)
    if widths.shape != (n,) or contacts.shape != (n,3):
        raise ValueError('Widths/contact points do not match the native poses')
    if not np.isfinite(widths).all() or np.any((widths < 0) | (widths > .08001)):
        raise ValueError('Invalid Panda opening')
    if not np.isfinite(contacts).all():
        raise ValueError('Invalid contact points')
    indices = arrays['native_indices']
    arrays.update(widths=widths[indices], native_widths=widths,
                  contact_points=contacts[indices], native_contact_points=contacts)
    return arrays


@dataclass
class ContactGraspNetAdapter:
    toolbox_root: Path
    toolbox_python: Path
    checkpoint_dir: Path
    max_candidates: int = 100
    seed: int = 0
    min_object_points: int = 100
    timeout: float = 600

    def generate(self, inputs, destination, overwrite=False):
        toolbox = self.toolbox_root.expanduser().resolve()
        # Do not resolve a venv Python symlink into its system executable.
        python = self.toolbox_python.expanduser().absolute()
        checkpoint = self.checkpoint_dir.expanduser().resolve()
        destination = Path(destination).resolve()
        if not python.is_file() or not (toolbox/'contact_graspnet/contact_grasp_estimator.py').is_file():
            raise ValueError('Missing Contact-GraspNet checkout or environment Python')
        if not (checkpoint/'config.yaml').is_file() or not list(checkpoint.glob('*.index')):
            raise ValueError('Missing checkpoint configuration or weights')
        if self.max_candidates <= 0 or self.min_object_points <= 0 or not 0 <= self.seed < 2**32:
            raise ValueError('Invalid candidate count, point count or seed')
        if not np.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError('Invalid timeout')
        if not inputs:
            raise ValueError('No input observations')
        destination.mkdir(parents=True, exist_ok=True)
        manifest = {'schema_version':1, 'method':'contact_graspnet',
                    'output_path_base':'manifest_directory', 'trials':[]}
        seen = set()
        for source in map(Path, inputs):
            points, meta = load_observation(source, self.min_object_points)
            key = (meta['object_id'], meta['view_id'])
            if key in seen:
                raise ValueError('Duplicate object/view pair')
            seen.add(key)
            target = destination/key[0]/key[1]
            if target.exists() and any(target.iterdir()) and not overwrite:
                raise FileExistsError(f'Output exists: {target}')
            target.mkdir(parents=True, exist_ok=True)
            if overwrite:
                for name in ('metadata.json', 'grasps.npz'):
                    (target/name).unlink(missing_ok=True)
            record = {'object_id':key[0], 'view_id':key[1], 'status':'failed'}
            manifest['trials'].append(record)
            try:
                with tempfile.TemporaryDirectory(dir=destination, prefix='.contact_worker_') as tmp:
                    work = Path(tmp)
                    with np.load(source, allow_pickle=False) as observation:
                        scene = np.asarray(observation['xyz'], dtype=np.float32)
                        mask = observation['point_object_mask']
                    if not np.isfinite(scene).all() or np.any(scene[:,2] <= 0):
                        raise ValueError('Scene points must be finite and in front of the camera')
                    np.savez(work/'input.npz', points=points, scene=scene, mask=mask)
                    request = dict(toolbox=str(toolbox), checkpoint=str(checkpoint),
                                   input=str(work/'input.npz'), output=str(work/'raw.npz'),
                                   response=str(work/'response.json'), seed=self.seed)
                    (work/'request.json').write_text(json.dumps(request))
                    env = os.environ.copy()
                    env['LD_LIBRARY_PATH'] = str(python.parent.parent/'lib') + ':' + env.get('LD_LIBRARY_PATH','')
                    env.setdefault('CUDA_VISIBLE_DEVICES','0')
                    env['TF_CPP_MIN_LOG_LEVEL'] = '2'
                    worker = Path(__file__).with_name('_contact_graspnet_worker.py')
                    with (target/'worker.log').open('w') as log:
                        subprocess.run([str(python),str(worker),str(work/'request.json')],
                                       cwd=toolbox, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       timeout=self.timeout, check=True)
                    with np.load(work/'raw.npz', allow_pickle=False) as raw:
                        arrays = normalize(raw, self.max_candidates)
                    runtime = json.loads((work/'response.json').read_text())
                np.savez_compressed(target/'grasps.npz', **arrays)
                metadata = dict(schema_version=1, method='contact_graspnet',
                    object_id=key[0], view_id=key[1], camera_frame=meta['camera_frame'],
                    tool_frame='panda_hand_tcp', native_tool_frame='Contact-GraspNet X-closing/Z-approach base',
                    length_unit='meter', transform_convention='T_A_B maps B into A',
                    pose_convention='poses = T_camera_native @ T_native_tool',
                    width_source='predicted offset + configured extra_opening, clipped at gripper_width',
                    input_policy='full observed scene; local object region; contact filtering to object mask',
                    collision_filtering=False, duplicate_suppression=False,
                    object_points=len(points), native_count=len(arrays['native_scores']),
                    candidate_count=len(arrays['scores']), seed=self.seed,
                    observation_sha256=sha256(source), metadata_sha256=sha256(source.with_name('metadata.json')),
                    **runtime)
                (target/'metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False))
                record.update(status='ok' if len(arrays['scores']) else 'no_grasps',
                              output=target.relative_to(destination).as_posix(),
                              candidate_count=len(arrays['scores']))
            except Exception as exc:
                record['error'] = f'{type(exc).__name__}: {exc}; see {target}/worker.log'
            (destination/'manifest.json').write_text(json.dumps(manifest, indent=2))
        return manifest


def convert_demo(source, target):
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    if (target/'observation.npz').exists() or (target/'metadata.json').exists():
        raise FileExistsError('Observation already exists; use --input to reuse it')
    with np.load(source, allow_pickle=False) as inp:
        depth, K, seg = inp['depth'], inp['K'], inp['segmap']
    yy, xx = np.where(np.isfinite(depth) & (depth > .2) & (depth < 1.8))
    z = depth[yy,xx]
    xyz = np.column_stack(((xx-K[0,2])*z/K[0,0], (yy-K[1,2])*z/K[1,1], z)).astype('float32')
    np.savez_compressed(target/'observation.npz', xyz=xyz, point_object_mask=(seg[yy,xx] == 1))
    (target/'metadata.json').write_text(json.dumps(dict(schema_version=1, length_unit='meter',
        object_id='006_mustard_bottle', view_id='synthetic_0', camera_frame='opencv_camera',
        source=str(Path(source).resolve())), indent=2))
    return target/'observation.npz'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--input', type=Path, nargs='+')
    group.add_argument('--from-demo', type=Path)
    parser.add_argument('--observation-dir', type=Path, default=Path('observations/mustard_contact'))
    parser.add_argument('--toolbox-root', type=Path, default=Path.home()/'robotics/contact_graspnet')
    parser.add_argument('--toolbox-python', type=Path)
    parser.add_argument('--checkpoint-dir', type=Path)
    parser.add_argument('--output', type=Path, default=Path('results/contact_graspnet'))
    parser.add_argument('--max-candidates', type=int, default=100)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    root = args.toolbox_root.expanduser()
    inputs = args.input or [convert_demo(args.from_demo, args.observation_dir)]
    adapter = ContactGraspNetAdapter(root, args.toolbox_python or root/'.pixi/envs/default/bin/python',
        args.checkpoint_dir or root/'checkpoints/contact_graspnet_models/scene_test_2048_bs3_hor_sigma_001',
        max_candidates=args.max_candidates)
    result = adapter.generate(inputs, args.output, args.overwrite)
    print(json.dumps(result, indent=2))
    return int(any(r['status'] == 'failed' for r in result['trials']))


if __name__ == '__main__':
    raise SystemExit(main())
