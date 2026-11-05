"""Executed only by the existing Contact-GraspNet TensorFlow environment."""
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time
import numpy as np


def run(request):
    toolbox = Path(request['toolbox'])
    sys.path.insert(0, str(toolbox))
    sys.path.insert(0, str(toolbox/'contact_graspnet'))
    import tensorflow.compat.v1 as tf
    tf.disable_eager_execution()
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        raise RuntimeError('No TensorFlow GPU available')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    import config_utils
    from contact_grasp_estimator import GraspEstimator
    random.seed(request['seed'])
    np.random.seed(request['seed'])
    tf.set_random_seed(request['seed'])
    checkpoint = Path(request['checkpoint'])
    cfg = config_utils.load_config(str(checkpoint), batch_size=1)
    if abs(float(cfg['DATA']['gripper_width'])-.08) > 1e-6:
        raise ValueError('Adapter supports standard 8cm Panda checkpoints only')
    if cfg['TEST'].get('center_to_tip',0) != 0:
        raise ValueError('Nonzero center_to_tip requires another frame conversion')
    start = time.perf_counter()
    estimator = GraspEstimator(cfg)
    estimator.build_network()
    saver = tf.train.Saver()
    state = tf.train.get_checkpoint_state(str(checkpoint))
    if state is None:
        raise FileNotFoundError('Missing TensorFlow checkpoint state')
    prefix = checkpoint/Path(state.model_checkpoint_path).name
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    with tf.Session(config=config) as session:
        saver.restore(session, str(prefix))
        load_seconds = time.perf_counter()-start
        with np.load(request['input'], allow_pickle=False) as inp:
            points = inp['points'].astype(np.float32)
            scene = inp['scene'].astype(np.float32)
        start = time.perf_counter()
        pose_map, score_map, contact_map, width_map = estimator.predict_scene_grasps(
            session, scene, pc_segments={1: points}, local_regions=True,
            filter_grasps=True, forward_passes=1)
        poses, scores = pose_map[1], score_map[1]
        contacts, widths = contact_map[1], width_map[1]
        seconds = time.perf_counter()-start
    np.savez(request['output'], poses=np.asarray(poses).reshape(-1,4,4),
             scores=np.asarray(scores).reshape(-1), contacts=np.asarray(contacts).reshape(-1,3),
             widths=np.asarray(widths).reshape(-1))
    assets = [checkpoint/'config.yaml', *checkpoint.glob(prefix.name+'.index'),
              *checkpoint.glob(prefix.name+'.data-*'),
              toolbox/'contact_graspnet/contact_graspnet.py',
              toolbox/'contact_graspnet/contact_grasp_estimator.py']
    hashes = {}
    for path in assets:
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024*1024), b''):
                digest.update(block)
        hashes[path.name] = digest.hexdigest()
    revision = subprocess.run(['git','-C',str(toolbox),'rev-parse','HEAD'],
                              capture_output=True,text=True,check=False).stdout.strip()
    response = dict(model_load_seconds=load_seconds, inference_seconds=seconds,
        runtime={'tensorflow':tf.__version__, 'numpy':np.__version__, 'gpu':str(gpus)},
        toolbox_revision=revision, assets_sha256=hashes,
        native_selection={'num_samples':cfg['TEST']['num_samples'],
                          'first_thres':cfg['TEST']['first_thres'],
                          'second_thres':cfg['TEST'].get('second_thres',cfg['TEST']['first_thres'])})
    Path(request['response']).write_text(json.dumps(response, allow_nan=False))


if __name__ == '__main__':
    run(json.loads(Path(sys.argv[1]).read_text()))
