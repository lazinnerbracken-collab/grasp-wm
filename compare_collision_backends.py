"""Compare stored grasp decisions and independently recheck the Coal detour."""
import json
from pathlib import Path
import numpy as np
from panda_lab import PandaLab,HOME
from collision_planning import PandaCollisionChecker
from coal_collision import CoalCollisionChecker


def main():
    root=Path(__file__).resolve().parent
    a=json.loads((root/'collision_results_all/filter_report.json').read_text())
    b=json.loads((root/'coal_results_all/filter_report.json').read_text())
    old={x['candidate_id']:x for x in a['trials']}
    differences=[dict(candidate_id=x['candidate_id'],coal=x['stage'],
                      mujoco=old.get(x['candidate_id'],{}).get('stage'))
                 for x in b['trials'] if x['stage'] != old.get(x['candidate_id'],{}).get('stage')]
    assert len(old)==len(b['trials']) and set(old)=={x['candidate_id'] for x in b['trials']}
    scene=root/'coal_obstacle_test/scene.xml'
    lab=PandaLab(scene)
    lab.segment(HOME,.5,phase='settle')
    with np.load(root/'coal_obstacle_test/detour.npz') as archive:
        path=archive['path']
    validations={}
    for name,cls in [('coal',CoalCollisionChecker),('mujoco',PandaCollisionChecker)]:
        checker=cls(scene,lab.data.qpos,resolution=.005)
        validations[name]=all(checker.edge(x,y)[0] for x,y in zip(path[:-1],path[1:]))
    report=dict(compared_candidates=len(old),decision_differences=differences,
                mujoco_counts=a['counts_by_stage'],coal_counts=b['counts_by_stage'],
                coal_detour_recheck_at_rad=.005,coal_detour_validations=validations,
                coal_execution=b.get('execution'),coal_version=b['coal_version'])
    (root/'coal_results_all/backend_comparison.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    assert all(validations.values()),'Detour recheck failed'


if __name__=='__main__':
    main()
