"""Deliberately insert a sphere across a known transit to test actual Panda detouring."""
from pathlib import Path
import argparse
import json
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from panda_lab import PandaLab, HOME
from collision_planning import PandaCollisionChecker,rrt_connect


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checker',choices=['mujoco','coal'],default='mujoco')
    parser.add_argument('--output',type=Path,default=Path('collision_obstacle_test'))
    parser.add_argument('--source',type=Path,default=Path('collision_results'))
    args=parser.parse_args()
    checker_type=PandaCollisionChecker
    if args.checker == 'coal':
        from coal_collision import CoalCollisionChecker
        checker_type=CoalCollisionChecker
    output=args.output.resolve()
    output.mkdir(exist_ok=True)
    source=(args.source/'scene.xml').resolve()
    lab=PandaLab(source)
    lab.segment(HOME,.5,phase='settle')
    with np.load(args.source/'candidate_5_path.npz') as data:
        start,goal=data['transit'][0],data['transit'][-1]
    original_qpos=lab.data.qpos.copy()
    # Search a few placements on the nominal hand sweep, requiring valid endpoints.
    for fraction in [.5,.4,.6,.3,.7]:
        work=mujoco.MjData(lab.model)
        work.qpos[:]=lab.data.qpos
        work.qpos[lab.aq]=(1-fraction)*start+fraction*goal
        mujoco.mj_forward(lab.model,work)
        center=work.xpos[lab.model.body('hand').id].copy()
        for radius in [.025,.015,.01]:
            tree=ET.parse(source)
            ET.SubElement(tree.getroot().find('worldbody'),'geom',name='test_obstacle',type='sphere',
                          pos=' '.join(map(str,center)),size=str(radius),rgba='.9 .1 .1 1')
            scene=output/'scene.xml'
            tree.write(scene)
            checker=checker_type(scene,lab.data.qpos)
            if not checker.valid(start) or not checker.valid(goal):
                continue
            valid,blocking=checker.edge(start,goal)
            if valid:
                continue
            plan=rrt_connect(start,goal,lab.bounds,lambda a,b:checker.edge(a,b)[0],seed=42,timeout=15,iterations=3000)
            if plan.path is None:
                continue
            # Finer recheck, independent of the planner's discretization.
            checker.resolution=.005
            assert all(checker.edge(a,b)[0] for a,b in zip(plan.path[:-1],plan.path[1:]))
            np.testing.assert_array_equal(lab.data.qpos,original_qpos)
            np.savez(output/'detour.npz',path=plan.path,start=start,goal=goal,obstacle_center=center,obstacle_radius=radius)
            report=dict(checker=args.checker,synthetic_obstacle=True,obstacle_center=center.tolist(),radius=radius,
                        endpoints_valid=True,direct_path_valid=False,blocking_contact=blocking,
                        planner=plan.status,waypoints=len(plan.path),iterations=plan.iterations,
                        seconds=plan.seconds,detour_revalidated_at_rad=.005,
                        clearance_m=checker.clearance,kinematic_only=True)
            (output/'report.json').write_text(json.dumps(report,indent=2))
            print(json.dumps(report,indent=2))
            return
    raise RuntimeError('No validated detour found for the tested synthetic obstacle placements')


if __name__=='__main__':
    main()
