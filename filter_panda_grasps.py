"""Filter real Contact-GraspNet candidates using IK, clearance and motion planning."""
import argparse
import json
import os
from pathlib import Path
import time
import xml.etree.ElementTree as ET
if os.name != 'nt':
    os.environ.setdefault('MUJOCO_GL','egl')
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
import panda_predicted_grasp as original
from panda_lab import PandaLab, HOME, Recorder
from lift_metrics import assess_lift
from collision_planning import PandaCollisionChecker, rrt_connect


def setup(output, obstacle=None):
    original.OUT=output
    path,object_position=original.scene()
    if obstacle is not None:
        tree=ET.parse(path)
        ET.SubElement(tree.getroot().find('worldbody'),'geom',name='test_obstacle',type='sphere',
                      pos=' '.join(map(str,obstacle[:3])),size=str(obstacle[3]),rgba='.8 .12 .12 1')
        tree.write(path)
    lab=PandaLab(path)
    lab.segment(HOME,.5,phase='settle')
    with np.load(original.INPUT/'grasp_project/results/contact_graspnet_scene/006_mustard_bottle/synthetic_0/grasps.npz') as a:
        poses,scores,widths,ids=a['poses'],a['scores'],a['widths'],a['candidate_ids']
    with np.load(original.INPUT/'contact_graspnet/ycb_demo/camera_metadata.npz') as a:
        camera,translation=a['T_world_camera'],a['mesh_translation']
    registration=np.eye(4)
    registration[:3,3]=lab.data.xpos[lab.object_body]-translation
    offset=np.eye(4)
    offset[2,3]=-.0034
    return path,lab,registration@camera@poses@offset,scores,widths,ids


def execute(lab, plan, output):
    """Execute the checked approach. Lift monitored dynamically, not preplanned."""
    recorder=Recorder(lab.model,output/'planned_lift.mp4')
    allow=False
    def capture(current):
        recorder(current)
        # Keep the dynamic model's soft-contact tolerance separate from planning margin.
        bad=original.bad_contacts(current,current.data,allow_fingers=allow,tolerance=.002)
        if bad:
            raise RuntimeError('Unexpected dynamic collision: '+str(bad))
    outcome={}
    try:
        for q in plan['transit'][1:]:
            duration=max(1.,float(np.max(np.abs(q-lab.q_ref)))*3.)
            lab.segment(q,duration,capture=capture,phase='transit')
        for q in plan['approach'][1:]:
            lab.segment(q,.4,capture=capture,phase='approach')
        lab.segment(lab.q_ref,.5,capture=capture,phase='settle')
        p,R=lab.pose()
        if np.linalg.norm(p-plan['pose'][:3,3])>.01 or Rotation.from_matrix(plan['pose'][:3,:3]@R.T).magnitude()>.08:
            raise RuntimeError('Grasp tracking error')
        allow=True
        lab.segment(lab.q_ref,1.5,.04,0,capture,'close')
        lab.segment(lab.q_ref,1,0,0,capture,'hold')
        if not np.all(lab.contact_forces()>.1):
            raise RuntimeError('No bilateral contact')
        before=lab.lift_monitor.sample(*lab.pose())
        q=lab.ik(plan['pose'][:3,3]+[0,0,.15],plan['pose'][:3,:3],seed=lab.q_ref)
        lab.segment(q,4,0,0,capture,'lift')
        lab.segment(q,3,0,0,capture,'hold_lift')
        outcome.update(assess_lift(before,[x for x in lab.log if x['phase']=='hold_lift'],lab.lift_monitor.sample(*lab.pose())))
        outcome['warnings']=[int(w.number) for w in lab.data.warning]
        if any(outcome['warnings']):
            outcome['lift_success']=False
    except (RuntimeError,ValueError) as exc:
        outcome.update(lift_success=False,error=str(exc))
    finally:
        recorder.close()
    (output/'execution.json').write_text(json.dumps(outcome,indent=2))
    (output/'telemetry.json').write_text(json.dumps(lab.log))
    return outcome


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checker',choices=['mujoco','coal'],default='mujoco')
    parser.add_argument('--output',type=Path,default=Path('collision_results'))
    parser.add_argument('--limit',type=int,default=100)
    parser.add_argument('--clearance',type=float,default=.002)
    parser.add_argument('--resolution',type=float,default=.01)
    parser.add_argument('--planning-seconds',type=float,default=5.)
    parser.add_argument('--obstacle',type=float,nargs=4,metavar=('X','Y','Z','RADIUS'))
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if args.limit < 1 or args.planning_seconds <= 0 or not np.isfinite(args.planning_seconds):
        parser.error('Positive candidate count and finite planning budget required')
    if args.obstacle and (not np.isfinite(args.obstacle).all() or args.obstacle[3] <= 0):
        parser.error('Obstacle requires finite coordinates and positive radius')
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    path,lab,poses,scores,widths,ids=setup(output,args.obstacle)
    checker_type=PandaCollisionChecker
    if args.checker == 'coal':
        from coal_collision import CoalCollisionChecker
        checker_type=CoalCollisionChecker
    checker=checker_type(path,lab.data.qpos,args.clearance,args.resolution)
    start=lab.q_ref.copy()
    if not checker.valid(start):
        raise RuntimeError('Initial pose is invalid: '+str(checker.explain(start)))
    report=dict(method=f'{args.checker} collision checker + local NumPy RRT-Connect',
                checker=args.checker,
                clearance_m=args.clearance,joint_sample_resolution_rad=args.resolution,
                collision_policy='MJCF active collision pairs; open fingers; only base/table mount exempt',
                obstacle=args.obstacle, approach='10cm Cartesian line, 1cm waypoints with checked joint interpolation',
                lift_planned=False,trials=[])
    if args.checker == 'coal':
        import coal
        report.update(coal_version=coal.__version__,collision_pair_count=len(checker.pairs),
                      kinematics='MuJoCo mj_kinematics; collision queries use Coal',
                      execution_collision_monitor='MuJoCo')
    plans={}
    started=time.perf_counter()
    for index in np.argsort(-scores)[:args.limit]:
        row=dict(index=int(index),candidate_id=str(ids[index]),score=float(scores[index]),
                 predicted_width_m=float(widths[index]),accepted=False)
        report['trials'].append(row)
        T=poses[index].copy()
        R=Rotation.from_matrix(T[:3,:3]).as_matrix()
        T[:3,:3]=R
        position=T[:3,3]
        stage='ik_pregrasp'
        try:
            pre=position-.10*R[:,2]
            qpre=lab.ik(pre,R,seed=start)
            stage='ik_grasp'
            qgoal=lab.ik(position,R,seed=qpre)
            for label,q in [('pregrasp_collision',qpre),('grasp_collision',qgoal)]:
                stage=label
                issues=checker.explain(q)
                if issues:
                    row['contacts']=issues
                    raise RuntimeError('Insufficient clearance')
            stage='approach'
            approach=[qpre]
            for u in np.linspace(0,1,11)[1:]:
                q=lab.ik((1-u)*pre+u*position,R,seed=approach[-1])
                valid,issue=checker.edge(approach[-1],q)
                if not valid:
                    row['blocked_edge']=issue
                    raise RuntimeError('Cartesian approach blocked')
                approach.append(q)
            stage='transit'
            direct_valid,blocked=checker.edge(start,qpre)
            if not direct_valid:
                row['blocked_direct_transit']=blocked
            motion=rrt_connect(start,qpre,lab.bounds,
                               lambda a,b:checker.edge(a,b)[0],seed=int(index),timeout=args.planning_seconds)
            row.update(planner=motion.status,planner_seconds=motion.seconds,planner_iterations=motion.iterations)
            if motion.path is None:
                raise RuntimeError('No transit path found within budget')
            # Independent final recheck of every returned edge.
            for a,b in zip(motion.path[:-1],motion.path[1:]):
                if not checker.edge(a,b)[0]:
                    raise RuntimeError('Returned transit failed validation')
            row.update(accepted=True,stage='accepted',transit_waypoints=len(motion.path))
            plan=dict(transit=motion.path,approach=np.array(approach),pose=T)
            plans[int(index)]=plan
            np.savez_compressed(output/f'candidate_{index}_path.npz',**plan)
        except (ValueError,RuntimeError) as exc:
            row.update(stage=stage,error=str(exc))
        print(json.dumps(row),flush=True)
        (output/'filter_report.json').write_text(json.dumps(report,indent=2))
    report.update(total_seconds=time.perf_counter()-started,collision_checks=checker.calls,
                  accepted_count=len(plans),tested_count=len(report['trials']))
    report['counts_by_stage']={s:sum(x['stage']==s for x in report['trials']) for s in sorted({x['stage'] for x in report['trials']})}
    report['selected_index']=next(iter(plans),None)
    accepted=np.array(list(plans),dtype=np.int64)
    np.savez_compressed(output/'filtered_grasps.npz',source_indices=accepted,
        poses_world_site=poses[accepted],scores=scores[accepted],widths=widths[accepted],
        candidate_ids=ids[accepted])
    if plans:
        selected=next(iter(plans))
        original.snapshot(lab,plans[selected]['approach'][-1],output/'selected_open_gripper.png')
        if args.execute:
            report['execution']=execute(lab,plans[selected],output)
    (output/'filter_report.json').write_text(json.dumps(report,indent=2))
    print('SUMMARY',json.dumps({k:v for k,v in report.items() if k!='trials'}),flush=True)


if __name__=='__main__':
    main()
