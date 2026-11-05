"""Validate and simulate actual server Contact-GraspNet candidates on Panda."""
from pathlib import Path
import json
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from panda_lab import PandaLab, Recorder, build_scene, HOME, TABLE_Z, OBJECT_XY
from lift_metrics import assess_lift

ROOT = Path(__file__).resolve().parent
INPUT = ROOT/'panda_grasp_inputs'
OUT = ROOT/'predicted_panda_results'


def scene():
    OUT.mkdir(exist_ok=True)
    path = build_scene(ROOT/'assets', OUT/'scene.xml')
    tree = ET.parse(path)
    xml = tree.getroot()
    mesh = INPUT/'datasets/ycb/006_mustard_bottle/google_16k/textured.obj'
    v = np.array([list(map(float,l.split()[1:4])) for l in mesh.read_text().splitlines() if l.startswith('v ')])
    low,high=v.min(0),v.max(0)
    position=np.r_[OBJECT_XY-(low[:2]+high[:2])/2,TABLE_Z-low[2]+.001]
    asset=xml.find('asset')
    asset.find("mesh[@name='mustard_bottle_visual_mesh']").set('file',str(mesh))
    asset.find("texture[@name='mustard_bottle_texture']").set('file',str(mesh.with_name('texture_map.png')))
    obj=xml.find(".//body[@name='mustard_bottle']")
    obj.set('pos',' '.join(map(str,position)))
    for g in list(obj.findall('geom')):
        if g.get('name') != 'mustard_bottle_visual_geom':
            obj.remove(g)
    # MuJoCo uses the convex hull for this collision mesh. Conservative at concavities.
    ET.SubElement(obj,'geom',name='mustard_convex_collision',type='mesh',mesh='mustard_bottle_visual_mesh',
                  group='3',friction='1.2 .01 .001',condim='4')
    inertia=obj.find('inertial')
    dimensions=high-low
    diag=.603/12*np.array([dimensions[1]**2+dimensions[2]**2,dimensions[0]**2+dimensions[2]**2,dimensions[0]**2+dimensions[1]**2])
    inertia.set('pos',' '.join(map(str,(low+high)/2)))
    inertia.set('diaginertia',' '.join(map(str,diag)))
    tree.write(path)
    return path,position


def robot_bodies(model):
    ids={model.body('link0').id}
    for i in range(model.nbody):
        if model.body_parentid[i] in ids:
            ids.add(i)
    return ids


def bad_contacts(lab, data, allow_fingers=False, tolerance=.001):
    m=lab.model
    robot=robot_bodies(m)
    base=m.body('link0').id
    out=[]
    for c in data.contact:
        a,b=int(m.geom_bodyid[c.geom1]),int(m.geom_bodyid[c.geom2])
        if c.dist >= -tolerance or not ({a,b}&robot):
            continue
        if {a,b}=={base,0}:
            continue
        if allow_fingers and lab.object_body in (a,b) and ({a,b}&set(lab.finger_bodies)):
            continue
        out.append(dict(geom1=m.geom(c.geom1).name,geom2=m.geom(c.geom2).name,penetration_m=float(-c.dist)))
    return out


def check_path(lab, start, end):
    d=mujoco.MjData(lab.model)
    d.qpos[:]=lab.data.qpos
    d.qpos[lab.fq]=.04
    count=max(40,int(np.max(np.abs(end-start))/.015)+1)
    for u in np.linspace(0,1,count):
        d.qpos[lab.aq]=(1-u)*start+u*end
        mujoco.mj_forward(lab.model,d)
        bad=bad_contacts(lab,d)
        if bad:
            raise RuntimeError('Sampled path collision: '+str(bad[:2]))


def snapshot(lab,q,path):
    from PIL import Image
    d=mujoco.MjData(lab.model)
    d.qpos[:]=lab.data.qpos
    d.qpos[lab.aq]=q
    d.qpos[lab.fq]=.04
    mujoco.mj_forward(lab.model,d)
    camera=mujoco.MjvCamera()
    camera.lookat[:]=[.48,0,.52]
    camera.distance,camera.azimuth,camera.elevation=.65,135,-20
    option=mujoco.MjvOption()
    option.geomgroup[3]=0
    with mujoco.Renderer(lab.model,height=480,width=640) as renderer:
        renderer.update_scene(d,camera=camera,scene_option=option)
        Image.fromarray(renderer.render()).save(path)


def main():
    path,object_position=scene()
    lab=PandaLab(path)
    result=INPUT/'grasp_project/results/contact_graspnet_scene/006_mustard_bottle/synthetic_0'
    with np.load(result/'grasps.npz') as a:
        poses,scores,widths,ids=a['poses'],a['scores'],a['widths'],a['candidate_ids']
    with np.load(INPUT/'contact_graspnet/ycb_demo/camera_metadata.npz') as a:
        camera,mesh_translation=a['T_world_camera'],a['mesh_translation']
    registration=np.eye(4)
    registration[:3,3]=object_position-mesh_translation
    offset=np.eye(4)
    offset[2,3]=-.0034
    targets=registration@camera@poses@offset
    report=dict(scene_registration=registration.tolist(),tcp_offset_m=-.0034,
        assumptions={'mass_kg':.603,'inertia':'uniform bounding-box approximation; not measured',
                     'collision':'original YCB mesh convex hull','friction':'simulation assumption 1.2',
                     'path_check':'joint interpolation sampled at <=0.015rad; penetration tolerance 1mm'},trials=[])
    selected=None
    for index in np.argsort(-scores):
        lab.reset()
        lab.segment(HOME,.5,phase='settle')
        T=targets[index].copy()
        # Account for the small vertical settling of the physical object.
        T[:3,3]+=lab.data.xpos[lab.object_body]-object_position
        row=dict(index=int(index),candidate_id=str(ids[index]),score=float(scores[index]),width_m=float(widths[index]))
        report['trials'].append(row)
        try:
            position,R=T[:3,3],T[:3,:3]
            R=Rotation.from_matrix(R).as_matrix()
            pre=position-.10*R[:,2]
            qpre=lab.ik(pre,R,seed=HOME)
            qgrasp=lab.ik(position,R,seed=qpre)
            check_path(lab,HOME,qpre)
            check_path(lab,qpre,qgrasp)
            qlift=lab.ik(position+[0,0,.15],R,seed=qgrasp)
            row['planning']='passed'
            snapshot(lab,qgrasp,OUT/f'candidate_{index}_open_gripper.png')
            recorder=Recorder(lab.model,OUT/f'candidate_{index}_lift.mp4')
            phase={'closing':False}
            def capture(current):
                recorder(current)
                bad=bad_contacts(current,current.data,allow_fingers=phase['closing'],tolerance=.002)
                if bad:
                    raise RuntimeError('Dynamic collision: '+str(bad[:2]))
            try:
                lab.segment(qpre,3,capture=capture,phase='pregrasp')
                lab.segment(qpre,.5,capture=capture,phase='settle')
                lab.segment(qgrasp,2,capture=capture,phase='approach')
                lab.segment(qgrasp,.5,capture=capture,phase='settle')
                p,r=lab.pose()
                if np.linalg.norm(p-position)>.01 or Rotation.from_matrix(R@r.T).magnitude()>.08:
                    raise RuntimeError('Grasp tracking error exceeds tolerance')
                phase['closing']=True
                lab.segment(qgrasp,1.5,.04,0,capture,'close')
                lab.segment(qgrasp,1,0,0,capture,'hold')
                if not np.all(lab.contact_forces()>.1):
                    raise RuntimeError('No bilateral contact; lift aborted')
                before=lab.lift_monitor.sample(*lab.pose())
                lab.segment(qlift,4,0,0,capture,'lift')
                lab.segment(qlift,3,0,0,capture,'hold_lift')
                row.update(assess_lift(before,[x for x in lab.log if x['phase']=='hold_lift'],lab.lift_monitor.sample(*lab.pose())))
                row['warnings']=[int(w.number) for w in lab.data.warning]
                if any(row['warnings']):
                    row['lift_success']=False
                if row['lift_success']:
                    selected=int(index)
            finally:
                recorder.close()
            (OUT/f'candidate_{index}_telemetry.json').write_text(json.dumps(lab.log))
        except (RuntimeError,ValueError) as exc:
            row['error']=str(exc)
        print(json.dumps(row),flush=True)
        (OUT/'report.json').write_text(json.dumps(report,indent=2))
        if selected is not None:
            break
    report['successful_candidate']=selected
    report['lift_success']=selected is not None
    (OUT/'report.json').write_text(json.dumps(report,indent=2))
    print('RESULT',report['lift_success'],selected)


if __name__=='__main__':
    main()
