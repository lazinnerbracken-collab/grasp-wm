"""Scientific figures for the grasp filtering and deliberate-obstacle benchmark."""
from pathlib import Path
import json
import numpy as np
import mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
from panda_lab import PandaLab,HOME
import panda_predicted_grasp as original


def main():
    output=Path('collision_results_all')
    report=json.loads((output/'filter_report.json').read_text())
    fig,ax=plt.subplots(figsize=(8,4.5))
    labels=['Accepted','IK pregrasp failed','IK grasp failed','Collision / clearance']
    keys=['accepted','ik_pregrasp','ik_grasp','grasp_collision']
    values=[report['counts_by_stage'].get(k,0) for k in keys]
    bars=ax.barh(labels,values,color=['#31875b','#64748b','#94a3b8','#c34545'])
    ax.bar_label(bars,padding=4)
    ax.set(xlabel='Candidates (100 tested)',xlim=(0,80),title='Panda grasp filtering: 2 mm clearance')
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(output/'filter_summary.png',dpi=160)
    plt.close(fig)
    # Compare actual full-robot placement for a rejected and accepted candidate.
    lab=PandaLab(output/'scene.xml')
    lab.segment(HOME,.5,phase='settle')
    with np.load(original.INPUT/'grasp_project/results/contact_graspnet_scene/006_mustard_bottle/synthetic_0/grasps.npz') as data:
        poses=data['poses']
    with np.load(original.INPUT/'contact_graspnet/ycb_demo/camera_metadata.npz') as data:
        camera,translation=data['T_world_camera'],data['mesh_translation']
    register=np.eye(4)
    register[:3,3]=lab.data.xpos[lab.object_body]-translation
    offset=np.eye(4)
    offset[2,3]=-.0034
    from scipy.spatial.transform import Rotation
    T=register@camera@poses[10]@offset
    R=Rotation.from_matrix(T[:3,:3]).as_matrix()
    qpre=lab.ik(T[:3,3]-.10*R[:,2],R,seed=HOME)
    q=lab.ik(T[:3,3],R,seed=qpre)
    original.snapshot(lab,q,output/'rejected_10.png')
    result=Image.new('RGB',(1280,520),'white')
    result.paste(Image.open(output/'rejected_10.png'),(0,40))
    result.paste(Image.open(output/'selected_open_gripper.png'),(640,40))
    draw=ImageDraw.Draw(result)
    draw.text((15,12),'Rejected #10: link5 intersects table (~12 mm)',fill='black')
    draw.text((655,12),'Accepted #5: checked approach, 2 mm margin',fill='black')
    result.save(output/'collision_comparison.png')
    # TCP traces visualize the path; collision tests use the complete robot.
    root=Path('collision_obstacle_test')
    obstacle=json.loads((root/'report.json').read_text())
    lab=PandaLab(root/'scene.xml')
    with np.load(root/'detour.npz') as data:
        path,start,goal=data['path'],data['start'],data['goal']
    def trace(path):
        d=mujoco.MjData(lab.model)
        d.qpos[:]=lab.data.qpos
        points=[]
        for a,b in zip(path[:-1],path[1:]):
            for t in np.linspace(0,1,80):
                d.qpos[lab.aq]=(1-t)*a+t*b
                mujoco.mj_forward(lab.model,d)
                points.append(d.xpos[lab.model.body('hand').id].copy())
        return np.array(points)
    fig=plt.figure(figsize=(8,6))
    ax=fig.add_subplot(111,projection='3d')
    for points,color,label in [(trace([start,goal]),'crimson','Blocked direct hand path'),
                               (trace(path),'seagreen','RRT-Connect detour')]:
        ax.plot(*points.T,color=color,label=label)
    u,v=np.mgrid[0:2*np.pi:32j,0:np.pi:16j]
    c=np.array(obstacle['obstacle_center']); r=obstacle['radius']
    ax.plot_surface(c[0]+r*np.cos(u)*np.sin(v),c[1]+r*np.sin(u)*np.sin(v),c[2]+r*np.cos(v),color='red',alpha=.4)
    ax.set(xlabel='X (m)',ylabel='Y (m)',zlabel='Z (m)',title='Deliberate obstacle test: kinematic planning\nCollision validation uses all active robot geometries')
    ax.legend()
    fig.tight_layout()
    fig.savefig(root/'detour.png',dpi=160)
    plt.close(fig)


if __name__=='__main__':
    main()
