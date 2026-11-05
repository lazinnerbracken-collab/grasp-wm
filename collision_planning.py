"""MuJoCo collision checks and a small bidirectional RRT-Connect planner.

This is our NumPy implementation, not an OMPL binding. Validity follows the
MJCF's collision masks/exclusions. Motion checking is discretized, not continuous.
"""
from dataclasses import dataclass
import math
import time
import numpy as np
import mujoco


class PandaCollisionChecker:
    def __init__(self, scene, reference_qpos, clearance=.002, resolution=.01):
        if not np.isfinite([clearance,resolution]).all() or clearance < 0 or resolution <= 0:
            raise ValueError('Invalid clearance/resolution')
        self.model = m = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(m)
        self.data.qpos[:] = reference_qpos
        self.arm = np.array([m.jnt_qposadr[m.joint(f'joint{i}').id] for i in range(1,8)])
        self.fingers = np.array([m.jnt_qposadr[m.joint(f'finger_joint{i}').id] for i in (1,2)])
        self.bounds = np.array([m.joint(f'joint{i}').range for i in range(1,8)])
        self.clearance, self.resolution = clearance, resolution
        self.robot = {m.body('link0').id}
        for body in range(m.nbody):
            if int(m.body_parentid[body]) in self.robot:
                self.robot.add(body)
        self.base = m.body('link0').id
        self.table = m.geom('table').id
        self.object = m.body('mustard_bottle').id
        self.calls = 0
        # An independent model: margins here never change simulation dynamics.
        active = (m.geom_contype != 0) | (m.geom_conaffinity != 0)
        m.geom_margin[active] = np.maximum(m.geom_margin[active], clearance + 1e-6)
        self.data.qpos[self.fingers] = .04

    def explain(self, q):
        self.calls += 1
        q = np.asarray(q)
        if q.shape != (7,) or not np.isfinite(q).all():
            return [{'kind':'invalid_configuration'}]
        if np.any(q < self.bounds[:,0]) or np.any(q > self.bounds[:,1]):
            return [{'kind':'joint_limit'}]
        self.data.qpos[self.arm] = q
        mujoco.mj_forward(self.model, self.data)
        issues = []
        for contact in self.data.contact:
            g1,g2 = int(contact.geom1),int(contact.geom2)
            b1,b2 = int(self.model.geom_bodyid[g1]),int(self.model.geom_bodyid[g2])
            if not ({b1,b2} & self.robot):
                continue
            # Only the fixed robot base/table mounting pair is permitted.
            if (b1 == self.base and g2 == self.table) or (b2 == self.base and g1 == self.table):
                continue
            if contact.dist >= self.clearance:
                continue
            kind = 'self_collision' if b1 in self.robot and b2 in self.robot else 'robot_environment'
            if self.object in (b1,b2):
                kind = 'robot_object'
            issues.append(dict(kind=kind,
                geom1=self.model.geom(g1).name or f'geom_{g1}',
                geom2=self.model.geom(g2).name or f'geom_{g2}',
                body1=self.model.body(b1).name, body2=self.model.body(b2).name,
                signed_distance_m=float(contact.dist)))
        return issues

    def valid(self, q):
        return not self.explain(q)

    def edge(self, a, b):
        n = max(1, int(np.ceil(np.max(np.abs(np.asarray(b)-a))/self.resolution)))
        for fraction in np.linspace(0,1,n+1):
            q = (1-fraction)*np.asarray(a)+fraction*np.asarray(b)
            issues = self.explain(q)
            if issues:
                return False, dict(fraction=float(fraction),configuration=q.tolist(),contacts=issues)
        return True, None


@dataclass
class Plan:
    path: np.ndarray | None
    status: str
    iterations: int
    seconds: float


def rrt_connect(start, goal, bounds, edge_valid, *, seed=0, step=.25,
                iterations=1500, timeout=10., shortcuts=30):
    """Grow alternating trees and connect them using checked joint-space edges."""
    if step <= 0 or timeout <= 0 or iterations < 1:
        raise ValueError('Invalid planning budget')
    start,goal,bounds = np.asarray(start,float),np.asarray(goal,float),np.asarray(bounds,float)
    if start.shape != goal.shape or bounds.shape != (len(start),2):
        raise ValueError('Invalid joint space')
    t0=time.perf_counter()
    if not edge_valid(start,start) or not edge_valid(goal,goal):
        return Plan(None,'invalid_endpoint',0,time.perf_counter()-t0)
    if edge_valid(start,goal):
        return Plan(np.array([start,goal]),'direct',0,time.perf_counter()-t0)
    rng=np.random.default_rng(seed)
    trees=[([start.copy()],[-1]),([goal.copy()],[-1])]
    def grow(tree,target):
        nodes,parents=tree
        near=int(np.argmin(np.linalg.norm(np.asarray(nodes)-target,axis=1)))
        delta=target-nodes[near]
        norm=np.linalg.norm(delta)
        if norm < 1e-10:
            return near,True
        new=nodes[near]+min(1.,step/norm)*delta
        if not edge_valid(nodes[near],new):
            return None,False
        nodes.append(new)
        parents.append(near)
        return len(nodes)-1,norm <= step
    def branch(tree,index):
        nodes,parents=tree
        path=[]
        while index >= 0:
            path.append(nodes[index])
            index=parents[index]
        return path[::-1]
    for iteration in range(1,iterations+1):
        if time.perf_counter()-t0 > timeout:
            return Plan(None,'timeout',iteration-1,time.perf_counter()-t0)
        side=iteration%2
        target=rng.uniform(bounds[:,0],bounds[:,1])
        if rng.random()<.15:
            target=trees[1-side][0][0]
        index,_=grow(trees[side],target)
        if index is None:
            continue
        target=trees[side][0][index]
        other=None
        while time.perf_counter()-t0 < timeout:
            other,reached=grow(trees[1-side],target)
            if other is None or reached:
                break
        if other is None or not reached:
            continue
        a,b=branch(trees[side],index),branch(trees[1-side],other)
        path=a+b[-2::-1] if side==0 else b+a[-2::-1]
        for _ in range(shortcuts):
            if len(path)<3 or time.perf_counter()-t0 >= timeout:
                break
            i,j=sorted(rng.choice(len(path),2,replace=False))
            if j>i+1 and edge_valid(path[i],path[j]):
                path=path[:i+1]+path[j:]
        return Plan(np.array(path),'rrt_connect',iteration,time.perf_counter()-t0)
    return Plan(None,'iteration_limit',iterations,time.perf_counter()-t0)
