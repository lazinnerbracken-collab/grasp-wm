"""NERO dual arm from the user's models: pose IK and finger impedance."""
from pathlib import Path
import copy
import json
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from lift_metrics import LiftMonitor, assess_lift

SIDES = ('right', 'left')
TABLE_Z = .40
DOWN = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]])
TCP_POS = np.array([.175, -5.28942e-7, -.0235005])
TCP_QUAT = np.array([.499998, -.5, .500002, -.5])  # wxyz, same axes as gripper_base

def build_scene(root='nero_assets', mustard='assets/mustard', output='nero_scene.xml', side='right'):
    if side not in SIDES:
        raise ValueError(side)
    root, mustard = Path(root).resolve(), Path(mustard).resolve()
    tree = ET.parse(root / 'nero_dual_arm.mpc.xml')
    xml = tree.getroot()
    xml.find('compiler').attrib.pop('meshdir', None)
    xml.find('option').attrib.update(timestep='.001', integrator='implicitfast', solver='Newton', iterations='80',
                                    cone='elliptic', impratio='10', noslip_iterations='20')
    # Sensor noise attributes in older generated MJCF are unnecessary for this TP.
    if xml.find('sensor') is not None:
        xml.remove(xml.find('sensor'))
    for mesh in xml.findall('./asset/mesh'):
        mesh.set('file', (root / 'meshes' / mesh.get('file')).as_posix())
    world = xml.find('worldbody')
    contact = ET.SubElement(xml, 'contact')
    # Touching/overlapping housings in the supplied CAD, not obstacles.
    for side_name in SIDES:
        ET.SubElement(contact, 'exclude', body1='root', body2=f'{side_name}_link1')
        ET.SubElement(contact, 'exclude', body1=f'{side_name}_link5', body2=f'{side_name}_link7')
    for geom in world.iter('geom'):
        if geom.get('rgba') == '':
            geom.attrib.pop('rgba')
        if geom.get('class') == 'visualgeom':
            geom.set('rgba', '.55 .60 .67 1')
            geom.set('group', '1')
        else:
            # Activate the collision copies supplied with the model.
            # MuJoCo uses their convex hulls: this is an approximation.
            geom.attrib.update(contype='1', conaffinity='1', group='3',
                               friction='1 .01 .001', solref='.01 1', solimp='.95 .99 .001')
    world.find("geom[@name='ground']").attrib.update(size='3 3 .1', group='0', rgba='.18 .22 .28 1')
    for side_name in SIDES:
        wrist = xml.find(f".//body[@name='{side_name}_link7']")
        ET.SubElement(wrist, 'site', name=f'{side_name}_grasp_center',
                      pos=' '.join(map(str, TCP_POS)), quat=' '.join(map(str, TCP_QUAT)),
                      size='.004', rgba='0 1 1 1', group='4')
        rotation = Rotation.from_quat(TCP_QUAT[[1, 2, 3, 0]]).as_matrix()
        for i, color in enumerate(['1 0 0 1', '0 1 0 1', '0 .3 1 1']):
            end = TCP_POS + .06 * rotation[:, i]
            ET.SubElement(wrist, 'site', type='cylinder', size='.0015', group='4', rgba=color,
                          fromto=' '.join(map(str, np.r_[TCP_POS, end])))
    ET.SubElement(world, 'geom', name='table', type='box', pos='0 -.5 .375', size='.7 .35 .025',
                  rgba='.62 .47 .32 1', contype='1', conaffinity='1', friction='1 .01 .001', group='0')
    for x in [-.6, .6]:
        for y in [-.75, -.25]:
            ET.SubElement(world, 'geom', type='box', pos=f'{x} {y} .175', size='.025 .025 .175',
                          rgba='.3 .3 .3 1', contype='1', conaffinity='1', group='0')
    source = ET.parse(mustard / 'source.xml').getroot()
    for asset in source.find('asset'):
        asset = copy.deepcopy(asset)
        if asset.get('file'):
            asset.set('file', (mustard / asset.get('file')).as_posix())
        xml.find('asset').append(asset)
    obj = copy.deepcopy(source.find('worldbody/body'))
    x = -.25 if side == 'right' else .25
    obj.set('pos', f'{x} -.45 {TABLE_Z + .07830708 + .001}')
    for geom in obj.findall('geom'):
        # Override disabled default contacts inherited from the robot XML.
        is_visual = geom.get('contype') == '0'
        geom.attrib.update(contype='0' if is_visual else '1', conaffinity='0' if is_visual else '1',
                           friction='1.2 .01 .001', solref='.01 1', solimp='.95 .99 .001')
    world.append(obj)
    visual = ET.SubElement(xml, 'visual')
    ET.SubElement(visual, 'global', offwidth='960', offheight='720')
    ET.indent(tree)
    tree.write(output, encoding='utf-8', xml_declaration=True)
    return Path(output)

class NeroLab:
    def __init__(self, scene='nero_scene.xml'):
        self.model = m = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(m)
        self.arms = {}
        for side in SIDES:
            names = [f'{side}_joint{i}' for i in range(1, 8)]
            fingers = [f'{side}_gripper_joint{i}' for i in [1, 2]]
            joint_ids = np.array([m.joint(n).id for n in names])
            fids = np.array([m.joint(n).id for n in fingers])
            self.arms[side] = dict(q=m.jnt_qposadr[joint_ids], v=m.jnt_dofadr[joint_ids],
                u=np.array([m.actuator(n).id for n in names]), limits=m.jnt_range[joint_ids],
                fq=m.jnt_qposadr[fids], fv=m.jnt_dofadr[fids],
                fu=np.array([m.actuator(n).id for n in fingers]), site=m.site(f'{side}_grasp_center').id,
                fingers=[m.body(f'{side}_gripper_link{i}').id for i in [1, 2]])
        self.object_body = m.body('mustard_bottle').id
        self.lift_monitor = LiftMonitor(m, self.data)
        self.kp = np.array([160., 160., 140., 140., 80., 60., 45.])
        self.kd = np.array([18., 18., 16., 16., 8., 6., 5.])
        self.k_finger, self.d_finger, self.force_limit = 300., 5., 8.
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.refs = {side: np.zeros(7) for side in SIDES}
        self.openings = {side: .10 for side in SIDES}
        for arm in self.arms.values():
            self.data.qpos[arm['fq']] = [.05, -.05]
        mujoco.mj_forward(self.model, self.data)
        self.log = []
        self.unexpected_contacts = set()

    def pose(self, side, data=None):
        d = self.data if data is None else data
        site = self.arms[side]['site']
        return d.site_xpos[site].copy(), d.site_xmat[site].reshape(3, 3).copy()

    def transform(self, side):
        p, r = self.pose(side)
        result = np.eye(4)
        result[:3, :3], result[:3, 3] = r, p
        return result

    def ik(self, side, position, rotation, seed=None):
        """Bounded 6D numerical IK; does not modify simulation state."""
        p, r = np.asarray(position, float), np.asarray(rotation, float)
        if p.shape != (3,) or r.shape != (3, 3) or not np.isfinite(p).all() or not np.isfinite(r).all():
            raise ValueError('Expected finite position (3,) and rotation (3,3).')
        if not np.allclose(r.T @ r, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(r), 1):
            raise ValueError('Expected a proper rotation matrix.')
        arm = self.arms[side]
        work = mujoco.MjData(self.model)
        work.qpos[:] = self.data.qpos
        def residual(q):
            work.qpos[arm['q']] = q
            mujoco.mj_forward(self.model, work)
            actual_p, actual_r = self.pose(side, work)
            return np.r_[actual_p-p, .3*Rotation.from_matrix(r @ actual_r.T).as_rotvec()]
        low, high = arm['limits'][:, 0]+.002, arm['limits'][:, 1]-.002
        seed = self.refs[side] if seed is None else np.asarray(seed)
        seeds = [np.clip(seed, low, high)]
        rng = np.random.default_rng(23)
        seeds += [rng.uniform(low*.7, high*.7) for _ in range(10)]
        best = np.inf
        for start in seeds:
            answer = least_squares(residual, start, bounds=(low, high), max_nfev=180,
                                   ftol=1e-9, xtol=1e-9, gtol=1e-9)
            e = residual(answer.x)
            best = min(best, np.linalg.norm(e))
            if np.linalg.norm(e[:3]) < 2e-4 and np.linalg.norm(e[3:])/.3 < .002:
                return answer.x
        raise RuntimeError(f'IK {side} impossible for this pose (weighted residual {best:.4g}).')

    def control(self):
        if not 0 < self.force_limit <= 10:
            raise ValueError('Finger force_limit must be in (0,10] N, as in the source model.')
        m, d = self.model, self.data
        for side, a in self.arms.items():
            tau = self.kp*(self.refs[side]-d.qpos[a['q']])-self.kd*d.qvel[a['v']]+d.qfrc_bias[a['v']]
            d.ctrl[a['u']] = np.clip(tau, -16, 16)
            # Source model: positive slide 1, NEGATIVE slide 2 when open.
            target = np.array([.5, -.5]) * self.openings[side]
            force = self.k_finger*(target-d.qpos[a['fq']])-self.d_finger*d.qvel[a['fv']]
            d.ctrl[a['fu']] = np.clip(force, -self.force_limit, self.force_limit)

    def contact_forces(self, side):
        forces, wrench = np.zeros(2), np.zeros(6)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            bodies = [self.model.geom_bodyid[c.geom1], self.model.geom_bodyid[c.geom2]]
            if self.object_body not in bodies:
                continue
            mujoco.mj_contactForce(self.model, self.data, i, wrench)
            for j, body in enumerate(self.arms[side]['fingers']):
                if body in bodies:
                    forces[j] += max(0., wrench[0])
        return forces

    def segment(self, targets=None, openings=None, duration=2., capture=None, phase='move'):
        targets = {} if targets is None else targets
        openings = {} if openings is None else openings
        if any(not 0 <= w <= .10 for w in openings.values()):
            raise ValueError('Total opening must be in [0, 0.10] m.')
        starts = {s:q.copy() for s,q in self.refs.items()}
        initial_openings = self.openings.copy()
        steps = max(1, round(duration/self.model.opt.timestep))
        for step in range(steps):
            u = (step+1)/steps
            blend = 10*u**3-15*u**4+6*u**5
            for side, target in targets.items():
                self.refs[side] = starts[side] + blend*(target-starts[side])
            for side, target in openings.items():
                self.openings[side] = initial_openings[side]+blend*(target-initial_openings[side])
            self.control()
            mujoco.mj_step(self.model, self.data)
            # Record obstacle contacts beyond the intended pot/table/finger contacts.
            for contact in self.data.contact:
                if contact.dist >= -.001:
                    continue
                b1, b2 = [self.model.geom_bodyid[g] for g in [contact.geom1, contact.geom2]]
                names = [self.model.body(b).name for b in [b1, b2]]
                intended = (self.object_body in [b1, b2] and
                            any(n == 'world' or 'gripper_link' in n for n in names))
                if not intended:
                    self.unexpected_contacts.add(tuple(sorted(names)))
            if not np.isfinite(self.data.qpos).all():
                raise RuntimeError('Unstable simulation')
            if step % 40 == 0:
                mujoco.mj_forward(self.model, self.data)
                row = dict(time=float(self.data.time), phase=phase)
                for side, a in self.arms.items():
                    qf = self.data.qpos[a['fq']]
                    row[side] = dict(opening=float(qf[0]-qf[1]), contact=self.contact_forces(side).tolist(),
                                     force=self.data.ctrl[a['fu']].tolist(),
                                     **self.lift_monitor.sample(*self.pose(side)))
                self.log.append(row)
            if capture is not None:
                capture(self)
        mujoco.mj_forward(self.model, self.data)

    def reach_poses(self, poses, duration=3., capture=None):
        targets = {side:self.ik(side, *pose) for side,pose in poses.items()}
        self.segment(targets=targets, duration=duration, capture=capture)
        self.segment(duration=.6, capture=capture, phase='settle')
        errors = {}
        for side, (p, r) in poses.items():
            actual_p, actual_r = self.pose(side)
            e = dict(position_m=float(np.linalg.norm(actual_p-p)),
                     orientation_rad=float(Rotation.from_matrix(r @ actual_r.T).magnitude()))
            errors[side] = e
            if e['position_m'] > .01 or e['orientation_rad'] > .08:
                raise RuntimeError(f'Tracking failed for {side}: {e}')
        return errors

    def demo(self, side='right', capture=None):
        self.reset()
        self.segment(duration=.5, capture=capture, phase='settle')
        object_pos = self.data.xpos[self.object_body].copy()
        grasp = np.array([object_pos[0], object_pos[1], TABLE_Z+.16])
        standby = np.array([.45 if side=='right' else -.45, -.30, .8])
        other = 'left' if side=='right' else 'right'
        self.reach_poses({side:(grasp+[0,0,.2], DOWN), other:(standby, DOWN)}, 4., capture)
        errors = self.reach_poses({side:(grasp, DOWN)}, 2.5, capture)
        self.segment(openings={side:0.}, duration=1.5, capture=capture, phase='close')
        self.segment(duration=1., capture=capture, phase='hold')
        forces = self.contact_forces(side)
        a = self.arms[side]
        qf = self.data.qpos[a['fq']]
        result = dict(arm=side, pose_error=errors, contact_N=forces.tolist(),
                      bilateral_contact=bool(np.all(forces > .1)), opening_m=float(qf[0]-qf[1]))
        if not result['bilateral_contact']:
            raise RuntimeError('No bilateral grasp: lift aborted.')
        before = self.lift_monitor.sample(*self.pose(side))
        lift_target = grasp + [0,0,.15]
        q_lift = self.ik(side, lift_target, DOWN)
        self.segment(targets={side:q_lift}, duration=4., capture=capture, phase='lift')
        self.segment(duration=3., capture=capture, phase='hold_lift')
        hold = [dict(time=row['time'], **row[side]) for row in self.log if row['phase']=='hold_lift']
        result.update(assess_lift(before, hold, self.lift_monitor.sample(*self.pose(side))))
        result.update(final_contact_N=self.contact_forces(side).tolist(),
                    final_tcp_error_m=float(np.linalg.norm(self.pose(side)[0]-lift_target)),
                    unexpected_contacts_over_1mm=sorted(self.unexpected_contacts),
                    warnings=[int(w.number) for w in self.data.warning])
        return result

class Recorder:
    def __init__(self, model, path='nero_demo.mp4', fps=25):
        import imageio.v2 as imageio
        self.renderer = mujoco.Renderer(model, height=480, width=640)
        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = [0., -.25, .65]
        self.camera.distance, self.camera.azimuth, self.camera.elevation = 2.8, 90, -20
        self.options = mujoco.MjvOption()
        self.options.geomgroup[3] = 0
        self.options.sitegroup[4] = 1
        self.writer = imageio.get_writer(path, fps=fps, codec='libx264', quality=7)
        self.next_frame, self.fps = 0., fps

    def __call__(self, lab):
        if lab.data.time+1e-9 >= self.next_frame:
            mujoco.mj_forward(lab.model, lab.data)
            self.renderer.update_scene(lab.data, camera=self.camera, scene_option=self.options)
            self.writer.append_data(self.renderer.render())
            self.next_frame += 1/self.fps

    def close(self):
        self.writer.close()
        self.renderer.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--side', choices=SIDES, default='right')
    args = parser.parse_args()
    build_scene(side=args.side)
    lab = NeroLab()
    recorder = Recorder(lab.model) if args.video else None
    try:
        result = lab.demo(args.side, recorder)
    finally:
        if recorder:
            recorder.close()
    Path('nero_validation.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
