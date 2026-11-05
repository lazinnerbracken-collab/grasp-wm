"""Panda + French's mustard: pose IK and force-based finger impedance."""
from pathlib import Path
import copy
import json
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from lift_metrics import LiftMonitor, assess_lift

HOME = np.array([0., -.4, 0., -2., 0., 1.6, .785398])
TABLE_Z = .4
OBJECT_XY = np.array([.48, 0.])
DOWN = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]])

def build_scene(root="assets", output="scene.xml"):
    root = Path(root).resolve()
    tree = ET.parse(root / "panda/panda.xml")
    xml = tree.getroot()
    xml.set("model", "Panda Frenchs laboratory")
    xml.find("compiler").attrib.pop("meshdir", None)
    for mesh in xml.findall("./asset/mesh"):
        mesh.set("file", (root / "panda/assets" / mesh.get("file")).as_posix())
    xml.find("option").attrib.update(timestep="0.002", gravity="0 0 -9.81", integrator="implicitfast",
                                     cone="elliptic", impratio="10", noslip_iterations="20")
    for tag in ["actuator", "tendon", "keyframe"]:
        node = xml.find(tag)
        if node is not None:
            xml.remove(node)
    actuator = ET.SubElement(xml, "actuator")
    for i, limit in enumerate([87, 87, 87, 87, 12, 12, 12], 1):
        ET.SubElement(actuator, "motor", name=f"arm_motor{i}", joint=f"joint{i}",
                      ctrlrange=f"{-limit} {limit}", gear="1")
    for i in [1, 2]:
        ET.SubElement(actuator, "motor", name=f"finger_motor{i}", joint=f"finger_joint{i}",
                      ctrlrange="-15 15", gear="1")
    world = xml.find("worldbody")
    world.find("body[@name='link0']").set("pos", f"0 0 {TABLE_Z}")
    hand = xml.find(".//body[@name='hand']")
    ET.SubElement(hand, "site", name="grasp_center", pos="0 0 0.1", size="0.004", rgba="0 1 1 1", group="4")
    # RGB axes are visual sites, so they do not change inertia or collisions.
    for axis, end, color in [("x", ".06 0 .1", "1 0 0 1"), ("y", "0 .06 .1", "0 1 0 1"),
                             ("z", "0 0 .16", "0 .3 1 1")]:
        ET.SubElement(hand, "site", name=f"tcp_{axis}", type="cylinder",
                      fromto=f"0 0 .1 {end}", size=".0015", rgba=color, group="4")
    ET.SubElement(world, "geom", name="floor", type="plane", size="2 2 .1", rgba=".18 .22 .28 1")
    ET.SubElement(world, "geom", name="table", type="box", pos=".3 0 .375", size=".7 .5 .025",
                  rgba=".62 .47 .32 1", friction="1 .005 .0001")
    for x in [-.3, .9]:
        for y in [-.4, .4]:
            ET.SubElement(world, "geom", type="box", pos=f"{x} {y} .175", size=".025 .025 .175",
                          rgba=".25 .27 .3 1")
    source = ET.parse(root / "mustard/source.xml").getroot()
    for asset in source.find("asset"):
        asset = copy.deepcopy(asset)
        if asset.get("file"):
            asset.set("file", (root / "mustard" / asset.get("file")).as_posix())
        xml.find("asset").append(asset)
    obj = copy.deepcopy(source.find("worldbody/body"))
    # Mesh is already upright (z is its long axis); keep real dimensions.
    vertices = np.array([list(map(float, line.split()[1:4])) for line in
                         (root / "mustard/mustard_bottle/textured.obj").read_text().splitlines()
                         if line.startswith("v ")])
    object_z = TABLE_Z - vertices[:, 2].min() + .001
    obj.set("pos", f"{OBJECT_XY[0]} {OBJECT_XY[1]} {object_z}")
    for geom in obj.findall("geom"):
        geom.set("friction", "1.2 .01 .001")
        geom.set("condim", "4")
    world.append(obj)
    ET.SubElement(world, "light", pos="1 -1 2.5", dir="-.3 .3 -1", diffuse=".7 .7 .7")
    visual = ET.SubElement(xml, "visual")
    ET.SubElement(visual, "global", offwidth="960", offheight="720")
    ET.indent(tree)
    tree.write(output, encoding="utf-8", xml_declaration=True)
    return Path(output)

class PandaLab:
    def __init__(self, scene="scene.xml"):
        self.model = m = mujoco.MjModel.from_xml_path(str(scene))
        self.data = d = mujoco.MjData(m)
        arm_j = np.array([m.joint(f"joint{i}").id for i in range(1, 8)])
        finger_j = np.array([m.joint(f"finger_joint{i}").id for i in [1, 2]])
        self.aq, self.av = m.jnt_qposadr[arm_j], m.jnt_dofadr[arm_j]
        self.fq, self.fv = m.jnt_qposadr[finger_j], m.jnt_dofadr[finger_j]
        self.au = np.array([m.actuator(f"arm_motor{i}").id for i in range(1, 8)])
        self.fu = np.array([m.actuator(f"finger_motor{i}").id for i in [1, 2]])
        self.bounds = m.jnt_range[arm_j].copy()
        self.site = m.site("grasp_center").id
        self.object_body = m.body("mustard_bottle").id
        self.lift_monitor = LiftMonitor(m, d)
        self.finger_bodies = [m.body(n).id for n in ["left_finger", "right_finger"]]
        self.kp_arm = np.array([400., 400., 350., 350., 180., 150., 100.])
        self.kd_arm = np.array([40., 40., 35., 35., 15., 12., 10.])
        self.k_finger, self.d_finger, self.force_limit = 400., 8., 12.
        self.log = []
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.aq] = HOME
        self.data.qpos[self.fq] = .04
        mujoco.mj_forward(self.model, self.data)
        self.q_ref = HOME.copy()
        self.log = []

    def pose(self, data=None):
        d = self.data if data is None else data
        return d.site_xpos[self.site].copy(), d.site_xmat[self.site].reshape(3, 3).copy()

    def transform(self):
        p, r = self.pose()
        transform = np.eye(4)
        transform[:3, :3], transform[:3, 3] = r, p
        return transform

    def ik(self, position, rotation, seed=None, max_iterations=500):
        """6D damped least squares, world-frame Jacobian, bounded joints.

        Returns a configuration, never writes simulation qpos. Raises on failure.
        This is pose IK, not a collision-avoiding motion planner.
        """
        position, rotation = np.asarray(position, float), np.asarray(rotation, float)
        if position.shape != (3,) or rotation.shape != (3, 3) or not np.isfinite(position).all():
            raise ValueError("Expected position (3,) and rotation (3,3).")
        if not np.isfinite(rotation).all() or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(rotation), 1):
            raise ValueError("rotation must be a proper rotation matrix")
        m = self.model
        work = mujoco.MjData(m)
        work.qpos[:] = self.data.qpos
        q = self.data.qpos[self.aq].copy() if seed is None else np.array(seed, float).copy()
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for iteration in range(max_iterations):
            work.qpos[self.aq] = q
            mujoco.mj_forward(m, work)
            p, r = self.pose(work)
            ep = position - p
            er = Rotation.from_matrix(rotation @ r.T).as_rotvec()
            if np.linalg.norm(ep) < 2e-4 and np.linalg.norm(er) < 2e-3:
                return q
            mujoco.mj_jacSite(m, work, jp, jr, self.site)
            # 0.3 m/rad gives position and orientation comparable weights.
            jac = np.vstack([jp[:, self.av], .3 * jr[:, self.av]])
            err = np.r_[ep, .3 * er]
            delta = jac.T @ np.linalg.solve(jac @ jac.T + .02**2 * np.eye(6), err)
            delta *= min(1., .12 / max(np.linalg.norm(delta), 1e-12))
            q = np.clip(q + delta, self.bounds[:, 0] + .005, self.bounds[:, 1] - .005)
        raise RuntimeError(f"IK failed: position {np.linalg.norm(ep):.4f} m, rotation {np.linalg.norm(er):.4f} rad. Choose another pose.")

    def control(self, q_desired, finger_desired):
        """Arm torque PD + bias compensation; finger spring-damper in N."""
        m, d = self.model, self.data
        tau = self.kp_arm * (q_desired - d.qpos[self.aq]) - self.kd_arm * d.qvel[self.av] + d.qfrc_bias[self.av]
        d.ctrl[self.au] = np.clip(tau, m.actuator_ctrlrange[self.au, 0], m.actuator_ctrlrange[self.au, 1])
        # finger_desired is EACH finger displacement [m], not total opening.
        force = self.k_finger * (finger_desired - d.qpos[self.fq]) - self.d_finger * d.qvel[self.fv]
        d.ctrl[self.fu] = np.clip(force, -self.force_limit, self.force_limit)
        return d.ctrl[self.fu].copy()

    def contact_forces(self):
        """Sum normal contact forces [N] on each finger against the bottle."""
        result = np.zeros(2)
        wrench = np.zeros(6)
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            bodies = [self.model.geom_bodyid[contact.geom1], self.model.geom_bodyid[contact.geom2]]
            if self.object_body not in bodies:
                continue
            mujoco.mj_contactForce(self.model, self.data, index, wrench)
            for finger, body in enumerate(self.finger_bodies):
                if body in bodies:
                    result[finger] += max(wrench[0], 0.)
        return result

    def segment(self, q_target, duration, finger_start=.04, finger_end=.04, capture=None, phase="move"):
        q_start = self.q_ref.copy()
        steps = int(round(duration / self.model.opt.timestep))
        for step in range(steps):
            u = (step + 1) / steps
            blend = 10*u**3 - 15*u**4 + 6*u**5
            q_desired = q_start + blend * (q_target - q_start)
            f_desired = finger_start + blend * (finger_end - finger_start)
            force = self.control(q_desired, f_desired)
            mujoco.mj_step(self.model, self.data)
            if not np.isfinite(self.data.qpos).all():
                raise RuntimeError("Non-finite simulation state")
            if step % 20 == 0:
                mujoco.mj_forward(self.model, self.data)
                self.log.append(dict(time=float(self.data.time), phase=phase,
                    opening=float(self.data.qpos[self.fq].sum()), force=force.tolist(),
                    contact=self.contact_forces().tolist(), tcp=self.pose()[0].tolist(),
                    **self.lift_monitor.sample(*self.pose())))
            if capture is not None:
                capture(self)
        self.q_ref = np.array(q_target).copy()
        mujoco.mj_forward(self.model, self.data)

    def reach_pose(self, position, rotation, duration=3., capture=None):
        target = self.ik(position, rotation, seed=self.q_ref)
        self.segment(target, duration, capture=capture)
        self.segment(target, .5, capture=capture, phase="settle")
        p, r = self.pose()
        error = {"position_m": float(np.linalg.norm(np.array(position) - p)),
                 "orientation_rad": float(Rotation.from_matrix(np.array(rotation) @ r.T).magnitude())}
        if error["position_m"] > .01 or error["orientation_rad"] > .08:
            raise RuntimeError(f"Tracking failed (collision or torque saturation possible): {error}")
        return error

    def demo(self, capture=None):
        self.reset()
        self.segment(HOME, .5, capture=capture, phase="settle")
        grasp = np.r_[OBJECT_XY, TABLE_Z + .16]
        pregrasp = grasp + [0., 0., .20]
        self.reach_pose(pregrasp, DOWN, 3., capture)
        error = self.reach_pose(grasp, DOWN, 2., capture)
        self.segment(self.q_ref, 1.5, .04, 0., capture, "close")
        self.segment(self.q_ref, 1., 0., 0., capture, "hold")
        contacts = self.contact_forces()
        result = dict(pose_error=error, opening_m=float(self.data.qpos[self.fq].sum()),
                      finger_contact_N=contacts.tolist(), bilateral_contact=bool(np.all(contacts > .1)))
        if not result['bilateral_contact']:
            raise RuntimeError('No bilateral grasp: lift aborted.')
        before = self.lift_monitor.sample(*self.pose())
        # Keep the impedance closure active throughout lifting and holding.
        lift_target = grasp + [0., 0., .15]
        q_lift = self.ik(lift_target, DOWN, seed=self.q_ref)
        self.segment(q_lift, 4., 0., 0., capture, 'lift')
        self.segment(q_lift, 3., 0., 0., capture, 'hold_lift')
        final = self.lift_monitor.sample(*self.pose())
        result.update(assess_lift(before, [row for row in self.log if row['phase']=='hold_lift'], final))
        result['final_contact_N'] = self.contact_forces().tolist()
        result['final_tcp_error_m'] = float(np.linalg.norm(self.pose()[0]-lift_target))
        result["warnings"] = [int(w.number) for w in self.data.warning]
        return result

class Recorder:
    """Offscreen frames written directly to MP4 to avoid exhausting Colab RAM."""
    def __init__(self, model, path="panda_demo.mp4", fps=25):
        import imageio.v2 as imageio
        self.renderer = mujoco.Renderer(model, height=480, width=640)
        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = [.35, 0., .65]
        self.camera.distance, self.camera.azimuth, self.camera.elevation = 1.7, 135, -25
        self.option = mujoco.MjvOption()
        self.option.geomgroup[3] = 0  # Hide collision proxies, keep textured visuals.
        self.option.sitegroup[4] = 1
        self.writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=7)
        self.next_frame, self.fps = 0., fps

    def __call__(self, lab):
        if lab.data.time + 1e-9 >= self.next_frame:
            mujoco.mj_forward(lab.model, lab.data)
            self.renderer.update_scene(lab.data, camera=self.camera, scene_option=self.option)
            self.writer.append_data(self.renderer.render())
            self.next_frame += 1 / self.fps

    def close(self):
        self.writer.close()
        self.renderer.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    build_scene()
    lab = PandaLab()
    recorder = Recorder(lab.model) if args.video else None
    try:
        result = lab.demo(recorder)
    finally:
        if recorder:
            recorder.close()
    Path("validation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
