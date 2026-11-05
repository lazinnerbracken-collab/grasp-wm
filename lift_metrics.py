"""Measure a real free-body lift independently of the gripper target."""
import numpy as np
import mujoco

class LiftMonitor:
    def __init__(self, model, data, object_name='mustard_bottle', table_name='table'):
        self.model, self.data = model, data
        self.body = model.body(object_name).id
        self.table = model.geom(table_name).id
        # Use the rendered bottle mesh, including the cap, for bottom clearance.
        self.visual = model.geom('mustard_bottle_visual_geom').id
        mesh = model.geom_dataid[self.visual]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        self.vertices = model.mesh_vert[start:start+count].copy()
        self.table_top = float(model.geom_pos[self.table,2] + model.geom_size[self.table,2])

    def sample(self, tcp_position, tcp_rotation):
        m,d = self.model,self.data
        rotation = d.geom_xmat[self.visual].reshape(3,3)
        bottom = float(np.min(self.vertices @ rotation[2,:]) + d.geom_xpos[self.visual,2])
        table_contact = False
        for c in d.contact:
            if self.table in (c.geom1,c.geom2):
                other = c.geom2 if c.geom1 == self.table else c.geom1
                if m.geom_bodyid[other] == self.body:
                    table_contact = True
        return dict(object_z_m=float(d.xpos[self.body,2]),
                    clearance_m=bottom-self.table_top,
                    object_in_tcp=(tcp_rotation.T @ (d.xpos[self.body]-tcp_position)).tolist(),
                    table_contact=table_contact)

def assess_lift(before, hold, final, *, required_rise=.10):
    """Require rise, table separation, persistent contact and a stable 3 s hold.

    hold contains timestamped samples, each with telemetry and contact forces.
    Heights are measured on the OBJECT, never inferred from a joint/TCP target.
    """
    if len(hold) < 2:
        raise ValueError('Need hold samples to evaluate lift stability.')
    offsets = np.array([row['object_in_tcp'] for row in hold])
    drift = float(np.max(np.linalg.norm(offsets-offsets[0],axis=1)))
    clearance = min(row['clearance_m'] for row in hold)
    rise = final['object_z_m']-before['object_z_m']
    separation = not any(row['table_contact'] for row in hold) and not final['table_contact']
    contact_fraction = float(np.mean([np.all(np.array(row['contact'])>.1) for row in hold]))
    duration = hold[-1]['time']-hold[0]['time']
    slip = float(np.linalg.norm(np.array(final['object_in_tcp'])-before['object_in_tcp']))
    success = (rise >= required_rise and clearance >= .08 and separation and
               contact_fraction >= .99 and drift <= .01 and duration >= 2.9)
    return dict(lift_success=bool(success), object_lift_m=float(rise),
                final_clearance_m=float(final['clearance_m']),
                min_hold_clearance_m=float(clearance), hold_relative_drift_m=drift,
                total_relative_slip_m=slip, hold_bilateral_contact_fraction=contact_fraction,
                hold_duration_measured_s=float(duration), table_separated=bool(separation))
