"""Coal collision queries on the compiled Panda MJCF geometry.

MuJoCo supplies forward kinematics only. Coal performs all narrow-phase queries.
The simulation and its state are separate. Meshes use convex hulls, matching the
existing MuJoCo collision representation rather than visual triangle surfaces.
"""
import numpy as np
import mujoco
import coal
from collision_planning import PandaCollisionChecker


class CoalCollisionChecker(PandaCollisionChecker):
    def __init__(self, scene, reference_qpos, clearance=.002, resolution=.01):
        super().__init__(scene, reference_qpos, clearance, resolution)
        m = self.model
        if m.npair or m.nflex:
            raise ValueError('Coal bridge does not support explicit contact pairs or flexes')
        self.shapes = {}
        self.mesh_cache = {}
        active = np.flatnonzero((m.geom_contype != 0) | (m.geom_conaffinity != 0))
        for g in active:
            self.shapes[int(g)] = self._shape(int(g))
        excluded = set(map(int, m.exclude_signature))
        parent_filter = not (m.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_FILTERPARENT))
        pairs = []
        for i, a in enumerate(active):
            for b in active[i+1:]:
                ba, bb = int(m.geom_bodyid[a]), int(m.geom_bodyid[b])
                if not ({ba, bb} & self.robot):
                    continue
                if (ba == self.base and b == self.table) or (bb == self.base and a == self.table):
                    continue
                if (min(ba,bb) << 16) + max(ba,bb) in excluded:
                    continue
                wa, wb = int(m.body_weldid[ba]), int(m.body_weldid[bb])
                if wa == wb:
                    continue
                pa = int(m.body_weldid[m.body_parentid[wa]])
                pb = int(m.body_weldid[m.body_parentid[wb]])
                if parent_filter and wa and wb and (wa == pb or wb == pa):
                    continue
                if not ((m.geom_contype[a] & m.geom_conaffinity[b]) or
                        (m.geom_contype[b] & m.geom_conaffinity[a])):
                    continue
                pairs.append((int(a),int(b)))
        self.pairs = np.array(pairs, dtype=int).reshape(-1,2)
        self.request = coal.CollisionRequest()
        self.request.security_margin = clearance
        self.request.enable_contact = True
        self.request.num_max_contacts = 1
        self.narrow_phase_calls = 0

    def _shape(self, g):
        m = self.model
        kind, s = int(m.geom_type[g]), m.geom_size[g]
        types = mujoco.mjtGeom
        if kind == types.mjGEOM_BOX:
            return coal.Box(*(2*s))
        if kind == types.mjGEOM_SPHERE:
            return coal.Sphere(float(s[0]))
        if kind == types.mjGEOM_CAPSULE:
            return coal.Capsule(float(s[0]),float(2*s[1]))
        if kind == types.mjGEOM_CYLINDER:
            return coal.Cylinder(float(s[0]),float(2*s[1]))
        if kind == types.mjGEOM_ELLIPSOID:
            return coal.Ellipsoid(*s)
        if kind == types.mjGEOM_PLANE:
            return coal.Halfspace(np.array([0.,0.,1.]),0.)
        if kind == types.mjGEOM_MESH:
            mesh = int(m.geom_dataid[g])
            if mesh not in self.mesh_cache:
                va, vn = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
                fa, fn = m.mesh_faceadr[mesh], m.mesh_facenum[mesh]
                vertices = np.asarray(m.mesh_vert[va:va+vn],dtype=float)
                faces = np.asarray(m.mesh_face[fa:fa+fn],dtype=np.int32)
                bvh = coal.BVHModelOBBRSS()
                bvh.beginModel(int(fn),int(vn))
                coal_vertices,coal_faces = coal.StdVec_Vec3s(),coal.StdVec_Triangle()
                for v in vertices:
                    coal_vertices.append(v)
                for f in faces:
                    coal_faces.append(coal.Triangle(*map(int,f)))
                bvh.addSubModel(coal_vertices, coal_faces)
                bvh.endModel()
                bvh.buildConvexHull(True, 'Qt')
                self.mesh_cache[mesh] = bvh.convex
            return self.mesh_cache[mesh]
        raise ValueError(f'Unsupported collision geom type {kind}: {g}')

    def explain(self, q):
        self.calls += 1
        q = np.asarray(q)
        if q.shape != (7,) or not np.isfinite(q).all():
            return [{'kind':'invalid_configuration'}]
        if np.any(q < self.bounds[:,0]) or np.any(q > self.bounds[:,1]):
            return [{'kind':'joint_limit'}]
        m, d = self.model, self.data
        d.qpos[self.arm] = q
        # Does not run MuJoCo collision detection or physics.
        mujoco.mj_kinematics(m,d)
        a,b = self.pairs.T
        radii = m.geom_rbound[a] + m.geom_rbound[b] + self.clearance + 1e-8
        near = np.linalg.norm(d.geom_xpos[a]-d.geom_xpos[b],axis=1) <= radii
        near |= (m.geom_type[a] == mujoco.mjtGeom.mjGEOM_PLANE) | (m.geom_type[b] == mujoco.mjtGeom.mjGEOM_PLANE)
        transforms = {}
        issues = []
        for g1,g2 in self.pairs[near]:
            for g in (g1,g2):
                if g not in transforms:
                    transforms[g] = coal.Transform3s(d.geom_xmat[g].reshape(3,3),d.geom_xpos[g])
            result = coal.CollisionResult()
            coal.collide(self.shapes[g1],transforms[g1],self.shapes[g2],transforms[g2],self.request,result)
            self.narrow_phase_calls += 1
            if not result.isCollision():
                continue
            ba,bb = int(m.geom_bodyid[g1]),int(m.geom_bodyid[g2])
            contact = result.getContact(0)
            kind = 'self_collision' if ba in self.robot and bb in self.robot else 'robot_environment'
            if self.object in (ba,bb):
                kind = 'robot_object'
            issues.append(dict(kind=kind, geom1=m.geom(g1).name or f'geom_{g1}',
                geom2=m.geom(g2).name or f'geom_{g2}', body1=m.body(ba).name,
                body2=m.body(bb).name, signed_distance_m=float(contact.penetration_depth)))
        return issues
