"""Integration checks for the real Coal backend; run in coal_env."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import coal
from coal_collision import CoalCollisionChecker
from collision_planning import PandaCollisionChecker
from panda_lab import PandaLab, HOME


class CoalTests(unittest.TestCase):
    def test_security_margin_and_signed_distance(self):
        sphere=coal.Sphere(.1)
        request=coal.CollisionRequest()
        request.security_margin=.002
        request.enable_contact=True
        for separation,expected in [(.201,True),(.203,False),(.19,True)]:
            result=coal.CollisionResult()
            coal.collide(sphere,coal.Transform3s(),sphere,
                coal.Transform3s(np.eye(3),np.array([separation,0,0])),request,result)
            self.assertEqual(result.isCollision(),expected)
            if expected:
                self.assertAlmostEqual(result.getContact(0).penetration_depth,separation-.2,places=8)

    def test_panda_backend_agreement_and_no_mujoco_collision_call(self):
        scene=Path('collision_obstacle_test/scene.xml')
        lab=PandaLab(scene)
        lab.segment(HOME,.5,phase='settle')
        before=lab.data.qpos.copy()
        reference=PandaCollisionChecker(scene,before)
        checker=CoalCollisionChecker(scene,before)
        archive=np.load('collision_obstacle_test/detour.npz')
        start,goal=archive['start'],archive['goal']
        samples=[(1-u)*start+u*goal for u in np.linspace(0,1,101)]
        expected=[reference.valid(q) for q in samples]
        self.assertIn(False,expected)
        with patch('mujoco.mj_forward',side_effect=AssertionError('Coal must not use mj_forward')), \
             patch('mujoco.mj_collision',side_effect=AssertionError('Coal must not use mj_collision')):
            actual=[checker.valid(q) for q in samples]
        self.assertEqual(actual,expected)
        self.assertTrue(checker.valid(start))
        self.assertTrue(checker.valid(goal))
        self.assertFalse(checker.edge(start,goal)[0])
        self.assertGreater(checker.narrow_phase_calls,0)
        np.testing.assert_array_equal(lab.data.qpos,before)
        self.assertFalse(checker.valid(np.full(7,np.nan)))
        self.assertFalse(checker.valid(np.full(7,100.)))

    def test_positive_clearance_on_real_geometry(self):
        scene=Path('collision_obstacle_test/scene.xml')
        lab=PandaLab(scene)
        lab.segment(HOME,.5,phase='settle')
        q=json.loads(Path('collision_obstacle_test/report.json').read_text())['blocking_contact']['configuration']
        checker=CoalCollisionChecker(scene,lab.data.qpos)
        touching=CoalCollisionChecker(scene,lab.data.qpos,clearance=0)
        gid=checker.model.geom('test_obstacle').id
        center=checker.model.geom_pos[gid].copy()
        for shift in np.linspace(0,.2,401):
            checker.model.geom_pos[gid]=center+[shift,0,0]
            touching.model.geom_pos[gid]=center+[shift,0,0]
            issues=checker.explain(q)
            if issues and all(x['signed_distance_m']>0 for x in issues) and touching.valid(q):
                return
        self.fail('Did not find a positive-distance clearance rejection')


if __name__=='__main__':
    unittest.main()
