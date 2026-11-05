import unittest
import numpy as np
from pathlib import Path
import json
from collision_planning import rrt_connect, PandaCollisionChecker


def disk_edge(a,b):
    a,b=np.asarray(a),np.asarray(b)
    if np.any(a < -1) or np.any(a > 1) or np.any(b < -1) or np.any(b > 1):
        return False
    delta=b-a
    fraction=np.clip(-np.dot(a,delta)/max(np.dot(delta,delta),1e-20),0,1)
    return np.linalg.norm(a+fraction*delta)>.3


class PlannerTests(unittest.TestCase):
    def test_detour_around_obstacle(self):
        a,b=np.array([-.8,0]),np.array([.8,0])
        self.assertFalse(disk_edge(a,b))
        plan=rrt_connect(a,b,np.array([[-1,1],[-1,1]]),disk_edge,seed=5,timeout=5)
        self.assertEqual(plan.status,'rrt_connect')
        np.testing.assert_allclose(plan.path[0],a)
        np.testing.assert_allclose(plan.path[-1],b)
        for x,y in zip(plan.path[:-1],plan.path[1:]):
            self.assertTrue(disk_edge(x,y))

    def test_invalid_goal(self):
        plan=rrt_connect([-.8,0],[0,0],np.array([[-1,1],[-1,1]]),disk_edge)
        self.assertIsNone(plan.path)
        self.assertEqual(plan.status,'invalid_endpoint')

    def test_impenetrable_wall(self):
        def edge(a,b):
            return bool((a[0]<-.1 and b[0]<-.1) or (a[0]>.1 and b[0]>.1))
        plan=rrt_connect([-.8,0],[.8,0],np.array([[-1,1],[-1,1]]),edge,
                         iterations=80,timeout=5,seed=3)
        self.assertIsNone(plan.path)

    def test_direct(self):
        plan=rrt_connect([0,0],[1,1],np.array([[0,1],[0,1]]),lambda a,b:True)
        self.assertEqual(plan.status,'direct')


@unittest.skipUnless(Path('collision_obstacle_test/report.json').exists(), 'Run Panda obstacle benchmark first')
class PandaGeometryTests(unittest.TestCase):
    def test_real_robot_detour_and_state_isolation(self):
        from panda_lab import PandaLab,HOME
        lab=PandaLab('collision_obstacle_test/scene.xml')
        lab.segment(HOME,.5,phase='settle')
        original=lab.data.qpos.copy()
        checker=PandaCollisionChecker('collision_obstacle_test/scene.xml',original,resolution=.005)
        with np.load('collision_obstacle_test/detour.npz') as archive:
            path,start,goal=archive['path'],archive['start'],archive['goal']
        self.assertTrue(checker.valid(start))
        self.assertTrue(checker.valid(goal))
        self.assertFalse(checker.edge(start,goal)[0])
        self.assertTrue(all(checker.edge(a,b)[0] for a,b in zip(path[:-1],path[1:])))
        np.testing.assert_array_equal(lab.data.qpos,original)

    def test_positive_clearance_rejection(self):
        from panda_lab import PandaLab,HOME
        lab=PandaLab('collision_obstacle_test/scene.xml')
        lab.segment(HOME,.5,phase='settle')
        report=json.loads(Path('collision_obstacle_test/report.json').read_text())
        q=np.array(report['blocking_contact']['configuration'])
        checker=PandaCollisionChecker('collision_obstacle_test/scene.xml',lab.data.qpos,clearance=.002)
        touching=PandaCollisionChecker('collision_obstacle_test/scene.xml',lab.data.qpos,clearance=0)
        gid=checker.model.geom('test_obstacle').id
        center=checker.model.geom_pos[gid].copy()
        found=False
        for shift in np.linspace(0,.20,401):
            center_shifted=center+[shift,0,0]
            checker.model.geom_pos[gid]=center_shifted
            touching.model.geom_pos[gid]=center_shifted
            issues=checker.explain(q)
            if issues and all(x.get('signed_distance_m',-1)>0 for x in issues) and touching.valid(q):
                found=True
                break
        self.assertTrue(found,'A positive separation below 2mm should fail the clearance check')


if __name__=='__main__':
    unittest.main()
