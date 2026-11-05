# Panda grasping results

Experiment: YCB French’s mustard bottle in MuJoCo.

- 100 Contact-GraspNet candidates evaluated.
- 12 accepted after IK, collision and trajectory checks.
- 10 rejected for collision or insufficient clearance.
- 78 failed an IK attempt.
- Selected grasp lifted the bottle approximately 14.5 cm.

Coal checks collisions during planning; RRT-Connect finds
a detour when the direct route to the pregrasp is blocked.
These results concern one object and one synthetic view.
