"""Physical integration checks; run after fetch_assets.py."""
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from panda_lab import build_scene, PandaLab, DOWN

build_scene()
lab = PandaLab()
saved = lab.data.qpos.copy()
target = np.array([.48, 0., .76])
solution = lab.ik(target, DOWN)
assert np.array_equal(saved, lab.data.qpos), "IK must not teleport the simulated robot"
work = mujoco.MjData(lab.model)
work.qpos[:] = saved
work.qpos[lab.aq] = solution
mujoco.mj_forward(lab.model, work)
p, r = lab.pose(work)
assert np.linalg.norm(p - target) < 2e-4
assert Rotation.from_matrix(DOWN @ r.T).magnitude() < .002
try:
    lab.ik([4., 0., 4.], DOWN, max_iterations=60)
except RuntimeError:
    pass
else:
    raise AssertionError("Unreachable pose must report failure")
# Verify the chosen site is exactly the Hydrax gripper offset in hand frame.
hand = lab.model.body("hand").id
local = lab.data.xmat[hand].reshape(3, 3).T @ (lab.pose()[0] - lab.data.xpos[hand])
assert np.allclose(local, [0., 0., .1])
result = lab.demo()
assert result["bilateral_contact"], result
assert result['lift_success'], result
assert result['object_lift_m'] >= .10 and result['table_separated'], result
assert result['final_tcp_error_m'] < .02, result
assert .005 < result["opening_m"] < .07, result
assert result["pose_error"]["position_m"] < .005, result
assert not any(result["warnings"]), result
assert all(max(abs(x) for x in row["force"]) <= lab.force_limit + 1e-8 for row in lab.log)
print("PASS: pose IK, non-mutating solver, unreachable target, TCP offset, dynamic tracking, bilateral contact, force limit, numerical stability.")
print(result)
