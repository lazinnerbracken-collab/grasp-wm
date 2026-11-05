"""Integration tests for both NERO arms and physical finger contact."""
import json
from pathlib import Path
import numpy as np
import mujoco
from nero_lab import NeroLab, build_scene, SIDES, DOWN, TCP_POS

results = {}
for side in SIDES:
    build_scene(side=side)
    lab = NeroLab()
    assert lab.model.nu == 18
    assert lab.data.ncon == 0, 'Initial robot geometry interpenetrates'
    saved = lab.data.qpos.copy()
    target = np.array([-.25 if side == 'right' else .25, -.45, .76])
    q = lab.ik(side, target, DOWN)
    assert np.array_equal(saved, lab.data.qpos), 'IK must not teleport the simulation'
    a = lab.arms[side]
    assert np.all(q > a['limits'][:, 0]) and np.all(q < a['limits'][:, 1])
    body = lab.model.body(f'{side}_link7').id
    local = lab.data.xmat[body].reshape(3,3).T @ (lab.pose(side)[0] - lab.data.xpos[body])
    assert np.allclose(local, TCP_POS)
    result = lab.demo(side)
    assert result['bilateral_contact'], result
    assert result['lift_success'], result
    assert result['object_lift_m'] >= .10 and result['table_separated'], result
    assert result['final_tcp_error_m'] < .02, result
    assert not any(result['warnings']), result
    assert not result['unexpected_contacts_over_1mm'], result
    assert result['pose_error'][side]['position_m'] < .001, result
    assert .02 < result['opening_m'] < .09, result
    for row in lab.log:
        for s in SIDES:
            assert np.max(np.abs(row[s]['force'])) <= lab.force_limit + 1e-8
    # The signed coordinates must stay opposed at the final grasp.
    f = lab.data.qpos[a['fq']]
    assert f[0] > 0 and f[1] < 0
    results[side] = result
    print(side, result)
try:
    lab.ik('right', [4, 0, 4], DOWN)
except RuntimeError:
    pass
else:
    raise AssertionError('Unreachable pose not rejected')
Path('nero_validation_both.json').write_text(json.dumps(results, indent=2))
print('PASS: both arms, non-mutating IK, TCP offsets, signed fingers, torque/force control, physical contacts, unreachable target.')
