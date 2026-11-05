import json
from copy import deepcopy
from pathlib import Path
from ycb_inertia import box_baseline,parse_answer,validate_estimate
case=json.loads(Path('ycb_inertia_inputs/case.json').read_text())
estimate=box_baseline(case['mass_kg'],case['bounds_m'])
estimate.update(object_id=case['object_id'],frame='mesh_local_O',
    inertia_reference='center_of_mass_axes_parallel_to_O',assumptions=['solid uniform box'],
    limitations=['not the real geometry or density'])
assert validate_estimate(estimate,case)['valid']
assert parse_answer('<answer>'+json.dumps(estimate)+'</answer>')==estimate
for field,value in [('mass_kg',1.),('com_m',[2,0,0]),('inertia_com_kg_m2',[[1,0,0],[0,1,0],[0,0,3]]),
                    ('inertia_com_kg_m2',[[1,0,0],[0,1,0],[0,0,1]]),('frame','pixels')]:
    bad=deepcopy(estimate);bad[field]=value
    assert not validate_estimate(bad,case)['valid'],field
bad=deepcopy(estimate);bad['inertia_com_kg_m2'][0][1]=.01
assert not validate_estimate(bad,case)['valid']
assert not validate_estimate(dict(estimate,com_m=None),case)['valid']
try: parse_answer('<answer>{"mass_kg": NaN}</answer>')
except ValueError: pass
else: raise AssertionError('NaN accepted')
print('PASS: known mass, frame, COM bounds, tensor symmetry, triangle inequality, size bounds, abstention and parser.')
