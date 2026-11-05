"""A lifted TCP alone, loss of contact, or an unstable hold must never pass."""
from copy import deepcopy
from lift_metrics import assess_lift

before = dict(object_z_m=.48, object_in_tcp=[0,0,0])
final = dict(object_z_m=.62, clearance_m=.14, object_in_tcp=[0,0,0], table_contact=False)
hold = [dict(time=t, contact=[8,8], object_in_tcp=[0,0,0], clearance_m=.14, table_contact=False)
        for t in [0.,1.,2.,3.]]
assert assess_lift(before,hold,final)['lift_success']
fallen = dict(final,object_z_m=.48,clearance_m=0,table_contact=True)
assert not assess_lift(before,hold,fallen)['lift_success']
lost_contact = deepcopy(hold)
lost_contact[-1]['contact'] = [0,0]
assert not assess_lift(before,lost_contact,final)['lift_success']
drifting = deepcopy(hold)
drifting[-1]['object_in_tcp'] = [0,0,.02]
assert not assess_lift(before,drifting,final)['lift_success']
print('PASS: object height, table separation, contact loss and hold drift checks.')
