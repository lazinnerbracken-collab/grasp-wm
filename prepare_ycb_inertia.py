"""Generate calibrated synthetic views and a known-mass case for the VLM."""
import copy
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from PIL import Image
from ycb_inertia import make_prompt, box_baseline

root=Path('assets/mustard').resolve()
out=Path('ycb_inertia_inputs'); out.mkdir(exist_ok=True)
vertices=np.array([list(map(float,line.split()[1:4])) for line in
                  (root/'mustard_bottle/textured.obj').read_text().splitlines() if line.startswith('v ')])
bounds=np.array([vertices.min(0),vertices.max(0)])
case=dict(object_id='006_mustard_bottle',mass_kg=.603,
    mass_provenance='Value in the supplied YCB-derived XML; replace with your measured mass if different.',
    frame='mesh_local_O',origin='Unmodified local origin of this processed YCB mesh; axes shown in the views.',
    bounds_m=bounds.tolist(),dimensions_m=np.ptp(vertices,axis=0).tolist(),
    fill_state='unknown; do not assume the label establishes actual contents or fill level',
    images_are='synthetic renders of the textured YCB mesh, not scans of its internal contents')
(out/'case.json').write_text(json.dumps(case,indent=2),encoding='utf-8')
(out/'prompt.txt').write_text(make_prompt(case),encoding='utf-8')
(out/'baseline.json').write_text(json.dumps(box_baseline(.603,bounds),indent=2),encoding='utf-8')
xml=ET.Element('mujoco',model='YCB inertia views')
ET.SubElement(xml,'compiler',angle='radian')
visual=ET.SubElement(xml,'visual'); ET.SubElement(visual,'global',offwidth='640',offheight='640')
asset=ET.SubElement(xml,'asset')
source=ET.parse(root/'source.xml').getroot()
for node in source.find('asset'):
    node=copy.deepcopy(node)
    if node.get('file'): node.set('file',(root/node.get('file')).as_posix())
    asset.append(node)
world=ET.SubElement(xml,'worldbody')
for pos in ['.3 -.4 .5','-.4 .2 .4']:
    ET.SubElement(world,'light',pos=pos,dir='0 0 -1',diffuse='.8 .8 .8',ambient='.3 .3 .3')
obj=ET.SubElement(world,'body',name='object')
ET.SubElement(obj,'geom',type='mesh',mesh='mustard_bottle_visual_mesh',material='mustard_bottle_material',contype='0',conaffinity='0')
for endpoint,color in [('.13 0 0','1 0 0 1'),('0 .13 0','0 1 0 1'),('0 0 .17','0 .3 1 1')]:
    ET.SubElement(world,'site',type='cylinder',fromto='0 0 0 '+endpoint,size='.0012',rgba=color)
model=mujoco.MjModel.from_xml_string(ET.tostring(xml,encoding='unicode'))
data=mujoco.MjData(model); mujoco.mj_forward(model,data)
renderer=mujoco.Renderer(model,height=640,width=640)
camera=mujoco.MjvCamera(); camera.lookat[:]=bounds.mean(0); camera.distance=.5
views=[]
try:
    for i,(az,el) in enumerate([(90,-12),(0,-12),(45,-45)],1):
        camera.azimuth,camera.elevation=az,el
        renderer.update_scene(data,camera=camera)
        filename=f'view_{i}.png'; Image.fromarray(renderer.render()).save(out/filename)
        views.append(dict(file=filename,azimuth_deg=az,elevation_deg=el,
                          axis_colors={'x':'red','y':'green','z':'blue'}))
finally: renderer.close()
(out/'views.json').write_text(json.dumps(views,indent=2))
print('Prepared',out, 'No VLM inference has been run.')
