# Contact-GraspNet adapter for grasp_project

Copy the two files under grasp/adapters into the server's grasp_project checkout.
Existing files and the shared common.py do not need changes. Invoke this adapter
as a Python module; it is NOT registered in the existing `grasp generate` CLI,
whose source was not available for inspection.

From the grasp_project directory, using the existing Contact-GraspNet Python:

```bash
~/robotics/contact_graspnet/.pixi/envs/default/bin/python -m grasp.adapters.contact_graspnet \
  --from-demo ~/robotics/contact_graspnet/ycb_demo/mustard.npz
```

For subsequent runs, use the converted observation instead:

```bash
~/robotics/contact_graspnet/.pixi/envs/default/bin/python -m grasp.adapters.contact_graspnet \
  --input observations/mustard_contact/observation.npz \
  --output results/contact_graspnet_second_run
```

Use --help for explicit toolbox, Python, checkpoint, output and candidate-count options.
Default paths follow the server setup in this conversation. No weights or dependencies
are installed by this adapter. Existing TensorFlow 2.12, custom compiled operators,
safe YAML loader and lazy renderer-import patches are prerequisites.

The worker preserves the observed scene and object mask. It calls the original
predict_scene_grasps with local_regions=True and filter_grasps=True, matching the
earlier successful depth-input demo's scene policy. It calls the estimator directly,
retains native confidence thresholds/sampling, and ranks the returned candidates.
It retains predicted openings, including configured extra opening and clipping.

Output: results/contact_graspnet/006_mustard_bottle/synthetic_0/grasps.npz,
metadata.json, worker.log; aggregate manifest.json in results/contact_graspnet.
Archives contain ordinary numeric/string arrays readable with allow_pickle=False.

The frame conversion follows the supplied project's standard Panda convention:
T_camera_tcp = T_camera_native @ T_native_tcp, where T_native_tcp is a +90 degree
Z rotation and +0.1034m Z translation. Upstream build_6d_grasp subtracts this
0.1034m depth from its predicted contact baseline. Custom grippers are unsupported.
Camera coordinates remain OpenCV-style. There is no camera-to-world conversion,
IK, collision check or lift validation. Check the actual simulator TCP before control.

Validation: three local contract tests pass against the user's supplied common.py,
with stubs for unavailable project utilities: rank/width alignment and rotated TCP
translation; empty/invalid outputs; depth conversion/masking and overwrite guard.
No remote GPU run or complete private-project integration has been performed.

References:
- https://github.com/NVlabs/contact_graspnet/blob/main/contact_graspnet/contact_grasp_estimator.py
- https://github.com/NVlabs/contact_graspnet/blob/main/contact_graspnet/contact_graspnet.py
