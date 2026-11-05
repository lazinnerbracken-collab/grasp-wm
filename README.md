# grasp-wm

Robotic grasping experiments using Contact-GraspNet, Franka Panda simulation, Coal collision checking and RRT-Connect. Additional notebooks explore NERO and visual estimates of YCB object inertial properties.

## Start here

The easiest standalone demonstration is **Panda_MuJoCo_Colab.ipynb**: upload it to Google Colab and run its cells. This basic demonstration uses a scripted target, not Contact-GraspNet predictions. See [README_PANDA_BASIC.md](README_PANDA_BASIC.md).

For the current grasp-filtering pipeline, install [Pixi](https://pixi.sh), clone this repository, and run commands from its root:

```bash
pixi install --manifest-path coal_env/pixi.toml
pixi run --manifest-path coal_env/pixi.toml python fetch_assets.py
```

The existing Contact-GraspNet experiment additionally requires your saved input bundle. Restore `panda_grasp_inputs/` from your original project or extract `panda_grasp_inputs.tar.gz` into the repository root. Check these paths before running:

```text
panda_grasp_inputs/
  datasets/ycb/006_mustard_bottle/google_16k/textured.obj
  datasets/ycb/006_mustard_bottle/google_16k/texture_map.png
  contact_graspnet/ycb_demo/camera_metadata.npz
  grasp_project/results/contact_graspnet_scene/006_mustard_bottle/synthetic_0/grasps.npz
  grasp_project/results/contact_graspnet_scene/006_mustard_bottle/synthetic_0/metadata.json
```

Then run:

```bash
pixi run --manifest-path coal_env/pixi.toml python filter_panda_grasps.py --checker coal --limit 100 --output coal_results_all --execute
```

Outputs include `coal_results_all/planned_lift.mp4`, `filter_report.json`, the filtered grasps, and checked joint paths. Use `--checker mujoco` to select the original collision backend. See [README_COAL.md](README_COAL.md) and [COLLISION_PLANNING.md](COLLISION_PLANNING.md).

## Code map

| Files | Purpose |
| --- | --- |
| `panda_lab.py`, `lift_metrics.py` | Panda model, IK, torque/impedance control and lift measurements |
| `ycb_grasp_demo.py` | Synthetic RGB-D preparation and prediction visualization |
| `contact_adapter/` | Contact-GraspNet adapter to copy into the existing `grasp_project` checkout |
| `panda_predicted_grasp.py` | Execute downloaded Contact-GraspNet predictions |
| `filter_panda_grasps.py` | Filter candidates and execute the selected grasp |
| `collision_planning.py`, `coal_collision.py` | RRT-Connect and alternative collision checkers |
| `benchmark_panda_obstacle.py` | Deliberately blocked-path planning test |
| `compare_collision_backends.py` | Compare stored MuJoCo/Coal results and recheck the Coal detour |
| `visualize_collision_results.py` | Plots from the MuJoCo baseline results; requires Matplotlib |
| `test_*.py` | Unit and simulation integration checks; some require generated scenes/data |
| `NERO_Bimanuel_Colab.ipynb`, `README_NERO.md` | Earlier NERO experiment; requires the separately retained NERO asset archive |
| `YCB_*Colab.ipynb`, `ycb_inertia.py` | VLM/inertia experiments with their own notebook dependencies |

The adapter needs `grasp_project`'s shared `common.py` and the separately installed original Contact-GraspNet toolbox, weights and TensorFlow/CUDA operators. This repository does not contain those upstream projects or weights. See `contact_adapter/README.md`.

## Reproduce the planning checks

After restoring the input bundle and fetching assets:

```bash
pixi run --manifest-path coal_env/pixi.toml python filter_panda_grasps.py --checker mujoco --limit 10 --output collision_results
pixi run --manifest-path coal_env/pixi.toml python benchmark_panda_obstacle.py
pixi run --manifest-path coal_env/pixi.toml python -m unittest test_collision_planning.py test_coal_collision.py
pixi run --manifest-path coal_env/pixi.toml python benchmark_panda_obstacle.py --checker coal --source coal_results_all --output coal_obstacle_test
```

For `compare_collision_backends.py`, first generate the full MuJoCo baseline with `--checker mujoco --limit 100 --output collision_results_all` as well as the Coal results above. The comparison script expects these exact output directories.

## Observed results and limits

The previous local Windows experiment used one synthetic view of YCB `006_mustard_bottle`, not the complete YCB dataset. Out of 100 candidates, 12 passed filtering, 10 failed collision/clearance checks and 78 failed one IK attempt. Coal and MuJoCo made the same decisions. Candidate index 5 lifted the object about 14.5 cm. Nine planning/backend tests passed in the original workspace.

Coal checks collision geometry using MuJoCo link transforms; RRT-Connect searches transit paths. MuJoCo still simulates execution. The final approach is straight; blocked approaches are rejected. Collision checks are discrete, with a 2 mm margin. The lift with a grasped object is monitored but not preplanned. These are simulation results, not real-robot safety validation.

## External data and environments

Installed environments, downloaded meshes/datasets, weights, videos, logs, and generated scenes/results are intentionally excluded. Keep your existing input bundle and NERO assets separately. This source snapshot does not download or regenerate the server's learned predictions automatically.

Upstream models and libraries retain their respective licenses. `fetch_assets.py` preserves model license files. No blanket license for third-party assets is granted by this repository.

- [Contact-GraspNet](https://github.com/NVlabs/contact_graspnet)
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
- [YCB](https://www.ycbbenchmarks.com/)
- [Coal](https://github.com/coal-library/coal)
