"""Prepare MuJoCo RGB-D input and visualize Contact-GraspNet output."""
import argparse
import os
from pathlib import Path
from xml.sax.saxutils import quoteattr

import numpy as np


def prepare(mesh_path, out):
    if os.name != "nt":
        os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco
    from PIL import Image

    vertices = np.array([list(map(float, line.split()[1:4]))
                         for line in mesh_path.read_text().splitlines()
                         if line.startswith("v ")])
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    if not 0.10 < max(hi - lo) < 0.35:
        raise ValueError(f"Unexpected mustard dimensions in metres: {hi-lo}")
    pos = np.array([-(lo[0]+hi[0])/2, -(lo[1]+hi[1])/2, -lo[2]])
    target = np.array([0., 0., (hi[2]-lo[2])/2])
    eye = target + [0.35, -0.45, 0.30]
    back = eye-target
    back /= np.linalg.norm(back)
    right = np.cross([0., 0., 1.], back)
    right /= np.linalg.norm(right)
    up = np.cross(back, right)
    fmt = lambda a: " ".join(map(str, a))
    xml = f'''<mujoco>
      <visual><global offwidth="640" offheight="480"/><map znear="0.01" zfar="10"/></visual>
      <asset><mesh name="bottle" file={quoteattr(str(mesh_path.resolve()))}/></asset>
      <worldbody>
        <light pos="0 -1 2" dir="0 0 -1"/>
        <geom name="table" type="plane" size="1 1 .02" rgba=".3 .35 .4 1"/>
        <geom name="object" type="mesh" mesh="bottle" pos="{fmt(pos)}" rgba=".95 .7 .05 1"/>
        <camera name="camera" pos="{fmt(eye)}" xyaxes="{fmt(np.r_[right,up])}" fovy="45"/>
      </worldbody></mujoco>'''
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        renderer.update_scene(data, camera="camera")
        rgb = renderer.render().copy()
        renderer.enable_depth_rendering()
        depth = renderer.render().copy()
        renderer.disable_depth_rendering()
        renderer.enable_segmentation_rendering()
        segmentation = renderer.render().copy()
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object")
    mask = (segmentation[..., 0] == gid) & (segmentation[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
    assert mask.sum() > 100, "Object not visible in camera"
    depth[(~np.isfinite(depth)) | (depth > 1.8)] = 0
    focal = 240 / np.tan(np.deg2rad(45)/2)
    K = np.array([[focal,0,319.5],[0,focal,239.5],[0,0,1]])
    T = np.eye(4)
    T[:3,:3] = np.column_stack([right, -up, -back])
    T[:3,3] = eye
    out.mkdir(parents=True, exist_ok=True)
    # The upstream loader expects BGR in the input file.
    np.savez(out / "mustard.npz", depth=depth, K=K, rgb=rgb[...,::-1], segmap=mask.astype(np.int32))
    np.savez(out / "camera_metadata.npz", T_world_camera=T, mesh_translation=pos)
    Image.fromarray(rgb).save(out / "mustard_rgb.png")
    print("Prepared:", out / "mustard.npz", "object pixels:", mask.sum())
    print("Synthetic single-view depth, metres. Yellow appearance; original mesh geometry.")


def visualize(out, predictions):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with np.load(out / "mustard.npz") as inp:
        rgb, K = inp["rgb"][...,::-1], inp["K"]
    with np.load(predictions, allow_pickle=True) as result:
        grasps = result["pred_grasps_cam"].item()
        scores = result["scores"].item()
        contacts = result["contact_pts"].item()
    fig, ax = plt.subplots(figsize=(10,7))
    ax.imshow(rgb)
    count = 0
    for key, poses in grasps.items():
        if float(key) <= 0:
            continue
        score = np.asarray(scores[key]).reshape(-1)
        count += len(poses)
        for i in np.argsort(score)[-10:]:
            T = poses[i]
            origin = T[:3,3]
            # Project pose axes, not a physical gripper/collision model.
            points = np.vstack([origin, origin + .035*T[:3,:3].T])
            if np.any(points[:,2] <= 0):
                continue
            uv = points @ K.T
            uv = uv[:,:2]/uv[:,2:3]
            for j, color in enumerate(["red", "lime", "cyan"], 1):
                ax.plot(uv[[0,j],0], uv[[0,j],1], color=color, linewidth=1.4)
        print(f"Object {key}: {len(poses)} grasps; best score: {score.max() if len(score) else 'none'}")
    ax.set(xlim=(0,640), ylim=(480,0), title=f"{count} predicted grasps; axes of up to 10 best per object\nRed=X, green=Y, cyan=Z; predictions, not validated lifts")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "mustard_grasps.png", dpi=160)
    plt.close(fig)
    print("Saved:", out / "mustard_grasps.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "visualize"])
    parser.add_argument("--mesh", type=Path, default=Path.home()/"robotics/datasets/ycb/006_mustard_bottle/google_16k/textured.obj")
    parser.add_argument("--out", type=Path, default=Path("ycb_demo"))
    parser.add_argument("--predictions", type=Path, default=Path("results/predictions_mustard.npz"))
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.mesh, args.out)
    else:
        visualize(args.out, args.predictions)
