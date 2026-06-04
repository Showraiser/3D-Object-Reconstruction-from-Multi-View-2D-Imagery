import os, argparse, json, math, glob
import numpy as np
import trimesh
import pyrender
from PIL import Image
from tqdm import tqdm

def normalize_mesh(mesh: trimesh.Trimesh):
    mesh = mesh.copy()
    # Center at origin
    mesh.vertices -= mesh.centroid
    # Scale to fit unit cube
    bounds = mesh.bounds
    size = (bounds[1] - bounds[0]).max()
    if size > 0:
        mesh.vertices /= size
    return mesh

def render_model(mesh_path, out_dir, views=24, img_size=256, distance=2.0, elevation=20.0):
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)

    mesh = trimesh.load(mesh_path, force='mesh')
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump().geometry.values())
    mesh = normalize_mesh(mesh)

    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 0.0])
    mesh_pyr = pyrender.Mesh.from_trimesh(mesh, smooth=True)
    scene.add(mesh_pyr)

    light = pyrender.DirectionalLight(intensity=3.0)
    scene.add(light)

    r = pyrender.OffscreenRenderer(viewport_width=img_size, viewport_height=img_size)

    cameras_meta = []
    for i in range(views):
        theta = 2 * math.pi * (i / views)
        cam_x = distance * math.cos(theta)
        cam_z = distance * math.sin(theta)
        cam_y = distance * math.sin(math.radians(elevation))

        camera_pose = np.eye(4)
        target = np.array([0.0, 0.0, 0.0])
        eye = np.array([cam_x, cam_y, cam_z])
        up = np.array([0.0, 1.0, 0.0])
        f = (target - eye); f = f / np.linalg.norm(f)
        s = np.cross(f, up); s = s / np.linalg.norm(s)
        u = np.cross(s, f)
        camera_pose[:3, :3] = np.stack([s, u, -f], axis=1)
        camera_pose[:3, 3] = eye

        cam = pyrender.PerspectiveCamera(yfov=np.deg2rad(45.0))
        node_cam = scene.add(cam, pose=camera_pose)

        color, _ = r.render(scene)
        Image.fromarray(color).save(os.path.join(out_dir, "images", f"{i:03d}.png"))

        scene.remove_node(node_cam)
        cameras_meta.append({"index": i, "pose": camera_pose.tolist()})

    with open(os.path.join(out_dir, "cameras.json"), "w") as f:
        json.dump({"views": cameras_meta}, f, indent=2)

def collect_meshes(mesh_root):
    exts = (".off", ".obj", ".ply", ".stl", ".glb", ".gltf")
    meshes = []
    for root, _, files in os.walk(mesh_root):
        for fn in files:
            if fn.lower().endswith(exts):
                path = os.path.join(root, fn)
                parts = os.path.normpath(path).split(os.sep)
                # category/train/model.off or category/test/model.off
                if len(parts) >= 3:
                    cat = parts[-3]
                    split = parts[-2]
                    model_id = os.path.splitext(fn)[0]
                    rel = f"{cat}/{split}/{model_id}"
                else:
                    rel = os.path.splitext(fn)[0]
                meshes.append((rel, path))
    return meshes

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--views", type=int, default=24)
    ap.add_argument("--img-size", type=int, default=256)
    args = ap.parse_args()

    meshes = collect_meshes(args.mesh_root)
    for rel, mpath in tqdm(meshes):
        out_dir = os.path.join(args.out_root, rel)
        if os.path.exists(os.path.join(out_dir, "images", "000.png")):
            continue
        os.makedirs(out_dir, exist_ok=True)
        try:
            render_model(mpath, out_dir, views=args.views, img_size=args.img_size)
        except Exception as e:
            print("FAILED:", mpath, e)

if __name__ == "__main__":
    main()
