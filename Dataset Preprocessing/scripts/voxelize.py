import os, argparse
import numpy as np
import trimesh
from tqdm import tqdm

def normalize_mesh(mesh: trimesh.Trimesh):
    mesh = mesh.copy()
    mesh.vertices -= mesh.centroid
    bounds = mesh.bounds
    size = (bounds[1] - bounds[0]).max()
    if size > 0:
        mesh.vertices /= size
    return mesh

def voxelize_mesh(mesh_path, out_file, resolution=32):
    mesh = trimesh.load(mesh_path, force='mesh')
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump().geometry.values())
    mesh = normalize_mesh(mesh)

    voxel = mesh.voxelized(pitch=2.0/resolution)
    mat = voxel.matrix.astype(np.uint8)

    D, H, W = mat.shape
    padded = np.zeros((resolution, resolution, resolution), dtype=np.uint8)
    d0 = (resolution - D) // 2
    h0 = (resolution - H) // 2
    w0 = (resolution - W) // 2
    padded[d0:d0+D, h0:h0+H, w0:w0+W] = mat
    mat = padded

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    np.save(out_file, mat)


def collect_meshes(mesh_root):
    exts = (".off", ".obj", ".ply", ".stl")
    meshes = []
    for root, _, files in os.walk(mesh_root):
        for fn in files:
            if fn.lower().endswith(exts):
                path = os.path.join(root, fn)
                parts = os.path.normpath(path).split(os.sep)
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
    ap.add_argument("--resolutions", nargs="+", type=int, default=[32])
    args = ap.parse_args()

    meshes = collect_meshes(args.mesh_root)
    for rel, mpath in tqdm(meshes):
        for res in args.resolutions:
            out_file = os.path.join(args.out_root, rel, f"vox_{res}.npy")
            if os.path.exists(out_file):
                continue
            try:
                voxelize_mesh(mpath, out_file, resolution=res)
            except Exception as e:
                print("FAILED:", mpath, e)

if __name__ == "__main__":
    main()
