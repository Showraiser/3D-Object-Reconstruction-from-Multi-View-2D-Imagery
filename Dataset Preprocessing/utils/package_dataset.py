import os
import shutil
import zipfile

# Settings
output_base = "PersonA_outputs_minimal"
max_zip_size = 5 * 1024 * 1024 * 1024   # 5 GB
voxel_res = "64"  # only keep 64³ voxels
img_size = "224"  # only keep 224px renders

folders_to_include = [
    ("datasets/renders/modelnet10", f"{img_size}"),  # pick 224px renders
    ("datasets/voxels/modelnet10", f"{voxel_res}"),  # pick 64³ voxels
    ("splits/modelnet10", None),                     # all split files
]

def add_folder_to_zip(ziph, folder, filter_substr=None):
    for root, dirs, files in os.walk(folder):
        for file in files:
            fpath = os.path.join(root, file)
            relpath = os.path.relpath(fpath, ".")
            if filter_substr and filter_substr not in fpath:
                continue
            ziph.write(fpath, relpath)

print("Packaging minimal dataset (64³ voxels + 224px renders)...")

part = 1
current_size = 0
ziph = zipfile.ZipFile(f"{output_base}_part{part}.zip", "w", zipfile.ZIP_DEFLATED)

for folder, filter_substr in folders_to_include:
    for root, dirs, files in os.walk(folder):
        for file in files:
            fpath = os.path.join(root, file)
            if filter_substr and filter_substr not in fpath:
                continue
            relpath = os.path.relpath(fpath, ".")
            ziph.write(fpath, relpath)
            current_size += os.path.getsize(fpath)

            # If current zip exceeds max size → start a new part
            if current_size > max_zip_size:
                ziph.close()
                print(f"Created {output_base}_part{part}.zip")
                part += 1
                current_size = 0
                ziph = zipfile.ZipFile(f"{output_base}_part{part}.zip", "w", zipfile.ZIP_DEFLATED)

ziph.close()
print(f"Packaging complete! Check {output_base}_part*.zip")

