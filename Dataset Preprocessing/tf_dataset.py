#this is the Tensorflow dataloader
import os, tensorflow as tf, numpy as np
from PIL import Image

def _load_item(rel, renders_root, voxels_root, voxel_res=64, img_size=224):
    img_path = os.path.join(renders_root, rel, "images", "000.png")
    vox_path = os.path.join(voxels_root, rel, f"vox_{voxel_res}.npy")

    img = Image.open(img_path).convert("RGB").resize((img_size, img_size))
    img = np.array(img) / 255.0

    vox = np.load(vox_path).astype(np.float32)
    vox = np.expand_dims(vox, -1)  # [D,H,W,1]

    return img, vox

def make_tf_dataset(renders_root, voxels_root, split_file,
                    voxel_res=64, img_size=224, batch_size=8, shuffle=True):
    with open(split_file, "r") as f:
        items = [line.strip() for line in f]

    def gen():
        for rel in items:
            yield _load_item(rel, renders_root, voxels_root, voxel_res, img_size)

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(img_size, img_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(voxel_res, voxel_res, voxel_res, 1), dtype=tf.float32),
        )
    )

    if shuffle:
        ds = ds.shuffle(len(items))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# Example usage:
# from tf_dataset import make_tf_dataset

# train_ds = make_tf_dataset(
#     renders_root="./datasets/renders/modelnet10",
#     voxels_root="./datasets/voxels/modelnet10",
#     split_file="./splits/modelnet10/train.txt",
#     voxel_res=64, img_size=224
# )

# for img, vox in train_ds.take(1):
#     print(img.shape)   # (224, 224, 3)
#     print(vox.shape)   # (64, 64, 64, 1)
