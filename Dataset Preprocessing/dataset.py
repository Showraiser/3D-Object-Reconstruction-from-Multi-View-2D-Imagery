import os, glob, numpy as np, torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class ImageVoxelDataset(Dataset):
    def __init__(self, renders_root, voxels_root, split_file, voxel_res=64, img_size=224, view_index=0):
        self.renders_root = renders_root
        self.voxels_root = voxels_root
        self.items = [l.strip() for l in open(split_file) if l.strip()]
        self.voxel_res = voxel_res
        self.img_size = img_size
        self.view_index = view_index
        self.transform = T.Compose([
            T.Resize((img_size,img_size)), T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        rel = self.items[idx]
        cat, model_id = rel.split("/",1)
        img_dir = os.path.join(self.renders_root,cat,model_id,"images")
        imgs = sorted(glob.glob(os.path.join(img_dir,"*.png")))
        img = Image.open(imgs[self.view_index]).convert("RGB")
        img_t = self.transform(img)
        vox_path = os.path.join(self.voxels_root,cat,model_id,f"vox_{self.voxel_res}.npy")
        vox = np.load(vox_path)
        vox_t = torch.from_numpy(vox).float().unsqueeze(0)
        return img_t, vox_t, {"rel":rel}
#this is the pytorch datasloader
import os, torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class ImageVoxelDataset(Dataset):
    def __init__(self, renders_root, voxels_root, split_file,
                 voxel_res=64, img_size=224, transform=None):
        self.renders_root = renders_root
        self.voxels_root = voxels_root
        self.voxel_res = voxel_res
        self.img_size = img_size
        self.transform = transform

        with open(split_file, "r") as f:
            self.items = [line.strip() for line in f]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        rel = self.items[idx]
        # Image path (pick first view)
        img_dir = os.path.join(self.renders_root, rel, "images")
        img_path = os.path.join(img_dir, "000.png")
        img = Image.open(img_path).convert("RGB").resize((self.img_size, self.img_size))
        img = np.array(img).transpose(2,0,1) / 255.0  # [C,H,W]
        img = torch.tensor(img, dtype=torch.float32)

        # Voxel path
        vox_path = os.path.join(self.voxels_root, rel, f"vox_{self.voxel_res}.npy")
        vox = np.load(vox_path)
        vox = np.expand_dims(vox, 0)  # [1,D,H,W]
        vox = torch.tensor(vox, dtype=torch.float32)

        meta = {"id": rel}

        return img, vox, meta

# Example usage:
# from dataset import ImageVoxelDataset

# train_ds = ImageVoxelDataset(
#     renders_root="./datasets/renders/modelnet10",
#     voxels_root="./datasets/voxels/modelnet10",
#     split_file="./splits/modelnet10/train.txt",
#     voxel_res=64, img_size=224
# )

# img, vox, meta = train_ds[0]
# print(img.shape)   # torch.Size([3, 224, 224])
# print(vox.shape)   # torch.Size([1, 64, 64, 64])
# print(meta)     # {'id': 'airplane/airplane_0001'}
