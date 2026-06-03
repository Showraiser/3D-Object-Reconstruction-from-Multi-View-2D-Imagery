import os
import torch
import torch.nn as nn
import torch.nn.functional as Func
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms as T
from PIL import Image
import numpy as np
from glob import glob
from tqdm import tqdm
import subprocess
import gc

# ============================================================
# GPU / TORCH Logging & Globals
# ============================================================
def log_gpu_usage():
    try:
        output_lines = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total", "--format=csv,nounits,noheader"],
            encoding="utf-8"
        ).strip().splitlines()
        for line in output_lines:
            idx, util, mem_used, mem_total = map(int, line.split(", "))
            print(f"[GPU {idx}] Utilization: {util}% | Memory: {mem_used} MiB / {mem_total} MiB")
    except Exception as e:
        print(f"[GPU] Could not log GPU usage: {e}")

def log_torch_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"[PyTorch] Allocated: {allocated:.1f} MiB | Reserved: {reserved:.1f} MiB")

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ============================================================
# DATASET LOADING
# ============================================================
class ShapeNetDataset(Dataset):
    def __init__(self, root, num_views=24, voxel_size=64, img_size=256, train=True):
        self.root = root
        self.num_views = num_views
        self.voxel_size = voxel_size
        self.img_size = img_size
        self.samples = []

        if train:
            self.transform = T.Compose([
                T.RandomResizedCrop(self.img_size, scale=(0.9, 1.0)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                T.ToTensor()
            ])
        else:
            self.transform = T.Compose([
                T.Resize((self.img_size, self.img_size)),
                T.ToTensor()
            ])

        for class_dir in os.listdir(root):
            class_path = os.path.join(root, class_dir)
            if not os.path.isdir(class_path):
                continue
            for obj_dir in os.listdir(class_path):
                obj_path = os.path.join(class_path, obj_dir)
                view_dir = os.path.join(obj_path, "images")
                voxel_path = os.path.join(obj_path, f"vox_{self.voxel_size}.npy")
                if os.path.exists(view_dir) and os.path.exists(voxel_path):
                    self.samples.append((view_dir, voxel_path, obj_dir))

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found in {root} for voxel_size={self.voxel_size}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        view_dir, voxel_path, obj_id = self.samples[idx]
        image_paths = sorted(glob(os.path.join(view_dir, "*.png")))[:self.num_views]

        views = []
        for p in image_paths:
            img = Image.open(p).convert("RGB")
            img_t = self.transform(img)
            views.append(img_t)

        while len(views) < self.num_views:
            views.append(views[-1].clone())

        views = torch.stack(views)
        voxel = np.load(voxel_path).astype(np.float32)
        if voxel.ndim == 4 and voxel.shape[0] == 1:
            voxel = voxel[0]
        if voxel.shape != (self.voxel_size,)*3:
            raise RuntimeError(f"Voxel at {voxel_path} has shape {voxel.shape}, expected {(self.voxel_size,)*3}")
        voxel = torch.from_numpy(voxel).unsqueeze(0)

        return {"views": views, "voxel": voxel, "id": obj_id}

# ============================================================
# MODEL COMPONENTS
# ============================================================
class ResBlock3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.InstanceNorm3d(channels),
            nn.ReLU(True),
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.InstanceNorm3d(channels)
        )
    def forward(self, x):
        return Func.relu(x + self.net(x))

class ViewFusion(nn.Module):
    def __init__(self, n_views, feat_dim):
        super().__init__()
        self.attn = nn.Linear(feat_dim, 1)
    def forward(self, feats):
        weights = self.attn(feats)
        weights = torch.softmax(weights.squeeze(-1), dim=1).unsqueeze(-1)
        return (feats * weights).sum(dim=1)

class Encoder(nn.Module):
    def __init__(self, feature_dim=512, img_size=256, n_views=24):
        super().__init__()
        effnet = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        self.backbone = effnet.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self._inferred_feat_channels = self._infer_backbone_channels(img_size)
        self.fc = nn.Linear(self._inferred_feat_channels, feature_dim)

    def _infer_backbone_channels(self, img_size):
        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size, img_size)
            c = self.backbone(dummy).shape[1]
        return c

    def forward_single(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        return self.fc(torch.flatten(x, 1))

    def forward(self, views):
        return self.forward_single(views)

class Decoder(nn.Module):
    def __init__(self, feature_dim=512, voxel_size=64, start_channels=256):
        super().__init__()
        self.voxel_size = voxel_size
        self.start_size = 4
        self.start_ch = start_channels
        self.fc = nn.Linear(feature_dim, self.start_ch * (self.start_size ** 3))
        self.up_blocks = nn.ModuleList([
            nn.Sequential(nn.ConvTranspose3d(self.start_ch, self.start_ch//2, 4, 2, 1), nn.InstanceNorm3d(self.start_ch//2), nn.ReLU(True)),
            nn.Sequential(nn.ConvTranspose3d(self.start_ch//2, self.start_ch//4, 4, 2, 1), nn.InstanceNorm3d(self.start_ch//4), nn.ReLU(True)),
            nn.Sequential(nn.ConvTranspose3d(self.start_ch//4, self.start_ch//8, 4, 2, 1), nn.InstanceNorm3d(self.start_ch//8), nn.ReLU(True)),
            nn.Sequential(nn.ConvTranspose3d(self.start_ch//8, self.start_ch//16, 4, 2, 1), nn.InstanceNorm3d(self.start_ch//16), nn.ReLU(True)),
        ])
        self.res_blocks = nn.ModuleList([
            ResBlock3D(self.start_ch//2),
            ResBlock3D(self.start_ch//4),
            ResBlock3D(self.start_ch//8),
            ResBlock3D(self.start_ch//16),
        ])
        final_channels = self.start_ch // 16
        self.final_conv = nn.Sequential(
            nn.Conv3d(final_channels, final_channels, 3, padding=1),
            nn.InstanceNorm3d(final_channels),
            nn.ReLU(True),
            nn.Conv3d(final_channels, 1, 1)
        )

    def forward(self, x):
        B = x.shape[0]
        x = self.fc(x).view(B, self.start_ch, self.start_size, self.start_size, self.start_size)
        for up, res in zip(self.up_blocks, self.res_blocks):
            x = res(up(x))
        return self.final_conv(x)

class EncoderDecoder(nn.Module):
    def __init__(self, feature_dim=512, voxel_size=64, img_size=256, n_views=24):
        super().__init__()
        self.encoder = Encoder(feature_dim, img_size, n_views)
        self.view_fuser = ViewFusion(n_views, feature_dim)
        self.decoder = Decoder(feature_dim, voxel_size)
    def forward(self, views):
        B, V, C, H, W = views.shape
        feats = self.encoder(views.view(B*V, C, H, W)).view(B, V, -1)
        fused = self.view_fuser(feats)
        return self.decoder(fused)

# ============================================================
# LOSSES & METRICS
# ============================================================
def compute_metrics(preds, targets, threshold=0.5):
    preds = torch.sigmoid(preds.detach().cpu())
    targets = targets.detach().cpu()
    preds_bin = (preds > threshold).float()
    acc = (preds_bin == (targets > 0.5)).float().mean().item()
    inter = (preds_bin * (targets > 0.5)).float().sum()
    union = ((preds_bin + (targets > 0.5)) > 0).float().sum()
    iou = (inter / union).item() if union > 0 else 0
    return acc, iou

def dice_loss(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred)
    inter = (pred * target).sum(dim=(1,2,3,4))
    union = pred.sum(dim=(1,2,3,4)) + target.sum(dim=(1,2,3,4))
    dice = (2*inter + eps) / (union + eps)
    return 1 - dice.mean()

def iou_loss(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred)
    inter = (pred * target).sum(dim=(1,2,3,4))
    union = pred.sum(dim=(1,2,3,4)) + target.sum(dim=(1,2,3,4)) - inter
    return 1 - ((inter + eps) / (union + eps)).mean()

def hybrid_loss(pred, target):
    bce = Func.binary_cross_entropy_with_logits(pred, target)
    return 0.5*bce + 0.3*dice_loss(pred, target) + 0.2*iou_loss(pred, target)

# ============================================================
# TRAIN LOOP
# ============================================================
def run_epoch(model, dataloader, optimizer, criterion, device, train=True, scaler=None, accumulate_steps=1, max_grad_norm=1.0):
    model.train() if train else model.eval()
    total_loss, all_preds, all_targets = 0.0, [], []
    loop = tqdm(dataloader, desc="Train" if train else "Val")

    optimizer.zero_grad(set_to_none=True)
    step = 0
    for batch in loop:
        views = batch["views"].to(device, non_blocking=True)
        voxel = batch["voxel"].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            out = model(views)
            loss = criterion(out, voxel) / accumulate_steps

        if train:
            scaler.scale(loss).backward()
            step += 1
            if step % accumulate_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        total_loss += float(loss.detach().cpu() * accumulate_steps)
        all_preds.append(out.detach().cpu())
        all_targets.append(voxel.detach().cpu())

        del views, voxel, out, loss

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    acc, iou = compute_metrics(all_preds, all_targets)
    gc.collect()
    torch.cuda.empty_cache()
    return total_loss/len(dataloader), acc, iou

# ============================================================
# MAIN
# ============================================================
def main():
    config = {
        "root": "/kaggle/input/modelnet-out/ModelNet_out_2/ModelNet_out_2",
        "epochs": 200,
        "batch_size": 4,
        "accumulate_steps": 1,
        "lr": 1e-6,
        "feature_dim": 512,
        "voxel_size": 64,
        "num_views": 24,
        "img_size": 256,
        "checkpoint_path": "/kaggle/input/modelnet-out/ep75_encoder_decoder_64_frz(iou0.4761_loss0.2302).pth",
        "checkpoint_dir": "/kaggle/working/"
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    full_dataset = ShapeNetDataset(config["root"], config["num_views"], config["voxel_size"], img_size=config["img_size"], train=True)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    num_workers = min(4, os.cpu_count() or 2)
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=num_workers, pin_memory=True)

    model = EncoderDecoder(config["feature_dim"], config["voxel_size"], config["img_size"], config["num_views"]).to(device)
    if torch.cuda.device_count() > 1:
        print(f"Multi-GPU ({torch.cuda.device_count()}) enabled")
        model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda"))

    os.makedirs(config["checkpoint_dir"], exist_ok=True)

   
    # ============================================================
    # CHECKPOINT RESUME SECTION
    # ============================================================
    start_epoch, best_val_loss, best_val_iou = 0, float('inf'), 0.0
    ck_path = config["checkpoint_path"]
    
    if os.path.exists(ck_path):
        print(f"Attempting to resume from checkpoint: {ck_path}")
        ckpt = torch.load(ck_path, map_location=device)
        model_state = ckpt.get("model_state_dict", ckpt)

        # 1. Load model safely
        try:
            if isinstance(model, nn.DataParallel):
                model.module.load_state_dict(model_state, strict=False)
            else:
                model.load_state_dict(model_state, strict=False)
            print("Model weights loaded successfully.")
        except Exception as e:
            print(f"Partial model load: {e}")

        # 2. Load optimizer safely
        if "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                print("Optimizer state restored.")
            except Exception as e:
                print(f"Optimizer load skipped: {e}")

        # 3. Load epoch and metrics
        start_epoch = ckpt.get("epoch", 0)
        best_val_iou = ckpt.get("best_iou", ckpt.get("val_iou", 0.0))
        best_val_loss = ckpt.get("best_loss", ckpt.get("val_loss", float('inf')))
        print(f"Resumed from epoch {start_epoch}, best IoU={best_val_iou:.4f}, best Loss={best_val_loss:.4f}")
    else:
        print("No checkpoint found — starting new training.")

    # ============================================================
    # Unfreeze / Freeze Logic AFTER Loading
    # ============================================================
    encoder_ref = model.module.encoder if isinstance(model, nn.DataParallel) else model.encoder
    decoder_ref = model.module.decoder if isinstance(model, nn.DataParallel) else model.decoder

    # Unfreeze encoder fully
    for p in encoder_ref.parameters():
        p.requires_grad = True

    # Unfreeze decoder fully
    for p in decoder_ref.parameters():
        p.requires_grad = True
    
    # Unfreeze last two EfficientNetV2-S blocks (indices 6,7). Used for Fine-Tuning.
    #try:
        #features_container = getattr(encoder_ref, "backbone", None) or getattr(encoder_ref, "features", None)
        #if features_container is None:
            #raise RuntimeError("Encoder features not found (check attribute names).")

        #for block_idx in (6, 7):
            #if block_idx < len(features_container):
                #for p in features_container[block_idx].parameters():
                    #p.requires_grad = True
            #else:
                #print(f"Encoder lacks block index {block_idx} (len={len(features_container)}).")
    #except Exception as e:
        #print(f"Error unfreezing encoder blocks: {e}")

    # ============================================================
    # Rebuild Optimizer (correct param grouping)
    # ============================================================
    encoder_lr = 1e-6
    decoder_lr = 1e-6
    
    param_groups = []
    # Encoder trainable layers
    param_groups.append({"params": encoder_ref.parameters(), "lr": encoder_lr})

    # Decoder trainable
    param_groups.append({"params": decoder_ref.parameters(), "lr": decoder_lr})

    # View fuser
    view_fuser_ref = model.module.view_fuser if isinstance(model, nn.DataParallel) else model.view_fuser
    param_groups.append({"params": view_fuser_ref.parameters(), "lr": decoder_lr})

    optimizer = torch.optim.Adam(param_groups, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, verbose=True)

    # ============================================================
    # Sanity Check: Print Trainable Parameters
    # ============================================================
    enc_trainable = sum(p.numel() for p in encoder_ref.parameters() if p.requires_grad)
    dec_trainable = sum(p.numel() for p in decoder_ref.parameters() if p.requires_grad)
    print(f"Trainable encoder params: {enc_trainable:,}")
    print(f"Trainable decoder params: {dec_trainable:,}")

    for ep in range(start_epoch, config["epochs"]):
        train_loss, train_acc, train_iou = run_epoch(model, train_loader, optimizer, hybrid_loss, device, True, scaler)
        val_loss, val_acc, val_iou = run_epoch(model, val_loader, optimizer, hybrid_loss, device, False, scaler)

        print(f"[Epoch {ep+1}/{config['epochs']}] Train {train_acc:.4f}/{train_loss:.4f}/{train_iou:.4f} | Val {val_acc:.4f}/{val_loss:.4f}/{val_iou:.4f}")
        if torch.cuda.is_available():
            log_gpu_usage(); log_torch_gpu_memory()
        scheduler.step(1-val_iou)

        if val_iou > best_val_iou or val_loss < best_val_loss:
            best_val_iou = max(best_val_iou, val_iou)
            best_val_loss = min(best_val_loss, val_loss)
            ck_name = f"ep{ep+1}_encoder_decoder_64_frz(iou{val_iou:.4f}_loss{val_loss:.4f}).pth"
            save_path = os.path.join(config["checkpoint_dir"], ck_name)
            torch.save({
                "epoch": ep+1,
                "model_state_dict": model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_iou": best_val_iou,
                "best_loss": best_val_loss
            }, save_path)
            print(f"Saved best model → {save_path}")

        torch.cuda.empty_cache(); gc.collect()

if __name__ == "__main__":
    main()