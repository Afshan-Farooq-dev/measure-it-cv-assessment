"""Train a lightweight PyTorch U-Net on the fixed Colab train/validation split only."""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class SpongeDataset(Dataset):
    def __init__(self, root, split):
        rows = list(csv.DictReader((root / "manifest.csv").open(encoding="utf-8", newline="")))
        self.rows = [row for row in rows if row["split"] == split]
        if not self.rows:
            raise ValueError(f"No {split} records in {root / 'manifest.csv'}.")
        self.root, self.split = root, split

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        name = self.rows[index]["filename"]
        image = np.asarray(Image.open(self.root / self.split / "images" / name).convert("RGB"), dtype=np.float32) / 255.0
        mask = np.asarray(Image.open(self.root / self.split / "masks" / name).convert("L"), dtype=np.float32)[None] / 255.0
        return torch.from_numpy(image.transpose(2, 0, 1)), torch.from_numpy(mask)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
                                    nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.layers(x)


class UNet(nn.Module):
    def __init__(self, base=16):
        super().__init__()
        self.e1, self.e2, self.e3 = ConvBlock(3, base), ConvBlock(base, base * 2), ConvBlock(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base * 4, base * 8)
        self.u3, self.d3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2), ConvBlock(base * 8, base * 4)
        self.u2, self.d2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2), ConvBlock(base * 4, base * 2)
        self.u1, self.d1 = nn.ConvTranspose2d(base * 2, base, 2, 2), ConvBlock(base * 2, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(self.pool(e1)); e3 = self.e3(self.pool(e2)); b = self.bottleneck(self.pool(e3))
        d3 = self.d3(torch.cat((self.u3(b), e3), dim=1)); d2 = self.d2(torch.cat((self.u2(d3), e2), dim=1)); d1 = self.d1(torch.cat((self.u1(d2), e1), dim=1))
        return self.out(d1)


def dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    numerator = 2 * (probs * targets).sum(dim=(1, 2, 3)) + eps
    denominator = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + eps
    return 1 - (numerator / denominator).mean()


def evaluate(model, loader, bce, device):
    model.eval(); losses = []; ious = []; dices = []
    with torch.no_grad():
        for image, target in loader:
            image, target = image.to(device), target.to(device)
            logits = model(image); loss = bce(logits, target) + dice_loss(logits, target)
            prediction = torch.sigmoid(logits) >= .5
            target_bool = target >= .5
            intersection = (prediction & target_bool).sum(dim=(1, 2, 3)).float()
            union = (prediction | target_bool).sum(dim=(1, 2, 3)).float()
            ious.extend(((intersection + 1e-6) / (union + 1e-6)).cpu().tolist())
            dices.extend(((2 * intersection + 1e-6) / (prediction.sum(dim=(1, 2, 3)) + target_bool.sum(dim=(1, 2, 3)) + 1e-6)).cpu().tolist())
            losses.append(loss.item())
    return {"loss": float(np.mean(losses)), "iou": float(np.mean(ious)), "dice": float(np.mean(dices))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("./unet_run"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(f"Output already exists: {args.output}")
    if not torch.cuda.is_available(): raise RuntimeError("A CUDA GPU is required. In Colab select Runtime > Change runtime type > T4 GPU.")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    train, validation = SpongeDataset(args.dataset, "train"), SpongeDataset(args.dataset, "validation")
    if len(train) != 54 or len(validation) != 15: raise ValueError("Expected exactly 54 train and 15 validation images.")
    # Deliberately do not construct a test Dataset or DataLoader in this training script.
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    validation_loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    model = UNet().to(device); optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate); bce = nn.BCEWithLogitsLoss()
    args.output.mkdir(parents=True); history = []; best_iou = -1.; stale = 0; started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train(); train_losses = []
        for image, target in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            image, target = image.to(device), target.to(device); optimizer.zero_grad()
            logits = model(image); loss = bce(logits, target) + dice_loss(logits, target); loss.backward(); optimizer.step(); train_losses.append(loss.item())
        metrics = evaluate(model, validation_loader, bce, device); metrics.update(epoch=epoch, train_loss=float(np.mean(train_losses)), elapsed_seconds=time.perf_counter() - started); history.append(metrics)
        print(json.dumps(metrics))
        if metrics["iou"] > best_iou:
            best_iou, stale = metrics["iou"], 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "validation": metrics, "image_size": [256, 256], "threshold": .5}, args.output / "best_model.pth")
        else:
            stale += 1
            if stale >= args.patience: break
    with (args.output / "history.json").open("w", encoding="utf-8") as handle: json.dump(history, handle, indent=2)
    print(json.dumps({"best_validation_iou": best_iou, "epochs_completed": len(history), "training_duration_seconds": time.perf_counter() - started, "checkpoint": str(args.output / 'best_model.pth')}, indent=2))


if __name__ == "__main__": main()
