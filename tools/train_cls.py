import os
import time
import copy
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from FMF.models import build_model
from FMF.datasets import build_dataset
from FMF.utils.metrics import ClassificationMetric

from types import SimpleNamespace

cfg = SimpleNamespace()
cfg.DATA = SimpleNamespace()
cfg.DATA.PATH_TO_DATA_DIR = "E:/xichao/deepcode/FMF-Benchmark-main/data/FMFBenchmarkV2/pixel-level"  # 改成你实际的数据路径
cfg.TASK = "classification"


def train_model(
    dataset_name="fmf_vicu",
    num_classes=2,
    batch_size=32,
    num_epochs=30,
    learning_rate=1e-3,
    val_ratio=0.2,
    save_path="state_dict_vicu_cls1.pth"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[INFO] Loading dataset...")
    full_dataset = build_dataset(name=dataset_name, cfg=cfg, split="train")

    # train / val split
    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print("[INFO] Building model...")
    model = build_model({"MODEL": {"NUM_CLASSES": num_classes}})
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        print(f"\n[Epoch {epoch+1}/{num_epochs}]")

        # ========== Train ==========
        model.train()
        running_loss, running_corrects = 0.0, 0
        for batch in tqdm(train_loader, ncols=100):
            y = batch[-1].to(device)
            x = [ipt.to(device) for ipt in batch[:-1]]

            optimizer.zero_grad()
            outputs = model(*x)
            loss = criterion(outputs, y)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * y.size(0)
            preds = torch.argmax(outputs, dim=1)
            running_corrects += torch.sum(preds == y).item()

        epoch_loss = running_loss / train_size
        epoch_acc = running_corrects / train_size
        print(f"Train Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}")

        # ========== Validation ==========
        model.eval()
        val_metric = ClassificationMetric(numClass=num_classes)
        with torch.no_grad():
            for batch in val_loader:
                y = batch[-1].to(device)
                x = [ipt.to(device) for ipt in batch[:-1]]

                outputs = model(*x)
                preds = torch.argmax(outputs, dim=1)
                val_metric.addBatch(preds.cpu().numpy(), y.cpu().numpy())

        val_acc = val_metric.Accuracy()
        val_f1 = val_metric.F1Score()
        print(f"Val Acc: {val_acc:.4f}  F1: {val_f1:.4f}")

        # ========== Save Best ==========
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, save_path)
            print(f"[INFO] New best model saved at epoch {epoch+1} with Acc {val_acc:.4f}")

    print("\nTraining complete. Best Val Acc: {:.4f}".format(best_acc))
    model.load_state_dict(best_model_wts)
    return model


if __name__ == "__main__":
    train_model()
