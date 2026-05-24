import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from dataset import get_dataloaders
from model import get_model

# ── Config ────────────────────────────────────────────
DATA_DIR   = "data/raw/PlantVillage"
EPOCHS     = 3
BATCH_SIZE = 32
LR         = 0.001
SAVE_PATH  = "models/best_model.pth"
# ──────────────────────────────────────────────────────

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Training on : {device}")

    # Data
    train_loader, val_loader, classes = get_dataloaders(DATA_DIR, BATCH_SIZE)
    print(f"📦 Classes      : {len(classes)}")

    # Model
    model = get_model(num_classes=len(classes)).to(device)

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.classifier.parameters(), lr=LR)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.5)

    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        # ── Training ──
        model.train()
        train_loss, correct, total = 0, 0, 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total   += labels.size(0)

            loop.set_postfix(loss=f"{train_loss/total:.4f}",
                             acc=f"{100.*correct/total:.2f}%")

        # ── Validation ──
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss    += loss.item()
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total   += labels.size(0)

        val_acc = 100. * val_correct / val_total
        print(f"📊 Epoch {epoch+1} | Val Loss: {val_loss/val_total:.4f} | Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"💾 Best model saved → {val_acc:.2f}%")

        scheduler.step()

    print(f"\n🎉 Training Complete! Best Accuracy: {best_val_acc:.2f}%")

if __name__ == "__main__":
    train()