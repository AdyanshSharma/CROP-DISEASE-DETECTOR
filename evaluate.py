import torch
import json
import time
from src.dataset import get_dataloaders
from src.model import get_model

DATA_DIR = "data/raw/PlantVillage"
BATCH_SIZE = 32
MODEL_PATH = "models/best_model.pth"
MAX_BATCHES = 20  # Limit to 20 batches for a quick estimate

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on: {device}", flush=True)
    
    train_loader, val_loader, classes = get_dataloaders(DATA_DIR, BATCH_SIZE)
    print(f"Validation batches: {len(val_loader)} (Will evaluate max {MAX_BATCHES} batches)", flush=True)
    
    model = get_model(num_classes=len(classes), pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    
    print("Starting evaluation...", flush=True)
    start_time = time.time()
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            if i >= MAX_BATCHES:
                break
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if (i+1) % 5 == 0:
                print(f"Processed {i+1} batches...", flush=True)
            
    accuracy = 100 * correct / total
    print(f"\nEstimated Validation Accuracy (on {total} images): {accuracy:.2f}%", flush=True)
    print(f"Time taken: {time.time() - start_time:.2f} seconds", flush=True)

if __name__ == "__main__":
    evaluate()
