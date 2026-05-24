import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_dataloaders(data_dir, batch_size=32, img_size=224):
    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(data_dir, transform=train_transforms)

    train_size = int(0.8 * len(full_dataset))
    val_size   = len(full_dataset) - train_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])

    val_set.dataset.transform = val_transforms

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, full_dataset.classes

if __name__ == "__main__":
    train_loader, val_loader, classes = get_dataloaders("data/raw/PlantVillage")
    print(f"✅ Classes found : {len(classes)}")
    print(f"✅ Train batches : {len(train_loader)}")
    print(f"✅ Val batches   : {len(val_loader)}")