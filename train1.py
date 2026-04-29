import os
import random
import torchvision.transforms as T
from pyexpat import model
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.models as models


class MultiModalDataset(Dataset):
    def __init__(self, root, transform=None, add_noise=False):
        self.samples = []
        self.transform = transform  # store properly
        self.add_noise = add_noise
        classes = ["clean", "caution", "dirty"]
        
        for label, cls in enumerate(classes):
            folder = os.path.join(root, cls)
            if not os.path.exists(folder):
                continue
            
            files = [f for f in os.listdir(folder) if "range" in f]
            for file in files:
                cam_file = file.replace("range", "cam")
                self.samples.append((
                    os.path.join(folder, file),
                    os.path.join(folder, cam_file),
                    label
                ))

        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        r_path, c_path, label = self.samples[idx]

        range_img = cv2.imread(r_path)
        cam_img = cv2.imread(c_path)

        range_img = cv2.resize(range_img, (224, 224))
        cam_img = cv2.resize(cam_img, (224, 224))

        # Convert to tensor
        lidar = torch.from_numpy(range_img).float().permute(2, 0, 1) / 255.0
        cam = torch.from_numpy(cam_img).float().permute(2, 0, 1) / 255.0

        if self.transform:
            cam = self.transform(cam)
            lidar = self.transform(lidar)
        if self.add_noise:
            lidar += torch.randn_like(lidar) * 0.02
            cam += torch.randn_like(cam) * 0.02

        return lidar, cam, torch.tensor(label).long()
    

# -------- MODEL --------
class MultiModalResNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.lidar_net = models.resnet18(weights="DEFAULT")
        self.cam_net = models.resnet18(weights="DEFAULT")

        for param in self.lidar_net.parameters():
            param.requires_grad = False

        for param in self.cam_net.parameters():
            param.requires_grad = False

        self.lidar_net.fc = nn.Identity()
        self.cam_net.fc = nn.Identity()

        self.fc = nn.Sequential(
            nn.Linear(512 * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(self, lidar, cam):
        f1 = self.lidar_net(lidar)
        f2 = self.cam_net(cam)

        fused = torch.cat([f1, f2], dim=1)
        return self.fc(fused)


# -------- TRAIN --------
def main():
    # 1. Define Augmentations for Training
    train_transform = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomErasing(p=0.2, scale=(0.02, 0.1)),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])
    dataset = MultiModalDataset("/home/vishnucharan/ROS/amthack/dataset", transform=train_transform)

    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size

    train_ds, _ = random_split(dataset, [train_size, val_size])
    _, val_ds   = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_ds, batch_size=64)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MultiModalResNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=3e-5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)    

    best_val_loss = float('inf')
    patience = 3
    counter = 0

    for epoch in range(15):
        model.train()
        total_loss = 0

        for i, (lidar, cam, label) in enumerate(train_loader):
            lidar, cam, label = lidar.to(device), cam.to(device), label.to(device)

            output = model(lidar, cam)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if i % 10 == 0:
                print(f"Epoch {epoch+1} | Batch {i} | Loss {loss.item():.4f}")
            
        avg_loss = total_loss / len(train_loader)
        # -------- VALIDATION --------
        model.eval()
        correct = 0
        total = 0
        val_loss = 0

        with torch.no_grad():
            for lidar, cam, label in val_loader:
                lidar, cam, label = lidar.to(device), cam.to(device), label.to(device)

                output = model(lidar, cam)
                loss = criterion(output, label)
                val_loss += loss.item()

                preds = torch.argmax(output, dim=1)
                correct += (preds == label).sum().item()
                total += label.size(0)
        
        val_loss /= len(val_loader)
        acc = correct / total if total > 0 else 0

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            counter += 1

        if counter >= patience:
            print("Early stopping triggered")
            break
        

        print(f"Epoch {epoch+1} | Train Loss: {total_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {acc:.4f}")



if __name__ == "__main__":
    main()