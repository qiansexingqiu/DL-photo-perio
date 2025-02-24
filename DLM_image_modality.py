import os
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.optim import Adam
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging

'''Dataset'''

class PeriodontalDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='bag'):
        self.root_dir = root_dir
        self.patients = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        patient_folder = self.patients[idx]
        images = []
        label = torch.tensor(int(os.path.basename(patient_folder).split('_')[-1]))
        img_files = sorted([os.path.join(patient_folder, f) for f in os.listdir(patient_folder) if f.endswith('.jpg')])
        for img_file in img_files:
            img = Image.open(img_file).convert('RGB')
            if self.transform:
                img = self.transform(img)
            images.append(img)
        images = torch.stack(images)
        if self.mode == 'single':
            images = images[0].unsqueeze(0)
        return images, label

def resize_and_crop(img, output_size=(224, 300)):
    w, h = img.size
    if w < h:
        new_w, new_h = 224, int(h * 224 / w)
    else:
        new_w, new_h = int(w * 224 / h), 224
    img = TF.resize(img, (new_h, new_w))

    img = TF.center_crop(img, output_size)
    return img

data_transforms = {
    'train': transforms.Compose([
        transforms.Lambda(lambda img: resize_and_crop(img)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Lambda(lambda img: resize_and_crop(img)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
}

train_path = ''
val_path = ''

train_dataset = PeriodontalDataset(train_path, transform=data_transforms['train'], mode='bag')
valid_dataset = PeriodontalDataset(val_path, transform=data_transforms['val'], mode='bag')
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True)

'''Network'''

# GAP based MIL model
class ModifiedMILModel(nn.Module):
    def __init__(self, num_classes):
        super(ModifiedMILModel, self).__init__()
        base_model = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(base_model.children())[:-2])
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(base_model.fc.in_features, num_classes)

    def forward(self, x):
        batch_size, num_images, c, h, w = x.size()
        x = x.view(-1, c, h, w)
        features = self.features(x)
        pooled_features = self.gap(features).view(batch_size, num_images, -1)
        pooled_features = pooled_features.mean(dim=1)
        output = self.fc(pooled_features)
        return output, features

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ModifiedMILModel(num_classes=2).to(device)

'''Training and Validation'''

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs, _ = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    return total_loss / len(dataloader), accuracy

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs, _ = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    return total_loss / len(dataloader), accuracy

optimizer = Adam(model.parameters(), lr=0.00005)
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

criterion = nn.CrossEntropyLoss()
num_epochs = 25
model_save_path = ''
os.makedirs(model_save_path, exist_ok=True)

logging.basicConfig(filename=model_save_path+'/log_file.txt', level=logging.INFO)
logger = logging.getLogger()

train_loss_list = []
train_acc_list = []
val_loss_list = []
val_acc_list = []
for epoch in tqdm(range(num_epochs)):
    train_loss, train_accuracy = train_one_epoch(model, train_loader, optimizer, criterion, device)
    train_loss_list.append(train_loss)
    train_acc_list.append(train_accuracy)
    val_loss, val_accuracy = validate(model, valid_loader, criterion, device)
    val_loss_list.append(val_loss)
    val_acc_list.append(val_accuracy)
    # scheduler.step(val_loss)
    logger.info(f'Epoch {epoch+1}: Train Loss: {train_loss}, Train Accuracy: {train_accuracy}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}')
    epoch_save_path = os.path.join(model_save_path, f'model_epoch_{epoch+1}.pth')
    torch.save(model.state_dict(), epoch_save_path)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_loss_list, label='Train Loss')
plt.plot(val_loss_list, label='Val Loss')
plt.title('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_acc_list, label='Train Acc')
plt.plot(val_acc_list, label='Val Acc')
plt.title('Accuracy')
plt.legend()

plt.savefig(os.path.join(model_save_path, 'curve.png'))
plt.show()
