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
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
import joblib

save_path = ''
if not os.path.exists(save_path):
    os.makedirs(save_path)

'''para-based data'''

idx_df = pd.read_excel('', sheet_name='')
para_df = pd.read_excel('', sheet_name='')
para_np = para_df.to_numpy()

index = ['Age', 'Age1', 'Gender', 'patient-reported aMMP8', 'aMMP8 value(ng/mL)',
          'Smoking status', 'Q1: Self-assessment of Gum Disease', 'Q2: Self-assessment of rating on Gum/Teeth Health',
          'Q3a: Self-reported experience of Supra-gingival Scaling/Ultrasonic Dental Cleaning',
          'Q3b: Self-reported experience of Professional Periodontal Treatment',
          'Q4: Self-perceived Tooth Mobility', 'Q5: Self-reported Professionally Diagnosed Bone Loss',
          'Q6: Self-awareness of the change of tooth appearance', 'Q7: Use of Dental Floss/other device per week',
          'Q8: Use of Mouth Rinse per week', 'Q9: Self-awareness of bleeding on brushing',
          'Q10: Self-reported frequency of bleeding on brushing', 'Q11: Self-report of chewing problems',
          'Q12: Self-report of food intake decrease because of chewing problems',
          'Q13: Self-report of food type changes because of chewing problems'
          ]

para_df_v0 = para_df[['ID']+index]

n = len(para_df_v0)

for i in range(n):
    if para_df_v0.iloc[i, 14] == 'Blank':
        para_df_v0.iloc[i, 14] = 0
    if para_df_v0.iloc[i, 15] == 'Blank':
        para_df_v0.iloc[i, 15] = 0

for i in range(n):
    if para_df_v0.iloc[i, 5] == '<10':
        para_df_v0.iloc[i, 5] = 10
    para_df_v0.iloc[i, 5] = int(para_df_v0.iloc[i, 5])

para_np_v0 = para_df_v0.to_numpy()

encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
X_categorical_raw = np.concatenate((para_np_v0[:, 2:5], para_np_v0[:, 6:14], para_np_v0[:, 16:21]), axis=1)
X_categorical = encoder.fit_transform(X_categorical_raw)
joblib.dump(encoder, os.path.join(save_path, 'encoder.pkl'))

X_numerical = np.concatenate((para_np_v0[:, 1:2], para_np_v0[:, 5:6], para_np_v0[:, 14:16]), axis=1)
scaler = StandardScaler()
X_numerical = scaler.fit_transform(X_numerical)
joblib.dump(scaler, os.path.join(save_path, 'scaler.pkl'))

X_processed = np.hstack((X_numerical, X_categorical))
X_ID_y = para_df[['ID']].to_numpy().tolist()
X_ID = [item[0] for item in X_ID_y]
X_processed_df = pd.DataFrame(X_processed, index=X_ID)
len_para = np.shape(X_processed)[1]

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
        patient_id = os.path.basename(patient_folder).split('_')[0]
        label = torch.tensor(int(os.path.basename(patient_folder).split('_')[-1]))
        img_files = sorted([os.path.join(patient_folder, f) for f in os.listdir(patient_folder) if f.endswith('.jpg')])
        for img_file in img_files:
            img = Image.open(img_file).convert('RGB')
            if self.transform:
                img = self.transform(img)
            images.append(img)
        images = torch.stack(images)
        paras = X_processed_df.loc[patient_id].to_numpy()
        paras = torch.FloatTensor(paras)
        if self.mode == 'single':
            images = images[0].unsqueeze(0)
        return images, paras, label

train_path = ''
val_path = ''

train_dataset = PeriodontalDataset(train_path, transform=data_transforms['train'], mode='bag')
valid_dataset = PeriodontalDataset(val_path, transform=data_transforms['val'], mode='bag')
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True)

'''Network'''

class MultimodalModel(nn.Module):
    def __init__(self, num_classes, len_para):
        super(MultimodalModel, self).__init__()
        base_model = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(base_model.children())[:-2])
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc_img = nn.Linear(base_model.fc.in_features, 512)
        self.fc_img_out = nn.Linear(512, num_classes)

        self.fc_params = nn.Sequential(
            nn.Linear(len_para, 128), 
            nn.ReLU(),
            nn.Linear(128, 512)
        )

        self.fc_fusion = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x, params):
        batch_size, num_images, c, h, w = x.size()
        x = x.view(-1, c, h, w)
        features = self.features(x)
        pooled_features = self.gap(features).view(batch_size, num_images, -1)
        img_features = pooled_features.mean(dim=1)
        img_features = self.fc_img(img_features)
        param_features = self.fc_params(params)
        fusion_features = torch.cat((img_features, param_features), dim=1)
        output = self.fc_fusion(fusion_features)
        return output, features

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultimodalModel(num_classes=2, len_para=len_para).to(device)

'''Train and Validation'''

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    for images, paras, labels in dataloader:
        images, paras, labels = images.to(device), paras.to(device), labels.to(device)
        outputs, _ = model(images, paras)
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
        for images, paras, labels in dataloader:
            images, paras, labels = images.to(device), paras.to(device), labels.to(device)
            outputs, _ = model(images, paras)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    return total_loss / len(dataloader), accuracy

optimizer = Adam(model.parameters(), lr=0.00005)

criterion = nn.CrossEntropyLoss()
num_epochs = 25
model_save_path = save_path

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

