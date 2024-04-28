import timm

# List all available models in timm
available_models = timm.list_models()
print(available_models)

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
from os.path import join
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim import AdamW
import timm

class AgeDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, annot_path, train=True):
        self.data_path = data_path
        self.annot_path = annot_path
        self.train = train
        self.ann = pd.read_csv(annot_path)
        self.files = self.ann['file_id']
        if train:
            self.ages = self.ann['age']
        self.transform = self._transform(train)

    def _transform(self, train):
        operations = [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip() if train else transforms.Lambda(lambda x: x),
            transforms.ColorJitter(0.5, 0.5, 0.5, 0.5) if train else transforms.Lambda(lambda x: x),
            transforms.RandomRotation(10) if train else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.5) if train else transforms.Lambda(lambda x: x)
        ]
        return transforms.Compose(operations)

    def read_img(self, file_name):
        im_path = join(self.data_path, file_name)
        with Image.open(im_path) as img:
            img = img.convert('RGB')
        return self.transform(img)

    def __getitem__(self, index):
        file_name = self.files[index]
        img = self.read_img(file_name)
        if self.train:
            age = self.ages[index]
            return img, age
        else:
            return img, file_name

    def __len__(self):
        return len(self.files)

def create_ensemble_models(device, model_names):
    models = {}
    for name in model_names:
        model = timm.create_model(name, pretrained=True, num_classes=1)
        model = model.to(device)
        if hasattr(model, 'classifier'):
            for param in model.classifier.parameters():
                param.requires_grad = True
        else:
            for param in model.fc.parameters():
                param.requires_grad = True
        models[name] = model
    return models

class ModelEnsemble(nn.Module):
    def __init__(self, models, weights):
        super(ModelEnsemble, self).__init__()
        self.models = nn.ModuleDict(models)
        self.weights = weights

    def forward(self, x):
        outputs = [self.weights[name] * model(x) for name, model in self.models.items()]
        return torch.sum(torch.stack(outputs), dim=0) / sum(self.weights.values())

def predict_ensemble(ensemble, dataloader, device):
    ensemble.eval()
    predictions = []
    file_ids = []
    with torch.no_grad():
        for images, ids in dataloader:
            images = images.to(device)
            outputs = ensemble(images)
            predicted_ages = outputs.squeeze().tolist()
            predictions.extend([int(round(age)) for age in predicted_ages])
            file_ids.extend(ids)
    return file_ids, predictions

def save_predictions_to_csv(file_ids, predictions, file_path):
    df = pd.DataFrame({'file_id': file_ids, 'age': predictions})
    df.to_csv(file_path, index=False)

def train_individual_model(model, dataloader, epochs, device):
    model.train()
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001)
    criterion = nn.MSELoss()
    scheduler = OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(dataloader), epochs=epochs)
    scaler = GradScaler(enabled=torch.cuda.is_available())

    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Training {model.__class__.__name__} - Epoch {epoch + 1}/{epochs}")
        for images, ages in progress_bar:
            images, ages = images.to(device), ages.to(device)
            optimizer.zero_grad()
            with autocast(enabled=torch.cuda.is_available()):
                outputs = model(images)
                loss = criterion(outputs.squeeze(), ages.float())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item() * images.size(0)
            progress_bar.set_postfix(loss=(total_loss / len(dataloader.dataset)))



def setup_and_train_models(train_loader, model_epochs, model_weights):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = create_ensemble_models(device, model_names=list(model_epochs.keys()))
    for name, model in models.items():
        print(f"Starting training for {name}")
        train_individual_model(model, train_loader, model_epochs[name], device)

    ensemble = ModelEnsemble(models, model_weights)
    return ensemble, device

# Paths for data
train_path = 'smai-24-age-prediction/content/faces_dataset/train'
train_ann = 'smai-24-age-prediction/content/faces_dataset/train.csv'
train_dataset = AgeDataset(train_path, train_ann, train=True)


test_path = 'smai-24-age-prediction/content/faces_dataset/test'
test_ann = 'smai-24-age-prediction/content/faces_dataset/submission.csv'
test_dataset = AgeDataset(test_path, test_ann, train=False)

# Setting up datasets and dataloaders
train_dataset = AgeDataset(train_path, train_ann, train=True)
test_dataset = AgeDataset(test_path, test_ann, train=False)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model_epochs = {
    'mobilenetv3_rw': 20,  # Use the correct, confirmed model name
    'resnet50': 50,
    'efficientnet_b0': 50
}
model_weights = {
    'mobilenetv3_rw': 0.5,  # Use the correct weight for the confirmed model
    'resnet50': 1.5,
    'efficientnet_b0': 1.0
}

# Train and predict
ensemble, device = setup_and_train_models(train_loader, model_epochs, model_weights)
file_ids, test_predictions = predict_ensemble(ensemble, test_loader, device)

# Save predictions
save_predictions_to_csv(file_ids, test_predictions, 'final_ages.csv')
