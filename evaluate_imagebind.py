import os

from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from mlp import SimpleMLP
from utils import MTTFileDataset
from torch.utils.data import DataLoader
import torch
from tqdm.auto import tqdm
import numpy as np
from sklearn.metrics import average_precision_score, r2_score, roc_auc_score

# Music Representation Model
device = "cuda"
imagebind = imagebind_model.imagebind_huge(pretrained=True)
imagebind.eval()
imagebind.to(device)

# Downstream Tagging Model
model = SimpleMLP(1024, [], 50, dropout_p=0.2)
if os.path.exists("./models/imagebind.pth"):
    model.load_state_dict(torch.load("./models/imagebind.pth", map_location='cpu'))
model.to(device)

print(f"Loading Train Data...")
train_data = MTTFileDataset(split="train")
train_dataloader = DataLoader(train_data, batch_size=8, shuffle=True, drop_last=True)
print(f"Train Size: {len(train_data)}")

print(f"Loading Validation Data...")
valid_data = MTTFileDataset(split="valid")
valid_dataloader = DataLoader(valid_data, batch_size=8, shuffle=True, drop_last=True)
print(f"Validation Size: {len(valid_data)}")

print(f"Loading Test Data...")
test_data = MTTFileDataset(split="test")
test_dataloader = DataLoader(test_data, batch_size=8, shuffle=True, drop_last=True)
print(f"Test Size: {len(test_data)}")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)


def train_epoch():
    model.train()
    pbar = tqdm(train_dataloader, total=len(train_dataloader))
    for X, Y in pbar:
        optimizer.zero_grad()
        with torch.no_grad():
            audio_embed = imagebind({ModalityType.AUDIO: data.load_and_transform_audio_data(X, device)})[
                ModalityType.AUDIO]
        _, loss = model(audio_embed, Y.cuda())
        pbar.set_description(f"Loss: {loss.detach().cpu().numpy()}")
        loss.backward()
        optimizer.step()


if not os.path.exists("./models"):
    os.makedirs("./models")


def eval(split="valid"):
    model.eval()
    logits = []
    y = []
    if split == "valid":
        pbar = tqdm(valid_dataloader, total=len(valid_dataloader))
    else:
        pbar = tqdm(test_dataloader, total=len(test_dataloader))
    for X, Y in pbar:
        with torch.no_grad():
            audio_embed = imagebind({ModalityType.AUDIO: data.load_and_transform_audio_data(X, device)})[
                ModalityType.AUDIO]
        logit, _ = model(audio_embed, Y.to(device))
        logits.append(logit.detach())
        y.append(Y.long())
    logits = torch.cat(logits, dim=0)
    y = torch.cat(y, dim=0).numpy()
    with torch.no_grad():
        y_probs = (
            torch.sigmoid(logits)
            .cpu()
            .numpy()
        )
    return roc_auc_score(y, y_probs, average="macro"), average_precision_score(y, y_probs, average="macro")


print("Training...")
best_auc = 0
miss_count = 0
epoch = 0
while True:
    print(f"Epoch {epoch + 1}")
    train_epoch()
    auc, ap = eval()
    print(f"AUC: {auc}, AP: {ap}\n")
    if auc > best_auc:
        best_auc = auc
        miss_count = 0
        torch.save(model.state_dict(), "./models/imagebind.pth")
    else:
        miss_count += 1
    epoch += 1
    if miss_count >= 5:
        break

print("Evaluating...")
model.load_state_dict(torch.load("./models/imagebind.pth", map_location='cpu'))
auc, ap = eval(split="test")
print(f"AUC: {auc}, AP: {ap}\n")
