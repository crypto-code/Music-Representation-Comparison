import os

import laion_clap
from mlp import SimpleMLP
from utils import MTTDataset
from torch.utils.data import DataLoader
import torch
from tqdm.auto import tqdm
import numpy as np
from sklearn.metrics import average_precision_score, r2_score, roc_auc_score

# Music Representation Model
clap = laion_clap.CLAP_Module(enable_fusion=False)
clap.load_ckpt()

# Downstream Tagging Model
device = "cuda"
model = SimpleMLP(512, [256, 128], 50, dropout_p=0.05)
if os.path.exists("./models/clap.pth"):
    model.load_state_dict(torch.load("./models/clap.pth", map_location='cpu'))
model.to(device)

print(f"Loading Train Data...")
train_data = MTTDataset(split="train", sr=48000)
train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True, drop_last=True)
print(f"Train Size: {len(train_data)}")

print(f"Loading Validation Data...")
valid_data = MTTDataset(split="valid", sr=48000)
valid_dataloader = DataLoader(valid_data, batch_size=32, shuffle=True, drop_last=True)
print(f"Validation Size: {len(valid_data)}")

print(f"Loading Test Data...")
test_data = MTTDataset(split="test", sr=48000)
test_dataloader = DataLoader(test_data, batch_size=32, shuffle=True, drop_last=True)
print(f"Test Size: {len(test_data)}")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)


def train_epoch():
    model.train()
    pbar = tqdm(train_dataloader, total=len(train_dataloader))
    for X, Y in pbar:
        optimizer.zero_grad()
        X_list = [x for x in X.numpy()]
        audio_embed = torch.tensor(clap.get_audio_embedding_from_data(x = X_list, use_tensor=False))
        _, loss = model(audio_embed.cuda(), Y.cuda())
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
        X_list = [x for x in X.numpy()]
        audio_embed = torch.tensor(clap.get_audio_embedding_from_data(x = X_list, use_tensor=False))
        logit, _ = model(audio_embed.cuda(), Y.to(device))
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
        torch.save(model.state_dict(), "./models/clap.pth")
    else:
        miss_count += 1
    epoch += 1
    if miss_count >= 5:
        break

print("Evaluating...")
model.load_state_dict(torch.load("./models/clap.pth", map_location='cpu'))
auc, ap = eval(split="test")
print(f"AUC: {auc}, AP: {ap}\n")
