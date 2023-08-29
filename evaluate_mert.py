import os

from transformers import Wav2Vec2FeatureExtractor
from transformers import AutoModel
from mlp import MERT_MLP
from utils import MTTDataset
from torch.utils.data import DataLoader
import torch
from tqdm.auto import tqdm
import numpy as np
from sklearn.metrics import average_precision_score, r2_score, roc_auc_score

# Music Representation Model
device = "cuda"
mert_model = AutoModel.from_pretrained("/hpctmp/e0589920/MERT-v1-330M", trust_remote_code=True).to(device)
processor = Wav2Vec2FeatureExtractor.from_pretrained("/hpctmp/e0589920/MERT-v1-330M", trust_remote_code=True)

# Downstream Tagging Model
model = MERT_MLP(1024, [], 50)
if os.path.exists("./models/mert.pth"):
    model.load_state_dict(torch.load("mert.pth", map_location='cpu'))
model.to(device)

print(f"Loading Train Data...")
train_data = MTTDataset(split="train")
train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True, num_workers=5, drop_last=True)
print(f"Train Size: {len(train_data)}")

print(f"Loading Validation Data...")
valid_data = MTTDataset(split="valid")
valid_dataloader = DataLoader(valid_data, batch_size=4, shuffle=True, num_workers=5, drop_last=True)
print(f"Validation Size: {len(valid_data)}")

print(f"Loading Test Data...")
test_data = MTTDataset(split="test")
test_dataloader = DataLoader(test_data, batch_size=4, shuffle=True, num_workers=5, drop_last=True)
print(f"Test Size: {len(test_data)}")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)


def train_epoch():
    model.train()
    pbar = tqdm(train_dataloader, total=len(train_dataloader))
    for X, Y in pbar:
        optimizer.zero_grad()
        X_list = [x for x in X.numpy()]
        inputs = processor(X_list, sampling_rate=24000, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = mert_model(**inputs, output_hidden_states=True)
        all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze()
        time_reduced_hidden_states = torch.swapaxes(all_layer_hidden_states.mean(-2), 0, 1)
        _, loss = model(time_reduced_hidden_states, Y.cuda())
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
        inputs = processor(X_list, sampling_rate=24000, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = mert_model(**inputs, output_hidden_states=True)
        all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze()
        time_reduced_hidden_states = torch.swapaxes(all_layer_hidden_states.mean(-2), 0, 1)
        logit, _ = model(time_reduced_hidden_states, Y.to(device))
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
        torch.save(model.state_dict(), "./models/mert.pth")
    else:
        miss_count += 1
    epoch += 1
    if miss_count > 2:
        break

print("Evaluating...")
model.load_state_dict(torch.load("./models/mert.pth"))
auc, ap = eval(split="test")
print(f"AUC: {auc}, AP: {ap}\n")
