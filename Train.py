import kagglehub
import pandas as pd
import torch 
import torch.nn as nn 
import numpy as np
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

# Download latest version
path = ("C:/Users/Arnav/.cache/kagglehub/datasets/architsharma01/loan-approval-prediction-dataset/versions/1/loan_approval_dataset.csv")
df=pd.read_csv(path)
num_cols=['income_annum','loan_amount','loan_term','cibil_score','bank_asset_value','residential_assets_value','no_of_dependents']
cat_cols = ["education", "self_employed"]
df.columns = df.columns.str.strip()
for c in df.select_dtypes(include=["object","string"]):
    df[c] = df[c].str.strip()
df=df.drop(columns=['loan_id'])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df["education"] = (df["education"] == "Graduate").astype(int)
df["self_employed"] = (df["self_employed"] == "Yes").astype(int)
train = df.iloc[:3360].copy()
test=df.iloc[3360:].copy()
y_train = (train["loan_status"] == "Approved").astype(int)
y_test  = (test["loan_status"] == "Approved").astype(int)
X_train = train.drop(columns="loan_status")
X_test  = test.drop(columns="loan_status")

mean=X_train[num_cols].mean()
std=X_train[num_cols].std()
X_train_scaled = (X_train[num_cols] - mean) / std
X_test_scaled = (X_test[num_cols] - mean) / std
X_train_final = pd.concat([X_train_scaled, X_train[cat_cols]], axis=1)
X_test_final = pd.concat([X_test_scaled, X_test[cat_cols]], axis=1)

Xtr = torch.tensor(X_train_final.values, dtype=torch.float32)
Xte = torch.tensor(X_test_final.values, dtype=torch.float32)
ytr = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
yte = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

model=nn.Sequential(
    nn.Linear(9,32), nn.ReLU(),
    nn.Linear(32,16),nn.ReLU(),
    nn.Linear(16,1),
)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)



for epoch in range(200):
    model.train()
    logits=model(Xtr)
    loss=loss_fn(logits,ytr)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch%20==0:
        model.eval()
        with torch.no_grad():
             pred = (model(Xte) > 0).int().squeeze().numpy()
        print(classification_report(yte.int().squeeze().numpy(), pred,
                            target_names=["Rejected", "Approved"]))


import json

torch.save(model.state_dict(),'loan_model.pt')

meta ={
    "num_cols": num_cols,
    "cat_cols":cat_cols,
    "mean":mean.to_dict(),
    "std": std.to_dict(),
    "feature_order": X_train_final.columns.tolist(),
}

with open("loan_meta.json","w") as f:
    json.dump(meta,f)
sample = test.sample(1)
print(sample.T)

def accuracy(X):
    model.eval()
    with torch.no_grad():
        return ((model(X) > 0).float() == yte).float().mean().item()

base = accuracy(Xte)
results = {}

for i, col in enumerate(X_test_final.columns):
    drops = []
    for _ in range(10):                          # repeat, since shuffling is random
        Xp = Xte.clone()
        Xp[:, i] = Xp[torch.randperm(len(Xp)), i]    # shuffle just this column
        drops.append(base - accuracy(Xp))
    results[col] = sum(drops) / len(drops)

for col, d in sorted(results.items(), key=lambda kv: -kv[1]):
    print(f"{col:28s} accuracy drop: {d:.3f}")