import json,torch,torch.nn as nn,pandas as pd
threshold=0.54
with open("loan_meta.json") as f:
    meta=json.load(f)

n_features=len(meta['feature_order'])

model=nn.Sequential(
    nn.Linear(n_features, 32), nn.ReLU(),
    nn.Linear(32, 16), nn.ReLU(),
    nn.Linear(16, 1)
)
model.load_state_dict(torch.load("loan_model.pt"))
model.eval()


def predict(applicant):
    row=pd.DataFrame([applicant])
    nums = (row[meta["num_cols"]] - pd.Series(meta["mean"])) / pd.Series(meta["std"])
    full = pd.concat([nums, row[meta["cat_cols"]]], axis=1)[meta["feature_order"]]
    x = torch.tensor(full.values, dtype=torch.float32)
    with torch.no_grad():
        return torch.sigmoid(model(x)).item()


def ask_applicant():
    print("Enter applicant details:\n")
    applicant = {
        "income_annum":             float(input("Annual income(min-200000): ")),
        "loan_amount":              float(input("Loan amount(300000-39500000): ")),
        "loan_term":                float(input("Loan term (years): ")),
        "cibil_score":              float(input("CIBIL score (300-900): ")),
        "bank_asset_value":         float(input("Bank asset value: ")),
        "residential_assets_value": float(input("Residential assets value: ")),
        "no_of_dependents":         float(input("Number of dependents: ")),
    }

    edu = input("Education (graduate / not graduate): ").strip().lower()
    applicant["education"] = 1 if edu == "graduate" else 0

    se = input("Self employed? (yes / no): ").strip().lower()
    applicant["self_employed"] = 1 if se == "yes" else 0

    return applicant


LIMITS = {
    "income_annum":             (200000, 9900000),
    "loan_amount":              (300000, 39500000),
    "loan_term":                (2, 20),
    "cibil_score":              (300, 900),
    "bank_asset_value":         (0, 15000000),
    "residential_assets_value": (0, 30000000),
    "no_of_dependents":         (0, 5),
}

def check(applicant):
    bad = [k for k, (lo, hi) in LIMITS.items() if not lo <= applicant[k] <= hi]
    return bad



import numpy as np
search=(
    ("loan_amount",-1,0.5),
    ("loan_term",-1,0.5),
    ("cibil_score",+1,0.10),
    ("bank_asset_value",+1,0.3)
)
def what_would_improve(applicant,step=100):
  found={}
  for key,direction,max_frac in search:
    start=applicant[key]
    end= start * (1+direction * max_frac)
    lo,hi=LIMITS[key]
    end=min(max(end,lo),hi)
    for i in np.linspace(start,end,step)[1:]:
      trial=dict(applicant)
      trial[key]=float(i)
      if predict(trial)>threshold:
        found[key]=(start,int(i))
        break
  return found




while True:
    applicant=ask_applicant()
    bad = check(applicant)
    if bad:
        print("Out of the range the model was trained on:", bad)
        continue
    prob=predict(applicant)
    print(prob)
    if prob>threshold:
        print("Approved")
    else:
        print('Rejected')
        improve=what_would_improve(applicant)
        if improve:
          print("Any one of these changes would make you approved for the loan")
          for name,(og,new) in improve.items():
              print(f"change {name} from {og} to {new}")
        else:
           print(f"No realistic change can make you eligible for the loan,sorry")
