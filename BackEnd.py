# =====================================================
#  BACKEND: the AI model and the calculations.
#  No buttons, no pages and no Streamlit in this file.
#  Anything here could also run in a normal Python script.
# =====================================================
import json
import pandas as pd
import torch
import torch.nn as nn

# ---------- SETTINGS ----------
DECISION_THRESHOLD = 0.54   # approved when the probability is above this
MODEL_ACCURACY = None       # ask your teammate, then write e.g. "91%"

# ---------- LOAD THE MODEL (teammate's code, don't edit) ----------
with open("loan_meta.json") as f:
    meta = json.load(f)

model = nn.Sequential(
    nn.Linear(len(meta["feature_order"]), 32), nn.ReLU(),
    nn.Linear(32, 16), nn.ReLU(),
    nn.Linear(16, 1),
)
model.load_state_dict(torch.load("loan_model.pt", map_location="cpu"))
model.eval()

NEEDS = list(meta["num_cols"]) + list(meta["cat_cols"])   # the model's input columns

# ---------- PREDICTING ----------
def predict(applicant):
    """One applicant (a dictionary) -> approval probability between 0 and 1."""
    row = pd.DataFrame([applicant])
    nums = (row[meta["num_cols"]] - pd.Series(meta["mean"])) / pd.Series(meta["std"])
    full = pd.concat([nums, row[meta["cat_cols"]]], axis=1)[meta["feature_order"]]
    x = torch.tensor(full.values, dtype=torch.float32)
    with torch.no_grad():
        return torch.sigmoid(model(x)).item()

def predict_batch(X):
    """Same maths as predict(), for a whole table of applicants (used by the fairness audit)."""
    nums = (X[meta["num_cols"]] - pd.Series(meta["mean"])) / pd.Series(meta["std"])
    full = pd.concat([nums, X[meta["cat_cols"]]], axis=1)[meta["feature_order"]]
    x = torch.tensor(full.values.astype("float32"))
    with torch.no_grad():
        return torch.sigmoid(model(x)).numpy().ravel()

# ---------- SMALL TOOLS ----------
# The allowed range for each input (the model was only trained on these ranges)
LIMITS = {
    "income_annum": (200000, 9900000), "loan_amount": (300000, 39500000),
    "loan_term": (2, 20), "cibil_score": (300, 900),
    "bank_asset_value": (0, 15000000), "residential_assets_value": (0, 30000000),
    "no_of_dependents": (0, 5),
}

def clip(key, value):
    """Keeps a value inside its allowed range."""
    lo, hi = LIMITS[key]
    return max(lo, min(hi, value))

def with_changes(a, **changes):
    """Copy of an applicant with one or more values changed."""
    t = dict(a)
    t.update({k: float(v) for k, v in changes.items()})
    return t

def build_applicant(ans):
    """Turns the quiz answers into the dictionary the model expects."""
    return {
        "income_annum": float(ans["income"]), "loan_amount": float(ans["loan_amount"]),
        "loan_term": float(ans["loan_term"]), "cibil_score": float(ans["cibil"]),
        "bank_asset_value": float(ans["bank_assets"]),
        "residential_assets_value": float(ans["home_assets"]),
        "no_of_dependents": float(ans["dependents"]),
        "education": 1 if ans["education"] == "Graduate" else 0,
        "self_employed": 1 if ans["self_employed"] == "Yes" else 0,
    }

# ---------- IDEAS TO IMPROVE (each one is tested on the real model) ----------
def suggestions(a):
    base = predict(a)
    ideas = {
        "Raise your CIBIL score by 50 points": {"cibil_score": clip("cibil_score", a["cibil_score"] + 50)},
        "Reduce the loan amount by 20%": {"loan_amount": clip("loan_amount", a["loan_amount"] * 0.8)},
        "Add ₹5,00,000 to your bank assets": {"bank_asset_value": clip("bank_asset_value", a["bank_asset_value"] + 500000)},
        "Shorten the loan term by 2 years": {"loan_term": clip("loan_term", a["loan_term"] - 2)},
        "Lengthen the loan term by 2 years": {"loan_term": clip("loan_term", a["loan_term"] + 2)},
    }
    rows = []
    for name, change in ideas.items():
        new = predict(with_changes(a, **change))
        rows.append({"Idea": name, "New probability (%)": round(new * 100, 1),
                     "Change (pts)": round((new - base) * 100, 1)})
    return pd.DataFrame(rows).sort_values("Change (pts)", ascending=False).reset_index(drop=True)

def min_cibil(a):
    """The lowest CIBIL score at which the model would approve (everything else the same)."""
    for s in range(300, 901, 10):
        if predict(with_changes(a, cibil_score=s)) > DECISION_THRESHOLD:
            return s
    return None

def max_loan(a):
    """The biggest loan the model would approve (everything else the same)."""
    ok = [m for m in range(300000, 39500001, 500000)
          if predict(with_changes(a, loan_amount=m)) > DECISION_THRESHOLD]
    return max(ok) if ok else None

# ---------- WEAK POINTS (simple rules of thumb, NOT from the AI model) ----------
def weak_points(a):
    pts = []
    loan_inc = a["loan_amount"] / a["income_annum"]
    cover = (a["bank_asset_value"] + a["residential_assets_value"]) / a["loan_amount"]
    repay = a["loan_amount"] / a["loan_term"] / a["income_annum"]
    c = int(a["cibil_score"])
    if c < 650:
        pts.append(("HIGH", "Low credit score", f"A CIBIL score of {c} is in the range lenders usually see as weak."))
    elif c < 750:
        pts.append(("MEDIUM", "Credit score has room to grow", f"A CIBIL score of {c} is decent, but 750+ is generally seen as strong."))
    if loan_inc > 4:
        pts.append(("HIGH", "Loan is large for your income", f"The loan is {loan_inc:.1f}× your yearly income."))
    elif loan_inc > 2.5:
        pts.append(("MEDIUM", "Loan is high compared with income", f"The loan is {loan_inc:.1f}× your yearly income."))
    if cover < 1:
        pts.append(("HIGH" if cover < 0.5 else "MEDIUM", "Asset coverage gap", f"Your assets cover only {cover:.1f}× the loan amount."))
    if repay > 0.35:
        pts.append(("HIGH", "Heavy yearly repayment", f"Principal repayment alone is about {repay * 100:.0f}% of your yearly income."))
    elif repay > 0.20:
        pts.append(("MEDIUM", "Noticeable yearly repayment", f"Principal repayment is about {repay * 100:.0f}% of your yearly income."))
    if a["no_of_dependents"] >= 4:
        pts.append(("LOW", "High number of dependents", f"{int(a['no_of_dependents'])} dependents can reduce spare income."))
    if a["self_employed"] == 1:
        pts.append(("LOW", "Self-employed income", "Lenders may see self-employed income as less predictable."))
    order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    return sorted(pts, key=lambda p: order[p[0]])



