"""Loan eligibility classifier with a cost-sensitive threshold and a fairness audit.

Dataset: German Credit ("credit-g", 1,000 applicants) from OpenML, fetched automatically.
Target: 'good' credit risk = eligible for the loan, 'bad' = not eligible.
IMPORTANT: this is a credit-RISK label, not a record of real "big loan" decisions.

Usage:
    python loan_eligibility.py                 # downloads credit-g from OpenML
    python loan_eligibility.py --csv file.csv --target class   # your own CSV
Writes results/model_results.csv and results/fairness_results.csv
"""
import argparse, warnings
import numpy as np, pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
COST_BAD_APPROVED, COST_GOOD_REJECTED = 5, 1   # German Credit's published cost matrix
SENSITIVE = ["age", "personal_status", "foreign_worker"]


class TorchMLP(BaseEstimator, ClassifierMixin):
    """Small PyTorch MLP with a scikit-learn interface (optional)."""
    def __init__(self, hidden=32, epochs=150, lr=1e-3, wd=1e-3, seed=0):
        self.hidden, self.epochs, self.lr, self.wd, self.seed = hidden, epochs, lr, wd, seed

    def fit(self, X, y):
        import torch, torch.nn as nn
        torch.manual_seed(self.seed)
        X = torch.tensor(np.asarray(X.todense() if hasattr(X, "todense") else X), dtype=torch.float32)
        y = torch.tensor(np.asarray(y), dtype=torch.float32)
        self.classes_ = np.array([0, 1])
        self.net_ = nn.Sequential(nn.Linear(X.shape[1], self.hidden), nn.ReLU(), nn.Dropout(0.3), nn.Linear(self.hidden, 1))
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr, weight_decay=self.wd)
        loss_fn = nn.BCEWithLogitsLoss()
        for _ in range(self.epochs):
            self.net_.train(); opt.zero_grad()
            loss_fn(self.net_(X).squeeze(-1), y).backward(); opt.step()
        return self

    def predict_proba(self, X):
        import torch
        X = torch.tensor(np.asarray(X.todense() if hasattr(X, "todense") else X), dtype=torch.float32)
        self.net_.eval()
        with torch.no_grad():
            p = torch.sigmoid(self.net_(X).squeeze(-1)).numpy()
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def load(args):
    if args.csv:
        df = pd.read_csv(args.csv)
        y = (df[args.target].astype(str).str.lower().isin(["good", "1", "yes", "approved"])).astype(int)
        return df.drop(columns=[args.target]), y
    from sklearn.datasets import fetch_openml
    d = fetch_openml("credit-g", version=1, as_frame=True)
    return d.data, (d.target == "good").astype(int)


def preprocessor(X):
    cat = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num = [c for c in X.columns if c not in cat]
    return ColumnTransformer([("num", StandardScaler(), num), ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])


def models(X, include_torch):
    m = {
        "logreg": make_pipeline(preprocessor(X), LogisticRegression(max_iter=2000)),
        "grad_boost": make_pipeline(preprocessor(X), HistGradientBoostingClassifier(random_state=0)),
    }
    if include_torch:
        try:
            import torch  # noqa: F401
            m["pytorch_mlp"] = make_pipeline(preprocessor(X), TorchMLP())
        except ImportError:
            print("torch not installed: skipping pytorch_mlp")
    return m


def best_threshold(p, y):
    """Threshold on P(good) that minimizes total cost on the given predictions."""
    grid = np.linspace(0.05, 0.95, 91)
    costs = [COST_BAD_APPROVED * ((p >= t) & (y == 0)).sum() + COST_GOOD_REJECTED * ((p < t) & (y == 1)).sum() for t in grid]
    return grid[int(np.argmin(costs))]


def group_labels(X):
    g = {}
    if "age" in X: g["age"] = np.where(X["age"] < 25, "under 25", "25 and over")
    if "foreign_worker" in X: g["foreign_worker"] = X["foreign_worker"].astype(str).values
    if "personal_status" in X:
        g["personal_status"] = np.where(X["personal_status"].astype(str).str.contains("female"), "female", "male")
    return g


def run(args):
    X, y = load(args)
    y = y.values
    results, fair = [], []
    for variant, cols in [("with_sensitive", list(X.columns)),
                          ("without_sensitive", [c for c in X.columns if c not in SENSITIVE])]:
        Xv = X[cols]
        for name, model in models(Xv, args.torch).items():
            for rep in range(args.repeats):
                tr, te = train_test_split(np.arange(len(y)), test_size=0.25, stratify=y, random_state=rep)
                inner = StratifiedKFold(5, shuffle=True, random_state=rep)
                oof = cross_val_predict(model, Xv.iloc[tr], y[tr], cv=inner, method="predict_proba")[:, 1]
                thr = best_threshold(oof, y[tr])           # threshold chosen on training data only
                model.fit(Xv.iloc[tr], y[tr])
                p = model.predict_proba(Xv.iloc[te])[:, 1]
                approve = p >= thr
                cost = COST_BAD_APPROVED * (approve & (y[te] == 0)).sum() + COST_GOOD_REJECTED * (~approve & (y[te] == 1)).sum()
                results.append(dict(variant=variant, model=name, rep=rep, auc=roc_auc_score(y[te], p),
                                    threshold=thr, cost_per_applicant=cost / len(te),
                                    approval_rate=approve.mean(), bad_approved_rate=(approve & (y[te] == 0)).sum() / max((y[te] == 0).sum(), 1)))
                # fairness audit always uses the full feature frame to read group membership
                for gname, labels in group_labels(X.iloc[te]).items():
                    for grp in np.unique(labels):
                        mask = labels == grp
                        good, bad = (y[te][mask] == 1), (y[te][mask] == 0)
                        fair.append(dict(variant=variant, model=name, rep=rep, attribute=gname, group=grp, n=mask.sum(),
                                         approval_rate=approve[mask].mean(),
                                         good_rejected_rate=(~approve[mask] & good).sum() / max(good.sum(), 1),
                                         bad_approved_rate=(approve[mask] & bad).sum() / max(bad.sum(), 1)))
    res, fr = pd.DataFrame(results), pd.DataFrame(fair)
    res.to_csv("results/model_results.csv", index=False); fr.to_csv("results/fairness_results.csv", index=False)
    pd.set_option("display.width", 160)
    print(res.groupby(["variant", "model"])[["auc", "cost_per_applicant", "approval_rate"]].agg(["mean", "std"]).round(3))
    print(fr.groupby(["variant", "model", "attribute", "group"])[["n", "approval_rate", "good_rejected_rate", "bad_approved_rate"]].mean().round(3))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv"); ap.add_argument("--target", default="class")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--torch", action="store_true", help="also train the PyTorch MLP")
    run(ap.parse_args())
