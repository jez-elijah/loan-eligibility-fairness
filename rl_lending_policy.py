"""Reinforcement learning view of the same problem: learn an approve/reject policy from reward.

Formulation: a contextual bandit (one decision per applicant, no sequence).
Reward = -cost from the German Credit cost matrix:
    approve good: 0 | approve bad: -5 | reject good: -1 | reject bad: 0
Bandit feedback: during training the agent only sees the reward of the action it took,
mirroring the real-world fact that rejected applicants' outcomes are never observed.
Agent: epsilon-greedy with a Q(x, action) regressor trained by SGD.
    --agent linear  (numpy, default)    --agent mlp  (PyTorch, needs torch)
Baselines on the same test split: supervised logistic regression + cost-optimal threshold
(from loan_eligibility.py), approve-all, reject-all.

Usage:  python rl_lending_policy.py [--csv file.csv --target class] [--agent mlp]
Writes results/rl_policy_results_<agent>.csv and rl_fairness_results_<agent>.csv
"""
import argparse
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from loan_eligibility import (load, preprocessor, best_threshold, group_labels,
                              COST_BAD_APPROVED, COST_GOOD_REJECTED)

REWARD = np.array([[-COST_GOOD_REJECTED, 0.0],      # action 0 = reject: [good, bad] -> rows are y=1,y=0 below
                   [0.0, -COST_BAD_APPROVED]])      # action 1 = approve


def reward(action, y):
    """action: 0 reject / 1 approve; y: 1 good / 0 bad."""
    return np.where(action == 1, np.where(y == 1, 0.0, -COST_BAD_APPROVED),
                    np.where(y == 1, -COST_GOOD_REJECTED, 0.0))


class LinearQ:
    def __init__(self, d, lr=0.01, seed=0):
        self.W, self.lr = np.zeros((2, d + 1)), lr

    def q(self, x):
        return self.W @ np.append(x, 1.0)

    def update(self, x, a, r):
        xb = np.append(x, 1.0)
        self.W[a] += self.lr * (r - self.W[a] @ xb) * xb

    def q_batch(self, X):
        return np.hstack([X, np.ones((len(X), 1))]) @ self.W.T


class MLPQ:
    def __init__(self, d, lr=1e-3, seed=0):
        import torch, torch.nn as nn
        torch.manual_seed(seed)
        self.torch = torch
        self.net = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, 2))
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)

    def q(self, x):
        with self.torch.no_grad():
            return self.net(self.torch.tensor(x, dtype=self.torch.float32)).numpy()

    def update(self, x, a, r):
        t = self.torch
        pred = self.net(t.tensor(x, dtype=t.float32))[a]
        loss = (pred - r) ** 2
        self.opt.zero_grad(); loss.backward(); self.opt.step()

    def q_batch(self, X):
        with self.torch.no_grad():
            return self.net(self.torch.tensor(X, dtype=self.torch.float32)).numpy()


def train(agent, X, y, epochs, eps0, rng):
    for ep in range(epochs):
        eps = max(0.05, eps0 * (1 - ep / epochs))
        for i in rng.permutation(len(X)):
            a = rng.integers(2) if rng.random() < eps else int(np.argmax(agent.q(X[i])))
            agent.update(X[i], a, float(reward(np.array(a), y[i])))   # only the taken action's reward is seen
    return agent


def run(args):
    X_df, y = load(args); y = y.values
    rows, fair = [], []
    for rep in range(args.repeats):
        rng = np.random.default_rng(rep)
        tr, te = train_test_split(np.arange(len(y)), test_size=0.25, stratify=y, random_state=rep)
        pre = preprocessor(X_df).fit(X_df.iloc[tr])
        f = lambda idx: (lambda m: m.toarray() if hasattr(m, "toarray") else m)(pre.transform(X_df.iloc[idx]))
        Xtr, Xte = f(tr), f(te)
        agent = (MLPQ if args.agent == "mlp" else LinearQ)(Xtr.shape[1])
        train(agent, Xtr, y[tr], args.epochs, args.eps, rng)
        policies = {"rl_" + args.agent: np.argmax(agent.q_batch(Xte), axis=1) == 1}
        sup = make_pipeline(preprocessor(X_df), LogisticRegression(max_iter=2000))
        oof = cross_val_predict(sup, X_df.iloc[tr], y[tr], cv=StratifiedKFold(5, shuffle=True, random_state=rep), method="predict_proba")[:, 1]
        thr = best_threshold(oof, y[tr]); sup.fit(X_df.iloc[tr], y[tr])
        policies["supervised_threshold"] = sup.predict_proba(X_df.iloc[te])[:, 1] >= thr
        policies["approve_all"] = np.ones(len(te), bool); policies["reject_all"] = np.zeros(len(te), bool)
        for name, approve in policies.items():
            cost = -reward(approve.astype(int), y[te]).mean()
            rows.append(dict(rep=rep, policy=name, cost_per_applicant=cost, approval_rate=approve.mean(),
                             bad_approved_rate=(approve & (y[te] == 0)).sum() / max((y[te] == 0).sum(), 1)))
            for g, labels in group_labels(X_df.iloc[te]).items():
                for grp in np.unique(labels):
                    m = labels == grp
                    good, bad = y[te][m] == 1, y[te][m] == 0
                    fair.append(dict(rep=rep, policy=name, attribute=g, group=grp, n=m.sum(), approval_rate=approve[m].mean(),
                                     good_rejected_rate=(~approve[m] & good).sum() / max(good.sum(), 1),
                                     bad_approved_rate=(approve[m] & bad).sum() / max(bad.sum(), 1)))
    res = pd.DataFrame(rows); res.to_csv(f"results/rl_policy_results_{args.agent}.csv", index=False)
    print(res.groupby("policy")[["cost_per_applicant", "approval_rate", "bad_approved_rate"]].agg(["mean", "std"]).round(3))
    fr = pd.DataFrame(fair); fr.to_csv(f"results/rl_fairness_results_{args.agent}.csv", index=False)
    fr = fr[~fr.policy.isin(["approve_all", "reject_all"])]
    print(fr.groupby(["policy", "attribute", "group"])[["n", "approval_rate", "good_rejected_rate", "bad_approved_rate"]].mean().round(3))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv"); ap.add_argument("--target", default="class")
    ap.add_argument("--agent", choices=["linear", "mlp"], default="linear")
    ap.add_argument("--repeats", type=int, default=5); ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--eps", type=float, default=0.5)
    run(ap.parse_args())
