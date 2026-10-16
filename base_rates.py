"""Share of 'good' applicants in each group, in the raw data (no model involved).
Tells you whether group gaps in model approvals could simply reflect different base rates.
Usage: python base_rates.py [--csv file.csv --target class]
"""
import argparse
from loan_eligibility import load, group_labels

ap = argparse.ArgumentParser(); ap.add_argument("--csv"); ap.add_argument("--target", default="class")
a = ap.parse_args()
X, y = load(a)
for attr, labels in group_labels(X).items():
    print(f"\n{attr}")
    for g in sorted(set(labels)):
        m = labels == g
        print(f"  {g:<14} n={m.sum():<4} share good = {y.values[m].mean():.3f}")
