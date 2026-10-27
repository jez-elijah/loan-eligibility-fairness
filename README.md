# Loan Eligibility Classifier with a Fairness Audit and an RL Policy

A classifier that predicts whether a bank applicant is a good credit risk (eligible for a loan), with a cost-sensitive approval threshold, an audit of how errors fall across age, sex, and foreign-worker groups, and a reinforcement learning formulation of the same decision.

**Scope note.** The German Credit dataset (1,000 applicants, OpenML `credit-g`) labels credit risk as good or bad. It is not a record of real "big loan" approval decisions, and this project does not claim to reproduce any bank's policy.

## Data
German Credit (Statlog), OpenML `credit-g`, downloaded automatically on first run. Original source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/144), donated 1994. Licensed CC BY 4.0.

Hofmann, H. (1994). *Statlog (German Credit Data)* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5NC77

## Questions
1. Which model ranks applicants best (AUC): logistic regression, gradient boosting, or a small PyTorch MLP?
2. Which approval threshold minimizes cost when approving a bad applicant is 5 times worse than rejecting a good one (the dataset's published cost matrix)? The threshold is chosen on training data only.
3. Do error rates differ across groups, and does dropping age, personal status, and foreign-worker status from the features close the gap, or do other columns act as proxies?
4. Can a reinforcement learning agent that sees only the outcomes of applicants it approves match a supervised model that sees every label?

## Run it
```bash
pip install -r requirements.txt
python loan_eligibility.py --repeats 10 --torch
python rl_lending_policy.py --agent linear --repeats 10
python rl_lending_policy.py --agent mlp --repeats 10
python base_rates.py
```
RL outputs go to `results/rl_policy_results_<agent>.csv` and `results/rl_fairness_results_<agent>.csv`, one pair per agent (`linear`, `mlp`).

## Method
Repeated stratified 75/25 splits (10 repeats, seeds 0 to 9). For each repeat the threshold is chosen from 5-fold out-of-fold predictions on the training split, then applied once to the held-out test split. Every model runs with and without the sensitive columns (age, personal status, foreign worker). Group metrics use the original group labels either way.

**RL extension.** Each approve/reject decision is a contextual bandit with reward equal to the negative cost (approve good 0, approve bad -5, reject good -1, reject bad 0). During training the agent sees only the reward of the action it took, so rejected applicants' outcomes stay hidden, as in real lending. Epsilon-greedy agent with a linear or PyTorch MLP Q-function, compared with the supervised threshold model and trivial policies on the same splits.

## Results (mean ± std over 10 repeats)

| Variant | Model | AUC | Cost per applicant | Approval rate |
|---|---|---|---|---|
| with sensitive columns | logreg | 0.782 ± 0.034 | 0.553 ± 0.042 | 0.385 ± 0.089 |
| with sensitive columns | grad_boost | 0.780 ± 0.025 | 0.548 ± 0.056 | 0.378 ± 0.031 |
| with sensitive columns | pytorch_mlp | 0.790 ± 0.034 | 0.538 ± 0.065 | 0.402 ± 0.095 |
| without sensitive columns | logreg | 0.779 ± 0.035 | 0.550 ± 0.058 | 0.378 ± 0.064 |
| without sensitive columns | grad_boost | 0.785 ± 0.031 | 0.554 ± 0.047 | 0.408 ± 0.070 |
| without sensitive columns | pytorch_mlp | 0.787 ± 0.029 | 0.519 ± 0.060 | 0.428 ± 0.056 |

Policy comparison (cost per applicant; lower is better):

| Policy | Cost per applicant | Approval rate | Bad applicants approved |
|---|---|---|---|
| approve_all | 1.500 | 1.000 | 1.000 |
| reject_all | 0.700 | 0.000 | 0.000 |
| supervised_threshold (logreg) | 0.553 ± 0.042 | 0.385 ± 0.089 | 0.132 ± 0.041 |
| rl_linear | 0.545 ± 0.053 | 0.282 ± 0.101 | 0.071 ± 0.050 |
| rl_mlp | 0.637 ± 0.091 | 0.536 ± 0.082 | 0.263 ± 0.078 |

Group results, logistic regression (mean over 10 test splits):

| Group | Approval rate (with sensitive cols) | Good applicants rejected (with) | Approval rate (without) | Good applicants rejected (without) |
|---|---|---|---|---|
| age 25 and over (n≈212) | 0.410 | 0.484 | 0.402 | 0.492 |
| age under 25 (n≈38) | 0.248 | 0.649 | 0.246 | 0.653 |
| personal status female (n≈74) | 0.316 | 0.566 | 0.364 | 0.512 |
| personal status male (n≈176) | 0.414 | 0.484 | 0.384 | 0.515 |

Share of good applicants in each group, raw data (`base_rates.py`):

| Group | n | Share good |
|---|---|---|
| age 25 and over | 851 | 0.719 |
| age under 25 | 149 | 0.591 |
| personal status female | 310 | 0.648 |
| personal status male | 690 | 0.723 |

## Findings and limitations

**Findings**
- The three supervised models are not clearly different: AUC 0.78 to 0.79 with a standard deviation of about 0.03 across repeats. The PyTorch MLP's slightly better mean is well within that spread. Differences are within the run-to-run spread (no significance test was run), so the data don't support ranking the models.
- Every model beats the trivial policies. Cost per applicant is about 0.52 to 0.55, against 0.70 for rejecting everyone and 1.50 for approving everyone.
- Age gap: under-25 applicants are approved at about 0.6 times the rate of applicants 25 and over (0.25 vs 0.41 for logistic regression), and a good applicant under 25 is rejected 65% of the time versus 48%. Removing the age column left the logistic regression approval gap essentially unchanged (about 0.16), narrowed it for gradient boosting (0.22 to 0.15), and barely moved it for the MLP (0.19 to 0.18). This is consistent with other columns acting as proxies for age.
- Personal-status gap: women's approval rate was lower with the column present (0.32 vs 0.41) and the gap shrank to about 0.02 once it was removed. This is consistent with the model using the column directly rather than through proxies.
- Base rates: under-25 applicants really do have a lower share of good outcomes in this data (0.59 vs 0.72), and women a lower share than men (0.65 vs 0.72). That explains part of the approval-rate gaps, but not all of them: the supervised model approves under-25 applicants at 0.6 times the rate of older applicants (0.25 vs 0.41), against a base-rate ratio of 0.82, and it rejects 65% of good under-25 applicants versus 48% of good applicants 25 and over. Among truly good applicants, young ones are treated worse. Because base rates differ, the common fairness criteria (equal approval rates, equal error rates) cannot all hold at once, so any claim of fairness must say which one it uses.
- RL linear agent (trained with the sensitive columns as features): it matched the supervised model on cost (0.545 vs 0.553, a difference far smaller than the spread across repeats) while approving fewer applicants (0.28 vs 0.39) and fewer bad ones (7% vs 13%). It did not treat groups more evenly: under-25 applicants were approved 0.16 of the time vs 0.30 for 25 and over (supervised: 0.25 vs 0.41), and good under-25 applicants were rejected 77% of the time vs 61% (supervised: 65% vs 48%). Women were approved less often than men (0.23 vs 0.30). It matched the supervised cost by rejecting more good applicants overall, not by treating groups more evenly.
- RL MLP agent: it was worse and unstable (cost 0.637 ± 0.091, against 0.553 for the supervised model) and barely beat rejecting everyone (0.70). It approved far more applicants (0.54) and far more bad ones (26% of bad applicants). Its group gaps persisted: approval 0.37 for under 25 vs 0.57 for 25 and over, and good under-25 applicants rejected 52% vs 32% for 25 and over (supervised: 65% vs 48%).

**Limitations**
- Small data: 1,000 rows and test splits of 250. The under-25 group has about 38 test applicants per split, and the non-foreign-worker group about 10, so foreign-worker results are too noisy to interpret and are not reported above.
- The "female" group is derived from the dataset's personal-status field, which mixes sex and marital status; it only covers women who are divorced, separated, or married.
- Base rates differ by group, and this project does not decide which fairness criterion is the right one. A gap in approvals does not by itself show unfairness, nor that it is justified.
- The group tables are averages over overlapping test splits and carry no confidence intervals.
- The linear RL agent was trained with the sensitive columns (age, personal status, foreign worker) as features, so its group gaps cannot be read as the result of a fairness-blind policy.
- The RL agents were not hyperparameter-tuned (learning rate, epochs, exploration), so the MLP agent's result may reflect that, not a limit of RL.
- The label is credit risk from an old German dataset, not current lending outcomes, and the outcomes of rejected applicants are never observed (selection bias).
- Using age, sex or nationality in real credit decisions is legally restricted in many places. This project is an audit exercise, not a deployable model.
