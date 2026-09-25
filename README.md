# Explaining a Loan Default Model (SHAP and Permutation Importance)

A model that says "this person will probably default" is not useful until it can also say **why**. This project opens up a loan default model and checks whether its reasons make sense.

## Explained in simplest language

Imagine a robot that decides whether to lend piggy-bank money to someone. It looks at clues about the person, like their credit score and how many bills they paid late, and then says "safe" or "risky".

Robots that just say "safe" or "risky" are scary, because we cannot tell if they are thinking the right way. So we ask the robot two kinds of questions:

1. **"What if I mess up one clue?"** (permutation importance). We scramble one clue, like mixing up one crayon color in a drawing. If the picture gets much worse, that clue was important.
2. **"How much did each clue push your answer?"** (SHAP). Think of friends sharing one pizza fairly. Each clue gets some slices of credit for the final answer, and we can see which clues pushed toward "risky" and which pushed toward "safe".

Then we test if the robot's reasons are real, by taking a clue away and seeing if the robot really gets worse.

## Problem statement

Train a loan default classifier, then explain which features drive its predictions, both overall and for single customers, and check that the explanations can be trusted.

## Dataset

Synthetic, generated inside the notebook (8,000 fake loan applicants, about 25 percent default). The file is also saved in `data/synthetic_loan_applicants.csv`.

I made the data myself on purpose. Because I planted the rules that decide who defaults, I know the answer key, so I can check if SHAP and permutation importance find the truth. The data also contains traps:

- `random_noise` and `num_credit_cards` have no effect at all.
- `monthly_income_reported` is a near copy of `annual_income` (correlation about 0.999), to test how the tools handle twin features.
- A real rule depends on loan size compared to income, but the model only sees the two columns separately.

## Approach

1. Generate the data with known rules.
2. Train two models: logistic regression (simple) and LightGBM (stronger, same family as my Day 8 stack).
3. Look at the built-in LightGBM importance, then compare it against better tools.
4. Permutation importance on the held-out test set (20 repeats per feature, scored on ROC AUC).
5. SHAP with TreeExplainer: global bar chart, beeswarm, dependence plot, and per-person waterfalls for the riskiest and safest test customers.
6. Compare all rankings against the planted answer key.
7. Twin feature test with grouped permutation importance.
8. Drop-the-feature check: remove a feature, retrain, and see if the score change matches what the explanation claimed.

## Results

Model scores on the 2,000-row test set:

| Model | ROC AUC | PR AUC | F1 at 0.5 |
|---|---|---|---|
| Logistic Regression | 0.796 | 0.612 | 0.507 |
| LightGBM | 0.793 | 0.592 | 0.520 |

The two models score about the same, which makes sense because the planted rules are simple add-up rules.

Explanation results (all on the test set):

- **Top drivers found by both SHAP and permutation importance:** credit score, loan amount, debt to income, number of late payments. This matches what was planted.
- **Permutation importance for credit score:** shuffling it drops ROC AUC by about 0.114. Shuffling `random_noise` drops it by about zero (slightly negative, within noise).
- **SHAP honesty check:** base value plus all feature pushes rebuilds the model's own raw score to within about 1e-14.
- **Built-in LightGBM importance was the least reliable.** It ranked `random_noise` 5th and `has_cosigner` 10th, while permutation and SHAP put the cosigner 5th and 6th and the noise near the bottom.
- **Twin features fool single-feature permutation.** Shuffling `annual_income` alone drops AUC by about 0.014 and the near-copy alone by about 0.006, but shuffling them together drops it by about 0.034.
- **SHAP gives real credit to the useless twin.** `monthly_income_reported` has no effect in the answer key, yet gets about two thirds of the SHAP importance of `annual_income`, because the model uses it as a stand-in.
- **Drop-the-feature check:** removing `credit_score` and retraining lowers ROC AUC by about 0.038. Removing `num_credit_cards`, `age`, or `random_noise` changes it by 0.004 to 0.007, so the explanations pointed at the right things.
- **Overfitting spotted by a waterfall:** the riskiest customer gets a small push from `random_noise`, which means the model memorized a little luck from the training data.

Charts are in `images/`.

## What this means for trust and deployment

- If the top reasons are late payments, credit score, and debt, a loan officer can agree with the model. If a random column or an ID showed up at the top, that would be a red flag even with a high score.
- Per-customer waterfalls give a concrete reason for a single decision, which matters when a customer has a right to know why they were turned down.
- Saving the importance ranking at release time gives a baseline. If it changes a lot in production, the data or the world has changed.

## Honest limits

- The data is synthetic. Real data is messier and real explanations are less clean.
- SHAP and permutation importance describe what the **model uses**, not what **causes** default in real life.
- Correlated features hide or share importance, as the twin test shows.
- One train and test split only. Rankings for the middle features could change with a different split.

## What I would improve next

1. Repeat the analysis on a real public loan dataset and compare the story.
2. Run several random splits and report how stable the rankings are.
3. Add SHAP interaction values to see which features work together, for example loan size and income.
4. Add a fairness check comparing error rates across age groups.
5. Return the top three SHAP reasons from the FastAPI `/predict` endpoint (Day 12).

## How to run

```bash
pip install -r requirements.txt
jupyter notebook model_explainability_shap_permutation.ipynb
```

Everything is seeded (`RANDOM_STATE = 42`), so the numbers above should reproduce.

## Repository layout

```
model-explainability/
  README.md
  requirements.txt
  model_explainability_shap_permutation.ipynb
  data/
    synthetic_loan_applicants.csv
    importance_rank_comparison.csv
  images/
    01_builtin_gain_importance.png
    02_permutation_importance.png
    03_shap_global_bar.png
    04_shap_beeswarm.png
    05_shap_dependence_credit_score.png
    06_waterfall_riskiest.png
    07_waterfall_safest.png
```
