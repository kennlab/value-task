import ast

import pandas as pd 
import numpy as np

from analyze.beh_stats.prospect import CPTAnalysis
df=pd.read_csv('data.csv')

pA = []
pB = []
choice = []
pair_exposure_counts = {}
pair_exposure = []

for _, row in df.iterrows():
    A, B = ast.literal_eval(row['distribution_options'])
    probs = ast.literal_eval(row['distribution_probabilities'])
    pair_exposure.append(pair_exposure_counts.setdefault((A,B), 0))
    pair_exposure_counts[A,B] += 1
    pA.append(probs[A])
    pB.append(probs[B])
    choice.append(row['chosen_distribution']==A)
# pair_exposure = 
pA, pB, choice = np.array(pA), np.array(pB), np.array(choice)
pA = pA / pA.sum(axis=1, keepdims=True)
pB = pB / pB.sum(axis=1, keepdims=True)
pair_exposure = np.array(pair_exposure)
trial_number = np.arange(choice.size)
pair_ids = df['distribution_options_sorted'].tolist()
folds, predictions, summary = pair_heldout_cv(pA, pB, choice, pair_ids)


digitized = np.digitize(trial_number, np.linspace(0, choice.size, 10))
bins = np.unique(digitized)
cpt = {}
for bin_ in bins:
    mask = digitized==bin_
    cpt[bin_] = c = CPTAnalysis(pA[mask], pB[mask], choice[mask], gamma_bounds=(0.05, 10))
    c.fit('weighting')
    print(bin_, c.results['weighting']['gamma'])


for x, c in cpt.items():
    print(x, c.results['weighting']['gamma'])




exposure_bin = pd.cut(
    pair_exposure,
    bins=[
        -0.5,
        50,
        100,
        np.inf
    ],
    labels=[
        "first 50",
        "50-100",
        "100+"
    ]
)
results = []

for bin_ in exposure_bin.categories:

    mask = np.asarray(
        exposure_bin == bin_
    )

    if mask.sum() == 0:
        continue

    c = CPTAnalysis(
        pA[mask],
        pB[mask],
        choice[mask],
        gamma_bounds=(0.05, 10)
    )

    fit = c.fit(
        "weighting",
        n_starts=20
    )

    results.append({
        "exposure": str(bin_),
        "N": mask.sum(),
        "gamma": fit["gamma"],
        "beta": fit["beta"],
        "LL": fit["log_likelihood"],
        "BIC": fit["BIC"],
        "success": fit["success"]
    })

results = pd.DataFrame(results)

print(results)