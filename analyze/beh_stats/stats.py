import ast
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


# ============================================================
# 1. LOAD AND PREPARE
# ============================================================
PATH = "data.csv"
df = pd.read_csv(PATH)
def parse_obj(x):
    if isinstance(x, str):
        return ast.literal_eval(x)
    return x
df["options"] = df["distribution_options"].map(parse_obj)
df["probabilities"] = df["distribution_probabilities"].map(parse_obj)
df["magnitudes"] = df["distribution_magnitude_values"].map(parse_obj)

# Proper datetime, useful for ordering the data
df["datetime"] = pd.to_datetime(
    df["date"].astype(str) + " " + df["time"].astype(str)
)

df = df.sort_values(
    ["datetime", "sessionid", "trialid"]
).reset_index(drop=True)


# ============================================================
# 2. CANONICALISE A/B
#
# option1 and option2 have no intrinsic meaning.
# We therefore give each trial a deterministic A and B label
# based on option identity.
#
# chosen_A = 1 if A was selected, 0 if B was selected.
# ============================================================

A = []
B = []
chosen_A = []

for _, row in df.iterrows():

    opts = sorted(row["options"], key=str)

    a = opts[0]
    b = opts[1]

    A.append(a)
    B.append(b)

    chosen_A.append(
        int(row["chosen_distribution"] == a)
    )

df["A"] = A
df["B"] = B
df["chosen_A"] = chosen_A

# Pair identity, independent of presentation order
df["pair"] = [
    tuple(sorted([a, b], key=str))
    for a, b in zip(df["A"], df["B"])
]

df["pair"] = df["pair"].astype(str)


# ============================================================
# 3. EXTRACT THE 5 PROBABILITY MASSES
# ============================================================

dp = np.zeros((len(df), 5))
ev_A = np.zeros(len(df))
ev_B = np.zeros(len(df))
for i, (_, row) in enumerate(df.iterrows()):
    a = row["A"]
    b = row["B"]

    pA = np.asarray(row["probabilities"][a], dtype=float)
    pB = np.asarray(row["probabilities"][b], dtype=float)

    xA = np.asarray(row["magnitudes"][a], dtype=float)
    xB = np.asarray(row["magnitudes"][b], dtype=float)

    # Probability differences:
    # p(A outcome k) - p(B outcome k)
    dp[i, :] = pA - pB

    # Expected values
    ev_A[i] = np.sum(pA * xA)
    ev_B[i] = np.sum(pB * xB)

for k in range(5):
    df[f"dp{k+1}"] = dp[:, k]

df["EV_A"] = ev_A
df["EV_B"] = ev_B
df["dEV"] = df["EV_A"] - df["EV_B"]


# ============================================================
# 4. EXPERIENCE VARIABLES
#
# Since options enter/leave the experiment at different times,
# calendar date is NOT our preferred learning variable.
#
# Instead calculate previous exposure to:
#   - A
#   - B
#   - the A/B pair
# ============================================================
import numpy as np
from collections import defaultdict


# ============================================================
# GENERATE EXPOSURE VARIABLES FROM TRIAL HISTORY
#
# For a current comparison A vs B:
#
# A_exposure:
#   total previous exposures to A
#
# B_exposure:
#   total previous exposures to B
#
# pair_exposure:
#   previous exposures to this exact A-B pair
#
# A_other_exposure:
#   previous exposures to A against options OTHER than B
#
# B_other_exposure:
#   previous exposures to B against options OTHER than A
#
# other_pair_exposure:
#   mean exposure to A/B in other pairings
#
# exposure_imbalance:
#   A_other_exposure - B_other_exposure
#
# log_exposure_imbalance:
#   log(1 + A_other) - log(1 + B_other)
# ============================================================


# ------------------------------------------------------------
# Make sure df is in chronological trial order BEFORE this
# ------------------------------------------------------------

# For example, if appropriate for your data:
#
# df = df.sort_values(
#     ["datetime", "sessionid", "trialid"]
# ).reset_index(drop=True)


# ------------------------------------------------------------
# Running history
# ------------------------------------------------------------

option_counts = defaultdict(int)
pair_counts = defaultdict(int)


# ------------------------------------------------------------
# Storage
# ------------------------------------------------------------

A_exposure = []
B_exposure = []

pair_exposure = []

A_other_exposure = []
B_other_exposure = []

other_pair_exposure = []

exposure_imbalance = []
log_exposure_imbalance = []


# ============================================================
# WALK THROUGH TRIALS CHRONOLOGICALLY
# ============================================================

for a, b in zip(df["A"], df["B"]):

    # Unordered pair identity
    pair = tuple(
        sorted([a, b], key=str)
    )


    # --------------------------------------------------------
    # TOTAL PREVIOUS OPTION EXPOSURE
    # --------------------------------------------------------

    e_A = option_counts[a]
    e_B = option_counts[b]


    # --------------------------------------------------------
    # PREVIOUS EXPOSURE TO THIS EXACT PAIR
    # --------------------------------------------------------

    e_pair = pair_counts[pair]


    # --------------------------------------------------------
    # EXPOSURE TO EACH OPTION OUTSIDE THIS PAIR
    #
    # Every A-B encounter contributed once to both A and B,
    # so subtract exact-pair exposure.
    # --------------------------------------------------------

    e_A_other = e_A - e_pair
    e_B_other = e_B - e_pair


    # --------------------------------------------------------
    # MEAN GENERAL EXPERIENCE WITH THESE OPTIONS
    # OUTSIDE THE CURRENT PAIR
    # --------------------------------------------------------

    e_other = (
        e_A_other + e_B_other
    ) / 2


    # --------------------------------------------------------
    # EXPOSURE IMBALANCE
    #
    # Positive:
    #   A has been encountered more than B elsewhere
    #
    # Negative:
    #   B has been encountered more than A elsewhere
    # --------------------------------------------------------

    imbalance = (
        e_A_other - e_B_other
    )


    # --------------------------------------------------------
    # LOG EXPOSURE IMBALANCE
    #
    # Equivalent to:
    #
    # log(
    #   (1 + A_other_exposure)
    #   /
    #   (1 + B_other_exposure)
    # )
    #
    # Positive -> A more familiar
    # Negative -> B more familiar
    # --------------------------------------------------------

    log_imbalance = (
        np.log1p(e_A_other)
        - np.log1p(e_B_other)
    )


    # --------------------------------------------------------
    # STORE
    # --------------------------------------------------------

    A_exposure.append(e_A)
    B_exposure.append(e_B)

    pair_exposure.append(e_pair)

    A_other_exposure.append(e_A_other)
    B_other_exposure.append(e_B_other)

    other_pair_exposure.append(e_other)

    exposure_imbalance.append(imbalance)
    log_exposure_imbalance.append(
        log_imbalance
    )


    # --------------------------------------------------------
    # UPDATE HISTORY AFTER CURRENT TRIAL
    #
    # Important: update AFTER storing, because we want
    # PREVIOUS exposure.
    # --------------------------------------------------------

    option_counts[a] += 1
    option_counts[b] += 1

    pair_counts[pair] += 1


# ============================================================
# ADD TO DATAFRAME
# ============================================================

df["A_exposure"] = A_exposure
df["B_exposure"] = B_exposure

df["pair_exposure"] = pair_exposure

df["A_other_exposure"] = A_other_exposure
df["B_other_exposure"] = B_other_exposure

df["other_pair_exposure"] = (
    other_pair_exposure
)

df["exposure_imbalance"] = (
    exposure_imbalance
)

df["log_exposure_imbalance"] = (
    log_exposure_imbalance
)


# ============================================================
# LOG TRANSFORMS FOR THE OTHER TWO EXPOSURE MEASURES
# ============================================================

df["log_pair_exposure"] = np.log1p(
    df["pair_exposure"]
)

df["log_other_pair_exposure"] = np.log1p(
    df["other_pair_exposure"]
)


# ============================================================
# OPTIONAL: ORIGINAL MEAN OPTION EXPOSURE
#
# Useful for checking that the decomposition works.
# ============================================================

df["mean_option_exposure"] = (
    df["A_exposure"]
    + df["B_exposure"]
) / 2

df["log_option_exposure"] = np.log1p(
    df["mean_option_exposure"]
)

# ============================================================
#
# Model fitting
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2


# ============================================================
# 1. MODEL FORMULAS
# ============================================================
formulas = {

    "Distribution": """
        chosen_A ~
            dp1 + dp2 + dp3 + dp4
    """,

    "Distribution*pair + imbalance": """
        chosen_A ~
            dp1 + dp2 + dp3 + dp4
            + log_pair_exposure
            + log_exposure_imbalance

            + dp1:log_pair_exposure
            + dp2:log_pair_exposure
            + dp3:log_pair_exposure
            + dp4:log_pair_exposure
    """,

    "Full model": """
        chosen_A ~
            dp1 + dp2 + dp3 + dp4

            + log_pair_exposure
            + log_exposure_imbalance

            + dp1:log_pair_exposure
            + dp2:log_pair_exposure
            + dp3:log_pair_exposure
            + dp4:log_pair_exposure

            + dp1:log_exposure_imbalance
            + dp2:log_exposure_imbalance
            + dp3:log_exposure_imbalance
            + dp4:log_exposure_imbalance
    """
}
# ============================================================
# 2. FIT UPDATED MODELS
#
# Ordinary fits are retained for AIC/BIC/LR.
# Robust fits are retained for inference.
# ============================================================

ordinary_fits = {}
robust_fits = {}

for name, formula in formulas.items():

    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial()
    )

    ordinary_fits[name] = model.fit()

    robust_fits[name] = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": df["sessionid"]}
    )


# ============================================================
# 3. MODEL COMPARISON TABLE
# ============================================================

comparison_rows = []

for name, fit in ordinary_fits.items():

    k = int(fit.df_model + 1)

    comparison_rows.append({
        "Model": name,
        "N": int(fit.nobs),
        "Parameters": k,
        "LogLik": fit.llf,
        "Deviance": fit.deviance,
        "AIC": fit.aic,
        "BIC": -2 * fit.llf + k * np.log(fit.nobs)
    })

comparison = pd.DataFrame(comparison_rows)

print("\nMODEL COMPARISON")
print(
    comparison.sort_values("AIC")
    .to_string(index=False)
)

m = robust_fits["Full model"]

wald_imbalance = m.wald_test("""
    dp1:log_exposure_imbalance = 0,
    dp2:log_exposure_imbalance = 0,
    dp3:log_exposure_imbalance = 0,
    dp4:log_exposure_imbalance = 0
""")

print(wald_imbalance)







# ============================================================
# CROSS-VALIDATED LOG LOSS
#
# Split by session, not individual trials.
# ============================================================

from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss
def grouped_cv_logloss(
    formula,
    data,
    group_col="sessionid",
    outcome_col="chosen_A",
    n_splits=5
):

    gkf = GroupKFold(n_splits=n_splits)

    fold_scores = []

    for train_idx, test_idx in gkf.split(
        data,
        data[outcome_col],
        groups=data[group_col]
    ):

        train = data.iloc[train_idx]
        test = data.iloc[test_idx]

        fit = smf.glm(
            formula=formula,
            data=train,
            family=sm.families.Binomial()
        ).fit()

        pred = fit.predict(test)

        pred = np.clip(
            pred,
            1e-8,
            1 - 1e-8
        )

        fold_scores.append(
            log_loss(
                test[outcome_col],
                pred
            )
        )

    return np.asarray(fold_scores)


cv_scores = {}

for name, formula in formulas.items():

    scores = grouped_cv_logloss(
        formula,
        df
    )

    cv_scores[name] = scores


comparison["CV_logloss_mean"] = [
    cv_scores[name].mean()
    for name in comparison["Model"]
]

comparison["CV_logloss_sd"] = [
    cv_scores[name].std(ddof=1)
    for name in comparison["Model"]
]


print("\nMODEL COMPARISON WITH CROSS-VALIDATION")
print(
    comparison
    .sort_values("CV_logloss_mean")
    .to_string(index=False)
)


# ============================================================
# PAIRWISE CV DIFFERENCES
#
# Particularly useful for the final comparison.
# Positive = first model has higher/worse log loss.
# ============================================================

def compare_cv_models(
    scores_a,
    scores_b,
    name_a,
    name_b
):

    diff = scores_a - scores_b

    out = pd.Series({
        "Model A": name_a,
        "Model B": name_b,
        "Mean CV difference": diff.mean(),
        "SD CV difference": diff.std(ddof=1),
        "Fold differences": diff.tolist()
    })

    return out


print("\nCV DIFFERENCE: FULL MODEL vs DISTRIBUTION")
print(
    compare_cv_models(
        cv_scores["Full model"],
        cv_scores["Distribution"],
        "Full model",
        "Distribution"
    )
)

print("\nCV DIFFERENCE: FULL MODEL vs PAIR LEARNING")
print(
    compare_cv_models(
        cv_scores["Full model"],
        cv_scores["Distribution*pair + imbalance"],
        "Full model",
        "Distribution + pair learning"
    )
)



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# ACTUAL-TRIAL PREDICTIONS FOR ALL THREE MODELS
# ============================================================

model_names = [
    "Distribution",
    "Distribution*pair + imbalance",
    "Full model"
]


for name in model_names:

    m = robust_fits[name]

    # P(A chosen) using the ACTUAL predictor values
    # on each observed trial
    df[f"pred_A_{name}"] = m.predict(df)