import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from scipy.stats import chi2
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss


def compare_binomial_models(
    data,
    formulas,
    outcome="chosen_A",
    group="sessionid",
    n_splits=5,
    robust=True,
    run_cv=True,
):
    """
    Compare candidate binomial/logistic regression models.

    Parameters
    ----------
    data : pandas.DataFrame
        Data containing outcome, predictors, and grouping variable.

    formulas : dict
        Dictionary mapping model names to patsy/statsmodels formulas.
        Example:
            {
                "Null": "chosen_A ~ 1",
                "EV": "chosen_A ~ dEV",
                "EV + learning": "chosen_A ~ dEV * log_option_exposure",
                "Distribution": "chosen_A ~ dp1 + dp2 + dp3 + dp4"
            }

    outcome : str
        Binary outcome column.

    group : str
        Cluster / CV grouping variable. For your data this should be
        sessionid.

    n_splits : int
        Number of grouped CV folds.

    robust : bool
        Whether to provide cluster-robust standard errors.

    run_cv : bool
        Whether to perform grouped cross-validation.

    Returns
    -------
    results : pandas.DataFrame
        Model comparison table.

    fits : dict
        Dictionary containing ordinary and robust fitted models.
    """

    data = data.copy()

    # ------------------------------------------------------------
    # Basic checks
    # ------------------------------------------------------------

    assert not data[outcome].isna().any() and not data[group].isna().any(), f"{group} or {outcome} contains missing values."

    unique_y = set(data[outcome].dropna().unique())
    if not unique_y.issubset({0, 1}):
        raise ValueError(
            f"{outcome} must contain only 0/1 values. "
            f"Found: {unique_y}"
        )

    # ------------------------------------------------------------
    # Fit models
    # ------------------------------------------------------------

    ordinary_fits = {}
    robust_fits = {}

    rows = []

    for name, formula in formulas.items():

        model = smf.glm(
            formula=formula,
            data=data,
            family=sm.families.Binomial()
        )

        # Ordinary fit:
        # used for likelihood, AIC, BIC, deviance, LR tests
        fit = model.fit()

        ordinary_fits[name] = fit

        # Cluster-robust fit:
        # used for coefficient inference
        if robust:
            robust_fit = model.fit(
                cov_type="cluster",
                cov_kwds={"groups": data[group]}
            )
            robust_fits[name] = robust_fit

        rows.append({
            "Model": name,
            "Formula": formula,
            "N": int(fit.nobs),
            "Parameters": int(fit.df_model + 1),
            "LogLik": fit.llf,
            "Deviance": fit.deviance,
            "AIC": fit.aic,

            # statsmodels GLM does not always expose BIC consistently
            # across versions, so calculate it directly.
            "BIC": (
                -2 * fit.llf
                + (fit.df_model + 1) * np.log(fit.nobs)
            ),
        })

    results = pd.DataFrame(rows)

    # ------------------------------------------------------------
    # Add LR tests for nested models
    #
    # User specifies comparisons separately below.
    # Automatically comparing every pair would be dangerous because
    # non-nested models do not admit the usual LR chi-square test.
    # ------------------------------------------------------------

    results["LR_vs_previous"] = np.nan
    results["LR_df"] = np.nan
    results["LR_p"] = np.nan

    names = list(formulas.keys())

    for i in range(1, len(names)):

        smaller = ordinary_fits[names[i - 1]]
        larger = ordinary_fits[names[i]]

        # Only make the calculation available as a convenience.
        # The function cannot prove that the models are nested.
        lr = 2 * (larger.llf - smaller.llf)

        df_diff = (
            (larger.df_model + 1)
            - (smaller.df_model + 1)
        )

        p = chi2.sf(lr, df_diff)

        results.loc[
            results["Model"] == names[i],
            "LR_vs_previous"
        ] = lr

        results.loc[
            results["Model"] == names[i],
            "LR_df"
        ] = df_diff

        results.loc[
            results["Model"] == names[i],
            "LR_p"
        ] = p

    # ------------------------------------------------------------
    # Grouped cross-validation
    #
    # IMPORTANT:
    # split by session, not by individual trial.
    # ------------------------------------------------------------

    if run_cv:

        gkf = GroupKFold(n_splits=n_splits)

        cv_scores = {
            name: []
            for name in formulas
        }

        for train_idx, test_idx in gkf.split(
            data,
            data[outcome],
            groups=data[group]
        ):

            train = data.iloc[train_idx].copy()
            test = data.iloc[test_idx].copy()

            for name, formula in formulas.items():

                try:
                    fit_cv = smf.glm(
                        formula=formula,
                        data=train,
                        family=sm.families.Binomial()
                    ).fit()

                    pred = fit_cv.predict(test)

                    # Numerical protection
                    pred = np.clip(pred, 1e-8, 1 - 1e-8)

                    score = log_loss(
                        test[outcome],
                        pred
                    )

                except Exception as exc:
                    print(
                        f"CV failed for {name}: {exc}"
                    )
                    score = np.nan

                cv_scores[name].append(score)

        results["CV_logloss_mean"] = [
            np.nanmean(cv_scores[name])
            for name in results["Model"]
        ]

        results["CV_logloss_sd"] = [
            np.nanstd(cv_scores[name], ddof=1)
            for name in results["Model"]
        ]

    # ------------------------------------------------------------
    # Sort by predictive performance if available
    # ------------------------------------------------------------

    if run_cv:
        results = results.sort_values(
            "CV_logloss_mean"
        ).reset_index(drop=True)

    else:
        results = results.sort_values(
            "AIC"
        ).reset_index(drop=True)

    return results, ordinary_fits, robust_fits


formulas = {

    "Null":
        "chosen_A ~ 1",

    "EV":
        "chosen_A ~ dEV",

    "EV + option learning":
        "chosen_A ~ dEV * log_option_exposure",

    "EV + pair learning":
        "chosen_A ~ dEV * log_pair_exposure",

    "EV + both learning":
        "chosen_A ~ dEV * log_option_exposure "
        "* log_pair_exposure",

    "Full distribution":
        "chosen_A ~ dp1 + dp2 + dp3 + dp4",

    "Full distribution + option learning":
        """
        chosen_A ~
            dp1 + dp2 + dp3 + dp4
            + dp1:log_option_exposure
            + dp2:log_option_exposure
            + dp3:log_option_exposure
            + dp4:log_option_exposure
        """,
    "Full distribution + pair learning": """
        chosen_A ~
            dp1 + dp2 + dp3 + dp4
            + dp1:log_pair_exposure
            + dp2:log_pair_exposure
            + dp3:log_pair_exposure
            + dp4:log_pair_exposure
    """,
    "Full distribution + both learning": """
        chosen_A ~
            dp1 + dp2 + dp3 + dp4

            + log_option_exposure
            + log_pair_exposure

            + dp1:log_option_exposure
            + dp2:log_option_exposure
            + dp3:log_option_exposure
            + dp4:log_option_exposure

            + dp1:log_pair_exposure
            + dp2:log_pair_exposure
            + dp3:log_pair_exposure
            + dp4:log_pair_exposure
    """
}
results, ordinary_fits, robust_fits = compare_binomial_models(
    data=df,
    formulas=formulas,
    outcome="chosen_A",
    group="sessionid",
    n_splits=5,
    robust=True,
    run_cv=True,
)