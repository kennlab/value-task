import numpy as np
import pandas as pd

from analyze.beh_stats.prospect import CPTAnalysis


def pair_heldout_cv(
    pA,
    pB,
    choice,
    pair_ids,
    model="weighting",
    n_folds=10,
    n_starts=10,
    random_state=1234,
    outcomes=None,
    alpha_bounds=(0.05, 3.0),
    gamma_bounds=(0.05, 10.0),
    beta_bounds=(1e-4, 100.0),
):
    """
    Pair-held-out cross-validation for CPT models.

    Entire lottery pairs are assigned to either train or test
    within each fold. Thus, no pair appearing in the test set
    appears in the corresponding training set.

    Parameters
    ----------
    pA, pB : array-like, shape (n_trials, n_outcomes)
        Probability vectors for lotteries A and B.

    choice : array-like, shape (n_trials,)
        1 = chose A
        0 = chose B

    pair_ids : array-like, shape (n_trials,)
        Identifier for the lottery pair on each trial.

    model : str
        "EV", "utility", "weighting", or "CPT".

    n_folds : int
        Number of pair-level CV folds.

    n_starts : int
        Number of optimization starts per fit.

    random_state : int
        Random seed controlling fold assignment.

    outcomes : array-like or None
        Defaults to [1, 2, 3, 4, 5].

    Returns
    -------
    folds : pd.DataFrame
        One row per CV fold.

    predictions : pd.DataFrame
        One row per held-out trial.

    summary : dict
        Pooled out-of-sample metrics.
    """

    # ========================================================
    # INPUTS
    # ========================================================

    pA = np.asarray(
        pA,
        dtype=float
    )

    pB = np.asarray(
        pB,
        dtype=float
    )

    choice = np.asarray(
        choice,
        dtype=int
    )

    pair_ids = list(pair_ids)
    n_trials = len(choice)
    if outcomes is None:
        outcomes = np.arange(
            1,
            pA.shape[1] + 1,
            dtype=float
        )

    outcomes = np.asarray(
        outcomes,
        dtype=float
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    if len(pA) != n_trials:
        raise ValueError(
            "pA and choice have different numbers of trials."
        )

    if len(pB) != n_trials:
        raise ValueError(
            "pB and choice have different numbers of trials."
        )

    if len(pair_ids) != n_trials:
        raise ValueError(
            "pair_ids must contain one ID per trial."
        )

    if pA.shape != pB.shape:
        raise ValueError(
            "pA and pB must have the same shape."
        )

    if not np.all(
        np.isin(choice, [0, 1])
    ):
        raise ValueError(
            "choice must contain only 0 and 1."
        )

    # ========================================================
    # UNIQUE PAIRS
    # ========================================================

    # dict preserves order and handles tuple/string IDs
    unique_pairs = list(
        dict.fromkeys(pair_ids)
    )

    n_pairs = len(unique_pairs)

    if n_folds > n_pairs:
        raise ValueError(
            f"{n_folds} folds requested but there are "
            f"only {n_pairs} unique pairs."
        )

    print(
        f"{n_trials:,} trials | "
        f"{n_pairs} unique pairs | "
        f"{n_folds} folds"
    )

    # ========================================================
    # RANDOMIZE PAIRS
    # ========================================================

    rng = np.random.default_rng(
        random_state
    )

    shuffled_pairs = unique_pairs.copy()

    rng.shuffle(
        shuffled_pairs
    )

    # --------------------------------------------------------
    # Split PAIRS, not trials
    # --------------------------------------------------------

    fold_pairs = np.array_split(
        np.asarray(
            shuffled_pairs,
            dtype=object
        ),
        n_folds
    )

    # ========================================================
    # STORAGE
    # ========================================================

    fold_rows = []
    prediction_rows = []

    # ========================================================
    # CV
    # ========================================================

    for fold_idx, heldout_pairs in enumerate(
        fold_pairs,
        start=1
    ):

        heldout_pairs = set(
            heldout_pairs.tolist()
        )

        # ----------------------------------------------------
        # Train/test masks
        # ----------------------------------------------------

        test_mask = np.asarray([
            pair in heldout_pairs
            for pair in pair_ids
        ])

        train_mask = ~test_mask

        print(
            f"Fold {fold_idx:02d}: "
            f"{train_mask.sum():,} train | "
            f"{test_mask.sum():,} test | "
            f"{len(heldout_pairs)} held-out pairs"
        )

        # ====================================================
        # FIT TRAINING SET
        # ====================================================

        train_cpt = CPTAnalysis(
            pA[train_mask],
            pB[train_mask],
            choice[train_mask],
            outcomes=outcomes,
            alpha_bounds=alpha_bounds,
            gamma_bounds=gamma_bounds,
            beta_bounds=beta_bounds,
            seed=random_state + fold_idx
        )

        fit = train_cpt.fit(
            model=model,
            n_starts=n_starts
        )

        params = {
            "alpha": fit["alpha"],
            "gamma": fit["gamma"],
            "beta": fit["beta"]
        }

        # ====================================================
        # TEST SET
        # ====================================================

        test_cpt = CPTAnalysis(
            pA[test_mask],
            pB[test_mask],
            choice[test_mask],
            outcomes=outcomes,
            alpha_bounds=alpha_bounds,
            gamma_bounds=gamma_bounds,
            beta_bounds=beta_bounds,
            seed=random_state + fold_idx
        )

        # IMPORTANT:
        # predict using TRAINING parameters.
        #
        # Nothing is fitted to the test choices.
        p_A = test_cpt.predict(
            model=model,
            params=params
        )

        p_A = np.asarray(
            p_A
        )

        # Avoid log(0)
        p_A = np.clip(
            p_A,
            1e-12,
            1 - 1e-12
        )

        y = choice[test_mask]

        # ====================================================
        # TEST LOG LIKELIHOOD
        # ====================================================

        trial_ll = (
            y * np.log(p_A)
            +
            (1 - y) * np.log(1 - p_A)
        )

        test_LL = np.sum(
            trial_ll
        )

        test_NLL = -test_LL

        test_NLL_per_trial = (
            test_NLL / len(y)
        )

        # ====================================================
        # BRIER SCORE
        # ====================================================

        brier = np.mean(
            (p_A - y) ** 2
        )

        # ====================================================
        # ACCURACY
        # ====================================================

        predicted_choice = (
            p_A >= 0.5
        ).astype(int)

        accuracy = np.mean(
            predicted_choice == y
        )

        # ====================================================
        # FOLD RESULTS
        # ====================================================

        fold_rows.append({
            "fold": fold_idx,
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
            "n_test_pairs": len(heldout_pairs),
            "alpha": fit["alpha"],
            "gamma": fit["gamma"],
            "beta": fit["beta"],
            "train_LL": fit["log_likelihood"],
            "test_LL": test_LL,
            "test_NLL_per_trial": test_NLL_per_trial,
            "Brier": brier,
            "accuracy": accuracy,
            "success": fit["success"]
        })

        # ====================================================
        # SAVE EVERY TEST PREDICTION
        # ====================================================
        test_indices = np.flatnonzero(
            test_mask
        )

        for (
            trial_idx,
            pair,
            observed,
            predicted_p,
            ll
        ) in zip(
            test_indices,
            np.asarray(pair_ids, dtype=object)[test_mask],
            y,
            p_A,
            trial_ll
        ):

            prediction_rows.append({
                "fold": fold_idx,
                "trial": int(trial_idx),
                "pair": pair,
                "choice": int(observed),
                "p_A": float(predicted_p),
                "log_likelihood": float(ll),
                "predicted_choice": int(predicted_p >= 0.5),
                "correct": int((predicted_p >= 0.5)== observed)
            })

    # ========================================================
    # RESULTS
    # ========================================================
    folds = pd.DataFrame(fold_rows)
    predictions = pd.DataFrame(prediction_rows)

    # ========================================================
    # POOLED METRICS
    # ========================================================

    pooled_LL = predictions[
        "log_likelihood"
    ].sum()

    pooled_NLL = -pooled_LL

    pooled_NLL_per_trial = (
        pooled_NLL
        / len(predictions)
    )

    pooled_Brier = np.mean(
        (
            predictions["p_A"]
            -
            predictions["choice"]
        ) ** 2
    )

    pooled_accuracy = np.mean(
        predictions["correct"]
    )

    summary = {

        "model":
            model,

        "n_trials":
            n_trials,

        "n_pairs":
            n_pairs,

        "n_folds":
            n_folds,

        "test_LL":
            pooled_LL,

        "test_NLL":
            pooled_NLL,

        "test_NLL_per_trial":
            pooled_NLL_per_trial,

        "Brier":
            pooled_Brier,

        "accuracy":
            pooled_accuracy,

        "all_fits_successful":
            bool(
                folds["success"].all()
            ),

        # Parameter stability across folds
        "gamma_mean":
            folds["gamma"].mean(),

        "gamma_sd":
            folds["gamma"].std(),

        "beta_mean":
            folds["beta"].mean(),

        "beta_sd":
            folds["beta"].std(),
    }

    return folds, predictions, summary