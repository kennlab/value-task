from __future__ import annotations

import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp
from scipy.optimize import minimize
from scipy.stats import chi2


# ============================================================
# JAX configuration
# ============================================================

jax.config.update("jax_enable_x64", True)


# ============================================================
# CPT Analysis
# ============================================================

class CPTAnalysis:
    """
    JAX-accelerated CPT analysis for two-alternative lottery choice.

    Expected input:

        A_probs : shape (n_trials, 5)
        B_probs : shape (n_trials, 5)
        choices : shape (n_trials,)

    where columns correspond to probabilities of:

        [1 drop, 2 drops, 3 drops, 4 drops, 5 drops]

    choices:
        1 = chose lottery A
        0 = chose lottery B

    Models:

        "EV"
            alpha = 1
            gamma = 1
            beta free

        "utility"
            alpha free
            gamma = 1
            beta free

        "weighting"
            alpha = 1
            sigma = 1
            gamma free
            beta free

        "weighting2"
            alpha = 1
            sigma free
            gamma free
            beta free

        "CPT"
            alpha free
            sigma = 1
            gamma free
            beta free

        "CPT2"
            alpha free
            sigma free
            gamma free
            beta free

    CPT:

        v(x) = x^alpha

        One-parameter Prelec:
            w(p) = exp(-(-ln(p))^gamma)

        Two-parameter Prelec:
            w(p) = exp(-sigma * (-ln(p))^gamma)

        pi_i = w(P(X >= x_i)) - w(P(X > x_i))

        V(L) = sum_i pi_i v(x_i)

        P(A) = sigmoid(beta * (V_A - V_B))
    """

    MODELS = {
        "EV": {
            "free": [],
            "description": "Expected-value model"
        },
        "utility": {
            "free": ["alpha"],
            "description": "Power utility + linear probability weighting"
        },
        "weighting": {
            "free": ["gamma"],
            "description": "One-parameter Prelec weighting"
        },
        "weighting2": {
            "free": ["sigma", "gamma"],
            "description": "Two-parameter Prelec weighting"
        },
        "CPT": {
            "free": ["alpha", "gamma"],
            "description": "CPT with one-parameter Prelec weighting"
        },
        "CPT2": {
            "free": ["alpha", "sigma", "gamma"],
            "description": "CPT with two-parameter Prelec weighting"
        }
    }

    def __init__(
        self,
        A_probs,
        B_probs,
        choices,
        outcomes=None,
        alpha_bounds=(0.05, 3.0),
        sigma_bounds=(0.01, 10.0),
        gamma_bounds=(0.05, 10.0),
        beta_bounds=(1e-4, 50.0),
        seed=1234,
    ):
        """
        Parameters
        ----------
        A_probs : array-like, shape (n_trials, 5)
            Probabilities for lottery A.

        B_probs : array-like, shape (n_trials, 5)
            Probabilities for lottery B.

        choices : array-like, shape (n_trials,)
            1 if A chosen, 0 if B chosen.

        outcomes : array-like, optional
            Defaults to [1, 2, 3, 4, 5].

        alpha_bounds : tuple
            Bounds used for alpha transformation.

        sigma_bounds : tuple
            Bounds used for sigma transformation.

        gamma_bounds : tuple
            Bounds used for gamma transformation.

        beta_bounds : tuple
            Lower/upper bounds for beta. Beta itself is positive.

        seed : int
            Random seed for recovery analyses.
        """

        # ----------------------------------------------------
        # Convert input
        # ----------------------------------------------------

        self.A_probs = np.asarray(A_probs, dtype=np.float64)
        self.B_probs = np.asarray(B_probs, dtype=np.float64)
        self.choices = np.asarray(choices, dtype=np.float64)

        if outcomes is None:
            outcomes = [1, 2, 3, 4, 5]

        self.outcomes = np.asarray(
            outcomes,
            dtype=np.float64
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        if self.A_probs.ndim != 2:
            raise ValueError("A_probs must be 2-dimensional.")

        if self.B_probs.ndim != 2:
            raise ValueError("B_probs must be 2-dimensional.")

        if self.A_probs.shape != self.B_probs.shape:
            raise ValueError(
                "A_probs and B_probs must have identical shapes."
            )

        if self.A_probs.shape[1] != len(self.outcomes):
            raise ValueError(
                "Number of probability columns must equal "
                "number of outcomes."
            )

        if len(self.choices) != self.A_probs.shape[0]:
            raise ValueError(
                "Number of choices must equal number of trials."
            )

        if not np.all(np.isin(self.choices, [0, 1])):
            raise ValueError(
                "choices must contain only 0 and 1."
            )

        # ----------------------------------------------------
        # Normalize probabilities
        # ----------------------------------------------------

        self.A_probs = self._normalize_probs(self.A_probs)
        self.B_probs = self._normalize_probs(self.B_probs)

        # ----------------------------------------------------
        # Store configuration
        # ----------------------------------------------------

        self.alpha_bounds = tuple(alpha_bounds)
        self.sigma_bounds = tuple(sigma_bounds)
        self.gamma_bounds = tuple(gamma_bounds)
        self.beta_bounds = tuple(beta_bounds)

        self.rng = np.random.default_rng(seed)

        self.n_trials = len(self.choices)

        # ----------------------------------------------------
        # JAX arrays
        # ----------------------------------------------------

        self._A_probs_jax = jnp.asarray(self.A_probs)
        self._B_probs_jax = jnp.asarray(self.B_probs)
        self._choices_jax = jnp.asarray(self.choices)
        self._outcomes_jax = jnp.asarray(self.outcomes)

        # ----------------------------------------------------
        # Fit results
        # ----------------------------------------------------

        self.results = {}

    # ========================================================
    # DATA UTILITIES
    # ========================================================

    @staticmethod
    def _normalize_probs(probs):
        """Normalize each row to sum to one."""

        probs = np.asarray(probs, dtype=np.float64)

        if np.any(probs < 0):
            raise ValueError(
                "Probabilities cannot be negative."
            )

        row_sums = probs.sum(axis=1)

        if np.any(row_sums <= 0):
            raise ValueError(
                "Every lottery must have positive total probability."
            )

        return probs / row_sums[:, None]

    # ========================================================
    # PARAMETER TRANSFORMS
    # ========================================================

    def _alpha_from_raw(self, z):
        lo, hi = self.alpha_bounds

        return lo + (hi - lo) * jax.nn.sigmoid(z)

    def _sigma_from_raw(self, z):
        lo, hi = self.sigma_bounds
        return lo + (hi - lo) * jax.nn.sigmoid(z)

    def _gamma_from_raw(self, z):
        lo, hi = self.gamma_bounds
        return lo + (hi - lo) * jax.nn.sigmoid(z)

    def _beta_from_raw(self, z):

        beta_min, beta_max = self.beta_bounds

        return beta_min + (
            beta_max - beta_min
        ) * jax.nn.sigmoid(z)

    def _raw_from_value(self, value, bounds):
        """
        Convert a bounded parameter into an unconstrained value.
        """

        lo, hi = bounds

        fraction = (
            (value - lo) /
            (hi - lo)
        )

        fraction = np.clip(
            fraction,
            1e-6,
            1.0 - 1e-6
        )

        return np.log(
            fraction / (1.0 - fraction)
        )

    # ========================================================
    # CPT COMPONENTS
    # ========================================================
    @staticmethod
    def _prelec2(p, sigma, gamma):
        """Two-parameter Prelec weighting: exp(-sigma * (-log(p))^gamma)."""
        p = jnp.asarray(p)
        eps = 1e-12
        p_safe = jnp.clip(p, eps, 1.0 - eps)
        z = -jnp.log(p_safe)
        w = jnp.exp(-sigma * (z ** gamma))
        w = jnp.where(p <= 0.0, 0.0, w)
        w = jnp.where(p >= 1.0, 1.0, w)
        return w

    def _cpt_value_batch(
        self,
        probs,
        alpha,
        gamma=1.0,
        sigma=1.0,
        weighting="prelec1"
    ):
        """Calculate CPT values for all trials in a batch."""

        probs = probs / jnp.sum(
            probs,
            axis=1,
            keepdims=True
        )

        cumulative = jnp.cumsum(
            probs[:, ::-1],
            axis=1
        )[:, ::-1]

        cumulative_above = jnp.concatenate(
            [
                cumulative[:, 1:],
                jnp.zeros(
                    (probs.shape[0], 1),
                    dtype=probs.dtype
                )
            ],
            axis=1
        )

        if weighting == "prelec1":
            w_lower = self._prelec2(cumulative, 1, gamma)
            w_upper = self._prelec2(cumulative_above, 1, gamma)
        elif weighting == "prelec2":
            w_lower = self._prelec2(cumulative, sigma, gamma)
            w_upper = self._prelec2(cumulative_above, sigma, gamma)
        else:
            raise ValueError(
                f"Unknown weighting function: {weighting}"
            )

        decision_weights = w_lower - w_upper
        subjective_values = self._outcomes_jax ** alpha

        return jnp.sum(
            decision_weights * subjective_values,
            axis=1
        )

    # ========================================================
    # MODEL PARAMETERIZATION
    # ========================================================

    def _make_nll_function(self, model_name):
        """Create a JAX NLL function for a particular model."""

        if model_name not in self.MODELS:
            raise ValueError(f"Unknown model: {model_name}")

        free_params = self.MODELS[model_name]["free"]

        def nll(raw_params):
            alpha = 1.0
            sigma = 1.0
            gamma = 1.0
            idx = 0

            for name in free_params:
                if name == "alpha":
                    alpha = self._alpha_from_raw(raw_params[idx])
                elif name == "sigma":
                    sigma = self._sigma_from_raw(raw_params[idx])
                elif name == "gamma":
                    gamma = self._gamma_from_raw(raw_params[idx])
                idx += 1

            beta = self._beta_from_raw(raw_params[idx])

            weighting = (
                "prelec2"
                if model_name in ["weighting2", "CPT2"]
                else "prelec1"
            )

            V_A = self._cpt_value_batch(
                self._A_probs_jax, alpha, gamma=gamma,
                sigma=sigma, weighting=weighting
            )
            V_B = self._cpt_value_batch(
                self._B_probs_jax, alpha, gamma=gamma,
                sigma=sigma, weighting=weighting
            )

            logits = beta * (V_A - V_B)
            log_p_A = jax.nn.log_sigmoid(logits)
            log_p_B = jax.nn.log_sigmoid(-logits)

            ll = (
                self._choices_jax * log_p_A
                + (1.0 - self._choices_jax) * log_p_B
            )

            return -jnp.sum(ll)

        return nll

    # ========================================================
    # PARAMETER DECODING
    # ========================================================

    def _decode_params(self, model_name, raw_params):
        """Convert optimizer parameters into interpretable parameters."""
        free_params = self.MODELS[model_name]["free"]
        raw_params = np.asarray(raw_params)
        idx = 0
        alpha = 1.0
        sigma = 1.0
        gamma = 1.0

        for name in free_params:
            if name == "alpha":
                alpha = float(self._alpha_from_raw(raw_params[idx]))
            elif name == "sigma":
                sigma = float(self._sigma_from_raw(raw_params[idx]))
            elif name == "gamma":
                gamma = float(self._gamma_from_raw(raw_params[idx]))
            idx += 1

        beta = float(self._beta_from_raw(raw_params[idx]))

        return {
            "alpha": alpha,
            "sigma": sigma,
            "gamma": gamma,
            "beta": beta
        }

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def _make_starting_points(self, model_name, n_starts=10):
        """Generate sensible starting points in raw parameter space."""
        free_params = self.MODELS[model_name]["free"]
        starts = []
        values = []

        for name in free_params:
            if name == "alpha":
                values.append(self._raw_from_value(0.7, self.alpha_bounds))
            elif name == "sigma":
                values.append(self._raw_from_value(1.0, self.sigma_bounds))
            elif name == "gamma":
                values.append(self._raw_from_value(0.7, self.gamma_bounds))

        values.append(
            self._raw_from_value(2.0, self.beta_bounds)
        )

        starts.append(np.asarray(values))

        for _ in range(max(0, n_starts - 1)):
            starts.append(
                self.rng.normal(
                    loc=0.0,
                    scale=2.0,
                    size=len(values)
                )
            )

        return starts

    # ========================================================
    # FIT ONE MODEL
    # ========================================================

    def fit(
        self,
        model="CPT",
        n_starts=10,
        optimizer="L-BFGS-B",
        maxiter=2000,
        tolerance=1e-8,
    ):
        """
        Fit one model.

        Returns a dictionary containing parameter estimates,
        likelihood, AIC, BIC, optimizer information, etc.
        """

        if model not in self.MODELS:
            raise ValueError(
                f"Unknown model: {model}"
            )

        # ----------------------------------------------------
        # Build JAX objective + gradient
        # ----------------------------------------------------

        nll = self._make_nll_function(model)

        nll_and_grad = jax.jit(
            jax.value_and_grad(nll)
        )

        # ----------------------------------------------------
        # Warm up / compile JAX
        # ----------------------------------------------------

        starts = self._make_starting_points(
            model,
            n_starts=n_starts
        )

        best_result = None

        # ----------------------------------------------------
        # Objective wrapper for scipy
        # ----------------------------------------------------

        def scipy_objective(raw_params):

            value, gradient = nll_and_grad(
                jnp.asarray(raw_params)
            )

            return (
                float(value),
                np.asarray(
                    gradient,
                    dtype=np.float64
                )
            )

        # ----------------------------------------------------
        # Multi-start optimization
        # ----------------------------------------------------

        for x0 in starts:

            result = minimize(
                scipy_objective,
                x0=np.asarray(
                    x0,
                    dtype=np.float64
                ),
                method=optimizer,
                jac=True,
                options={
                    "maxiter": maxiter,
                    "gtol": tolerance
                }
            )

            if (
                best_result is None
                or result.fun < best_result.fun
            ):
                best_result = result

        assert best_result is not None
        # ----------------------------------------------------
        # Decode fitted parameters
        # ----------------------------------------------------

        params = self._decode_params(
            model,
            best_result.x
        )

        log_likelihood = -float(
            best_result.fun
        )

        k = len(
            self.MODELS[model]["free"]
        ) + 1  # beta

        n = self.n_trials

        AIC = (
            2 * k
            - 2 * log_likelihood
        )

        BIC = (
            k * np.log(n)
            - 2 * log_likelihood
        )

        fit_result = {
            "model": model,
            **params,
            "log_likelihood": log_likelihood,
            "AIC": AIC,
            "BIC": BIC,
            "n_parameters": k,
            "n_trials": n,
            "success": bool(best_result.success),
            "optimizer_message": str(
                best_result.message
            ),
            "n_iterations": best_result.nit
        }

        # Keep raw optimizer result too
        fit_result["_scipy_result"] = best_result

        self.results[model] = fit_result

        return fit_result

    # ========================================================
    # FIT ALL MODELS
    # ========================================================

    def fit_all(
        self,
        models=None,
        n_starts=10,
        maxiter=2000,
        tolerance=1e-8
    ):
        """Fit selected candidate models."""
        if models is None:
            models = [
                "EV", "utility", "weighting",
                "weighting2", "CPT", "CPT2"
            ]

        output = {}
        for model in models:
            output[model] = self.fit(
                model=model,
                n_starts=n_starts,
                maxiter=maxiter,
                tolerance=tolerance
            )
        return output

    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    def compare_models(self):
        """
        Return a model-comparison DataFrame.

        Models must have been fitted first.
        """

        rows = []

        for model, result in self.results.items():

            rows.append({
                "model": model,
                "alpha": result.get("alpha", 1.0),
                "sigma": result.get("sigma", 1.0),
                "gamma": result.get("gamma", 1.0),
                "beta": result["beta"],
                "log_likelihood": result[
                    "log_likelihood"
                ],
                "AIC": result["AIC"],
                "BIC": result["BIC"],
                "n_parameters": result[
                    "n_parameters"
                ],
                "success": result["success"]
            })

        comparison = pd.DataFrame(rows)

        # Best model by each criterion
        comparison["delta_AIC"] = (
            comparison["AIC"]
            - comparison["AIC"].min()
        )

        comparison["delta_BIC"] = (
            comparison["BIC"]
            - comparison["BIC"].min()
        )

        comparison = comparison.sort_values(
            "BIC"
        ).reset_index(drop=True)

        return comparison

    # ========================================================
    # LIKELIHOOD-RATIO TEST
    # ========================================================

    def likelihood_ratio_test(
        self,
        reduced_model,
        full_model
    ):
        """
        Likelihood-ratio test for nested models.

        Example:

            likelihood_ratio_test("utility", "CPT")

        gives the test of:

            utility-only
            vs.
            full CPT

        Note:
        LRT is appropriate only when the models are nested.
        """

        if reduced_model not in self.results:
            raise ValueError(
                f"{reduced_model} has not been fitted."
            )

        if full_model not in self.results:
            raise ValueError(
                f"{full_model} has not been fitted."
            )

        ll_reduced = self.results[
            reduced_model
        ]["log_likelihood"]

        ll_full = self.results[
            full_model
        ]["log_likelihood"]

        lr_stat = 2.0 * (
            ll_full - ll_reduced
        )

        df = (
            self.results[full_model][
                "n_parameters"
            ]
            -
            self.results[reduced_model][
                "n_parameters"
            ]
        )

        p = chi2.sf(
            lr_stat,
            df
        )

        return {
            "reduced_model": reduced_model,
            "full_model": full_model,
            "LR": lr_stat,
            "df": df,
            "p_value": p
        }

    # ========================================================
    # PREDICT CHOICE PROBABILITIES
    # ========================================================

    def predict(self, model="CPT", params=None):
        """Return P(choose A) for every trial."""

        if params is None:
            if model not in self.results:
                raise ValueError(f"{model} has not been fitted.")
            result = self.results[model]
            alpha = result.get("alpha", 1.0)
            sigma = result.get("sigma", 1.0)
            gamma = result.get("gamma", 1.0)
            beta = result["beta"]
        else:
            alpha = params.get("alpha", 1.0)
            sigma = params.get("sigma", 1.0)
            gamma = params.get("gamma", 1.0)
            beta = params["beta"]

        weighting = (
            "prelec2"
            if model in ["weighting2", "CPT2"]
            else "prelec1"
        )

        VA = self._cpt_value_batch(
            self._A_probs_jax, alpha, gamma=gamma,
            sigma=sigma, weighting=weighting
        )
        VB = self._cpt_value_batch(
            self._B_probs_jax, alpha, gamma=gamma,
            sigma=sigma, weighting=weighting
        )

        return np.asarray(
            jax.nn.sigmoid(beta * (VA - VB))
        )

    # ========================================================
    # SIMULATION
    # ========================================================

    def simulate(self, model="CPT", params=None, seed=None):
        """Generate synthetic choices using this experiment's lotteries."""
        if params is None:
            if model not in self.results:
                raise ValueError(f"{model} must be fitted first.")
            fitted = self.results[model]
            params = {
                "alpha": fitted.get("alpha", 1.0),
                "sigma": fitted.get("sigma", 1.0),
                "gamma": fitted.get("gamma", 1.0),
                "beta": fitted["beta"],
                "_weighting": "prelec2" if model in ["weighting2", "CPT2"] else "prelec1"
            }
        else:
            params = dict(params)
            params.setdefault("alpha", 1.0)
            params.setdefault("sigma", 1.0)
            params.setdefault("gamma", 1.0)
            params.setdefault("_weighting", "prelec2" if model in ["weighting2", "CPT2"] else "prelec1")

        sim_choices, p_A = self._simulate_with_params(params)

        if seed is not None:
            rng = np.random.default_rng(seed)
            p_A = np.asarray(p_A)
            sim_choices = (rng.random(self.n_trials) < p_A).astype(int)

        return sim_choices, p_A

    # ========================================================
    # PARAMETER RECOVERY
    # ========================================================

    def parameter_recovery(
        self,
        model="CPT",
        n_reps=100,
        true_params=None,
        n_starts=5,
        verbose=True
    ):
        """Generate synthetic data from known parameters and refit the same model."""
        if model not in self.MODELS:
            raise ValueError(f"Unknown model: {model}")

        if true_params is None:
            true_params = {
                "alpha": 1.0,
                "sigma": 1.0,
                "gamma": 0.7,
                "beta": 2.0
            }
        else:
            true_params = dict(true_params)
            true_params.setdefault("alpha", 1.0)
            true_params.setdefault("sigma", 1.0)
            true_params.setdefault("gamma", 1.0)
            true_params.setdefault("beta", 2.0)

        true_params["_weighting"] = (
            "prelec2"
            if model in ["weighting2", "CPT2"]
            else "prelec1"
        )

        records = []

        for rep in range(n_reps):
            sim_choices, _ = self._simulate_with_params(true_params)

            sim_data = CPTAnalysis(
                self.A_probs, self.B_probs, sim_choices,
                outcomes=self.outcomes,
                alpha_bounds=self.alpha_bounds,
                sigma_bounds=self.sigma_bounds,
                gamma_bounds=self.gamma_bounds,
                beta_bounds=self.beta_bounds,
                seed=rep + 1000
            )

            fit = sim_data.fit(
                model=model,
                n_starts=n_starts
            )

            records.append({
                "rep": rep,
                "model": model,
                "alpha_true": true_params["alpha"],
                "alpha_recovered": fit["alpha"],
                "sigma_true": true_params["sigma"],
                "sigma_recovered": fit["sigma"],
                "gamma_true": true_params["gamma"],
                "gamma_recovered": fit["gamma"],
                "beta_true": true_params["beta"],
                "beta_recovered": fit["beta"],
                "log_likelihood": fit["log_likelihood"],
                "success": fit["success"]
            })

            if verbose and (rep + 1) % 10 == 0:
                print(f"Parameter recovery: {rep + 1}/{n_reps}")

        return pd.DataFrame(records)

    # ========================================================
    # INTERNAL SIMULATION
    # ========================================================

    def _simulate_with_params(self, params):
        """Simulate choices from explicit parameters."""
        alpha = params.get("alpha", 1.0)
        sigma = params.get("sigma", 1.0)
        gamma = params.get("gamma", 1.0)
        beta = params["beta"]
        weighting = params.get("_weighting", "prelec1")

        VA = self._cpt_value_batch(
            self._A_probs_jax, alpha, gamma=gamma,
            sigma=sigma, weighting=weighting
        )
        VB = self._cpt_value_batch(
            self._B_probs_jax, alpha, gamma=gamma,
            sigma=sigma, weighting=weighting
        )

        p_A = np.asarray(
            jax.nn.sigmoid(beta * (VA - VB))
        )

        choices = (
            self.rng.random(self.n_trials) < p_A
        ).astype(int)

        return choices, p_A

    # ========================================================
    # PARAMETER RECOVERY SUMMARY
    # ========================================================
    @staticmethod
    def summarize_parameter_recovery(recovery):

        output = []

        for parameter in ["alpha", "gamma", "beta"]:

            true = recovery[
                f"{parameter}_true"
            ].to_numpy(dtype=float)

            recovered = recovery[
                f"{parameter}_recovered"
            ].to_numpy(dtype=float)

            error = recovered - true

            bias = np.mean(error)

            rmse = np.sqrt(
                np.mean(error ** 2)
            )

            true_sd = np.std(true)

            if true_sd < 1e-12:
                correlation = np.nan
            else:
                correlation = np.corrcoef(
                    true,
                    recovered
                )[0, 1]

            output.append({
                "parameter": parameter,
                "true_mean": np.mean(true),
                "recovered_mean": np.mean(recovered),
                "bias": bias,
                "RMSE": rmse,
                "correlation": correlation
            })

        return pd.DataFrame(output)
    # ========================================================
    # MODEL RECOVERY
    # ========================================================

    def model_recovery(
        self,
        models=None,
        n_reps=50,
        n_starts=5,
        params_by_model=None,
        verbose=True
    ):
        """Simulate from each candidate model and test BIC model selection."""
        if models is None:
            models = [
                "EV", "utility", "weighting",
                "weighting2", "CPT", "CPT2"
            ]

        for model in models:
            if model not in self.MODELS:
                raise ValueError(f"Unknown model: {model}")

        if params_by_model is None:
            params_by_model = {
                "EV": {"alpha": 1.0, "sigma": 1.0, "gamma": 1.0, "beta": 1.0},
                "utility": {"alpha": 0.6, "sigma": 1.0, "gamma": 1.0, "beta": 1.0},
                "weighting": {"alpha": 1.0, "sigma": 1.0, "gamma": 0.7, "beta": 1.0},
                "weighting2": {"alpha": 1.0, "sigma": 1.3, "gamma": 0.7, "beta": 1.0},
                "CPT": {"alpha": 0.6, "sigma": 1.0, "gamma": 0.7, "beta": 1.0},
                "CPT2": {"alpha": 0.6, "sigma": 1.3, "gamma": 0.7, "beta": 1.0}
            }

        records = []

        for generating_model in models:
            base = dict(params_by_model[generating_model])
            base.setdefault("alpha", 1.0)
            base.setdefault("sigma", 1.0)
            base.setdefault("gamma", 1.0)
            base.setdefault("beta", 1.0)
            base["_weighting"] = (
                "prelec2" if generating_model in ["weighting2", "CPT2"] else "prelec1"
            )

            for rep in range(n_reps):
                sim_choices, _ = self._simulate_with_params(base)
                candidate_results = []

                for candidate_model in models:
                    sim_analysis = CPTAnalysis(
                        self.A_probs, self.B_probs, sim_choices,
                        outcomes=self.outcomes,
                        alpha_bounds=self.alpha_bounds,
                        sigma_bounds=self.sigma_bounds,
                        gamma_bounds=self.gamma_bounds,
                        beta_bounds=self.beta_bounds,
                        seed=10000 + rep
                    )
                    fit = sim_analysis.fit(
                        model=candidate_model,
                        n_starts=n_starts
                    )
                    candidate_results.append(fit)
                    records.append({
                        "generating_model": generating_model,
                        "rep": rep,
                        "fitted_model": candidate_model,
                        "log_likelihood": fit["log_likelihood"],
                        "AIC": fit["AIC"],
                        "BIC": fit["BIC"],
                        "alpha": fit["alpha"],
                        "sigma": fit["sigma"],
                        "gamma": fit["gamma"],
                        "beta": fit["beta"],
                        "success": fit["success"]
                    })

                best = min(candidate_results, key=lambda x: x["BIC"])

                if verbose:
                    print(
                        f"Model recovery | {generating_model} | "
                        f"rep {rep + 1}/{n_reps} | selected = {best['model']}"
                    )

        recovery = pd.DataFrame(records)
        winner_rows = []

        for (generating_model, rep), group in recovery.groupby(["generating_model", "rep"]):
            best_idx = group["BIC"].idxmin()
            selected_model = recovery.loc[best_idx, "fitted_model"]
            winner_rows.append({
                "generating_model": generating_model,
                "rep": rep,
                "selected_model": selected_model,
                "correct": selected_model == generating_model
            })

        winners = pd.DataFrame(winner_rows)
        confusion = pd.crosstab(
            winners["generating_model"],
            winners["selected_model"]
        ).reindex(index=models, columns=models, fill_value=0)

        self.model_recovery_results = {
            "fits": recovery,
            "winners": winners,
            "confusion": confusion
        }

        return recovery, confusion

    # ========================================================
    # CONVENIENCE METHOD
    # ========================================================

    def full_analysis(
        self,
        models=None,
        n_starts=10,
        recovery_reps=100,
        model_recovery_reps=50
    ):
        """Run fitting, model comparison, LRTs, parameter recovery and model recovery."""
        if models is None:
            models = [
                "EV", "utility", "weighting",
                "weighting2", "CPT", "CPT2"
            ]

        print("=" * 60)
        print("FITTING MODELS")
        print("=" * 60)

        self.fit_all(models=models, n_starts=n_starts)
        comparison = self.compare_models()

        print("\nMODEL COMPARISON")
        print(comparison)

        lrt_results = []
        nested_pairs = []
        candidate_set = set(models)
        possible_pairs = [
            ("EV", "utility"),
            ("EV", "weighting"),
            ("weighting", "weighting2"),
            ("utility", "CPT"),
            ("weighting", "CPT"),
            ("weighting2", "CPT2"),
            ("CPT", "CPT2"),
            ("EV", "CPT"),
            ("EV", "CPT2")
        ]
        for pair in possible_pairs:
            if pair[0] in candidate_set and pair[1] in candidate_set:
                nested_pairs.append(pair)

        for reduced, full in nested_pairs:
            lrt_results.append(
                self.likelihood_ratio_test(reduced, full)
            )

        lrt_table = pd.DataFrame(lrt_results)
        print("\nLIKELIHOOD RATIO TESTS")
        print(lrt_table)

        # Parameter recovery for the most flexible requested model by default.
        recovery_model = "CPT2" if "CPT2" in models else ("CPT" if "CPT" in models else models[-1])

        print("\n" + "=" * 60)
        print("PARAMETER RECOVERY")
        print("=" * 60)

        parameter_recovery = self.parameter_recovery(
            model=recovery_model,
            n_reps=recovery_reps,
            n_starts=max(3, n_starts // 2)
        )
        parameter_summary = self.summarize_parameter_recovery(
            parameter_recovery
        )
        print(parameter_summary)

        print("\n" + "=" * 60)
        print("MODEL RECOVERY")
        print("=" * 60)

        model_recovery, confusion = self.model_recovery(
            models=models,
            n_reps=model_recovery_reps,
            n_starts=max(3, n_starts // 2)
        )
        print("\nConfusion matrix:")
        print(confusion)

        return {
            "model_comparison": comparison,
            "lrt": lrt_table,
            "parameter_recovery": parameter_recovery,
            "parameter_recovery_summary": parameter_summary,
            "model_recovery": model_recovery,
            "model_recovery_confusion": confusion
        }

