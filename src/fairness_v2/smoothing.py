from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import beta


def safe_divide(num, den):
    n = pd.to_numeric(num, errors="coerce")
    d = pd.to_numeric(den, errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        out = n / d
    if isinstance(out, pd.Series):
        return out.replace([np.inf, -np.inf], np.nan)
    return np.where(np.isfinite(out), out, np.nan)


def percentile_0_100(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(np.nan, index=s.index, dtype="float64")
    return s.rank(pct=True, method="average") * 100


def empirical_bayes_rate(successes, trials, benchmark_rate, alpha: float = 20):
    successes = pd.to_numeric(successes, errors="coerce").fillna(0)
    trials = pd.to_numeric(trials, errors="coerce").fillna(0)
    benchmark = pd.to_numeric(benchmark_rate, errors="coerce")
    if not isinstance(benchmark, pd.Series):
        benchmark = pd.Series(float(benchmark), index=successes.index)
    benchmark = benchmark.fillna(benchmark.mean()).fillna(0)
    return (successes + alpha * benchmark) / (trials + alpha)


def beta_posterior_summary(successes, trials, benchmark_rate, alpha: float = 20) -> pd.DataFrame:
    successes = pd.to_numeric(successes, errors="coerce").fillna(0).clip(lower=0)
    trials = pd.to_numeric(trials, errors="coerce").fillna(0).clip(lower=0)
    failures = (trials - successes).clip(lower=0)
    benchmark = pd.to_numeric(benchmark_rate, errors="coerce")
    if not isinstance(benchmark, pd.Series):
        benchmark = pd.Series(float(benchmark), index=successes.index)
    benchmark = benchmark.clip(0.001, 0.999).fillna(benchmark.mean()).fillna(0.5)
    a0 = benchmark * alpha
    b0 = (1 - benchmark) * alpha
    a = successes + a0
    b = failures + b0
    mean = a / (a + b)
    lower = pd.Series(beta.ppf(0.025, a, b), index=successes.index)
    upper = pd.Series(beta.ppf(0.975, a, b), index=successes.index)
    p_below = pd.Series(beta.cdf(benchmark, a, b), index=successes.index)
    p_above = 1 - p_below
    return pd.DataFrame(
        {
            "posterior_mean": mean,
            "lower_credible_bound_95": lower,
            "upper_credible_bound_95": upper,
            "probability_rate_below_benchmark": p_below,
            "probability_rate_above_benchmark": p_above,
        }
    )


def band_from_score(score: pd.Series) -> pd.Series:
    return pd.Series(
        np.select(
            [score.lt(40), score.ge(40) & score.lt(65), score.ge(65) & score.lt(85), score.ge(85)],
            ["Low", "Moderate", "High", "Very High"],
            default="Unknown",
        ),
        index=score.index,
    )


def reliability_from_volume(volume: pd.Series, moderate: int, strong: int) -> pd.Series:
    v = pd.to_numeric(volume, errors="coerce").fillna(0)
    return pd.Series(
        np.select(
            [v.lt(moderate), v.ge(moderate) & v.lt(strong), v.ge(strong)],
            ["insufficient LSOA-level evidence", "exploratory", "reliable"],
            default="insufficient LSOA-level evidence",
        ),
        index=v.index,
    )
