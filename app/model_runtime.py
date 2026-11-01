"""Small, dependency-light model pipeline used by the offline executable."""
from __future__ import annotations
import re
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures, StandardScaler


def feature_map(x, kind):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != 5 or not np.isfinite(x).all() or np.any(x[:, :4] <= 0):
        raise ValueError("Expected five finite dimensionless inputs with positive capillary and time-scale numbers")
    log = np.log10(x[:, :4])
    n = x[:, 4]
    if kind == "log":
        return np.column_stack((log, n))
    if kind == "physics":
        thinning = (1.0 - n) * log[:, 3]
        log_effective_ca = log[:, 2] - np.logaddexp(0.0, thinning * np.log(10.0)) / np.log(10.0)
        return np.column_stack((log, n, log[:, 0] - log[:, 1], log[:, 2] - log[:, 1], thinning, log_effective_ca))
    raise ValueError(f"Unknown feature map: {kind}")


def log_target(y):
    return np.log(np.asarray(y, dtype=float))


def exp_target(y):
    return np.exp(np.asarray(y, dtype=float))


def parse_model_name(name):
    m = re.fullmatch(r"(log|physics)_d([1-3])_([0-9]+)-([0-9]+)_(tanh|relu)_a([0-9.]+)_log([01])", name)
    if not m:
        raise ValueError(f"Unsupported model configuration: {name}")
    kind, degree, first, second, activation, alpha, log_y = m.groups()
    return kind, int(degree), (int(first), int(second)), activation, float(alpha), bool(int(log_y))


def build_member(name, *, max_iter=5000, max_fun=80000):
    kind, degree, layers, activation, alpha, log_y = parse_model_name(name)
    network = MLPRegressor(hidden_layer_sizes=layers, activation=activation,
                           alpha=alpha, solver="lbfgs", max_iter=max_iter,
                           max_fun=max_fun, tol=1e-6, random_state=42)
    regression = Pipeline([
        ("features", FunctionTransformer(feature_map, kw_args={"kind": kind})),
        ("polynomial", PolynomialFeatures(degree, include_bias=False)),
        ("scale", StandardScaler()),
        ("network", network),
    ])
    target = (Pipeline([
        ("log", FunctionTransformer(log_target, inverse_func=exp_target)),
        ("scale", StandardScaler()),
    ]) if log_y else StandardScaler())
    return TransformedTargetRegressor(regressor=regression, transformer=target)


def build_ensemble(names):
    if not names:
        raise ValueError("At least one model member is required")
    return VotingRegressor([(f"member_{i+1}", build_member(name)) for i, name in enumerate(names)])
