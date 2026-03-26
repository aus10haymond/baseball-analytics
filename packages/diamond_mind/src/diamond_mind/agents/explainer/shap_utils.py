"""
SHAP utility functions for the Explainer Agent.

Provides helpers for creating SHAP explainers, computing feature attributions,
and generating counterfactual scenarios for XGBoost / tree-based models.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False


def shap_available() -> bool:
    """Return True if the shap library is installed."""
    return _SHAP_AVAILABLE


def create_tree_explainer(model: Any) -> Any:
    """
    Create a SHAP TreeExplainer for the given tree-based model.

    Args:
        model: Trained tree-based model (e.g. XGBoost, LightGBM, sklearn).

    Returns:
        shap.TreeExplainer instance.

    Raises:
        ImportError: If shap is not installed.
        ValueError: If model is None.
    """
    if not _SHAP_AVAILABLE:
        raise ImportError(
            "shap is not installed. Run: pip install shap"
        )
    if model is None:
        raise ValueError("model must not be None")
    return shap.TreeExplainer(model)


def compute_shap_values(
    explainer: Any,
    X: np.ndarray,
    feature_names: List[str],
) -> List[Dict[str, float]]:
    """
    Compute per-instance SHAP values for each row in X.

    Args:
        explainer: shap.TreeExplainer instance.
        X: 2-D feature array of shape (n_samples, n_features).
        feature_names: List of feature names matching the columns of X.

    Returns:
        List of dicts mapping feature name → SHAP value, one dict per row.
    """
    raw = explainer.shap_values(X)

    # Multi-output models return a list of arrays (one per class/output).
    # Use the last element (positive class for binary classifiers).
    if isinstance(raw, list):
        values = raw[-1]
    else:
        values = raw

    values = np.asarray(values)
    if values.ndim == 1:
        values = values.reshape(1, -1)

    return [
        {name: float(sv) for name, sv in zip(feature_names, row)}
        for row in values
    ]


def get_top_features(
    shap_dict: Dict[str, float],
    n: int = 5,
) -> List[Dict[str, Any]]:
    """
    Return the top-N features by absolute SHAP value.

    Args:
        shap_dict: Mapping of feature name → SHAP value for a single instance.
        n: Number of top features to return.

    Returns:
        List of dicts with keys 'feature', 'shap_value', 'direction'.
        direction is 'positive' if shap_value > 0, else 'negative'.
    """
    sorted_features = sorted(
        shap_dict.items(),
        key=lambda kv: abs(kv[1]),
        reverse=True,
    )[:n]

    return [
        {
            "feature": name,
            "shap_value": round(value, 6),
            "direction": "positive" if value >= 0 else "negative",
        }
        for name, value in sorted_features
    ]


def generate_counterfactuals(
    model: Any,
    instance: np.ndarray,
    feature_names: List[str],
    predicted_class: int,
    perturbation_steps: Tuple[float, ...] = (-0.2, -0.1, 0.1, 0.2),
    max_counterfactuals: int = 3,
) -> List[str]:
    """
    Generate "what-if" counterfactual descriptions.

    For each feature, perturb its value and check whether the model's
    top predicted class changes.  Returns human-readable descriptions of
    the minimal changes that flip the prediction.

    Args:
        model: Trained model with a ``predict`` or ``predict_proba`` method.
        instance: 1-D feature array for a single instance.
        feature_names: Feature names matching the instance columns.
        predicted_class: Index of the currently predicted class.
        perturbation_steps: Relative perturbation magnitudes to try.
        max_counterfactuals: Maximum number of counterfactuals to return.

    Returns:
        List of human-readable counterfactual strings.
    """
    counterfactuals: List[str] = []
    base = instance.copy().reshape(1, -1)

    for i, fname in enumerate(feature_names):
        if len(counterfactuals) >= max_counterfactuals:
            break

        original_value = base[0, i]

        for step in perturbation_steps:
            perturbed = base.copy()
            delta = original_value * step
            perturbed[0, i] = original_value + delta

            try:
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(perturbed)[0]
                    new_class = int(np.argmax(probs))
                else:
                    new_class = int(model.predict(perturbed)[0])
            except Exception:
                continue

            if new_class != predicted_class:
                direction = "increase" if step > 0 else "decrease"
                pct = abs(int(step * 100))
                counterfactuals.append(
                    f"If {fname} were {direction}d by {pct}%, "
                    f"the prediction would change."
                )
                break  # One counterfactual per feature is enough

    return counterfactuals


def features_to_array(
    features: Dict[str, float],
    feature_names: List[str],
) -> np.ndarray:
    """
    Convert a feature dict to a 1-D numpy array ordered by feature_names.

    Args:
        features: Mapping of feature name → value.
        feature_names: Ordered list of feature names.

    Returns:
        1-D float array of shape (len(feature_names),).

    Raises:
        KeyError: If any feature_name is missing from the dict.
    """
    return np.array([features[name] for name in feature_names], dtype=float)
