"""Metric definitions and mappings for sweep analysis."""

# Comprehensive metric mapping for display labels
METRIC_MAPPING: dict[str, str] = {
    # Performance metrics
    "train_accuracy": "Training Accuracy",
    "valid_accuracy": "Validation Accuracy",
    "test_accuracy": "Test Accuracy",
    "train_auc": "Training AUC",
    "valid_auc": "Validation AUC",
    "test_auc": "Test AUC",
    "train_categorical_loglikelihood": "Training Log-Likelihood",
    "valid_categorical_loglikelihood": "Validation Log-Likelihood",
    "test_categorical_loglikelihood": "Test Log-Likelihood",
    # Basic mutual information
    "mi_E_C": "I(E;C)",
    "mi_I_C": "I(I;C)",
    "mi_V_C": "I(V;C)",
    "mi_Vout_C": "I(Vout;C)",
    # Pairwise mutual information
    "mi_E_I": "I(E;I)",
    "mi_E_V": "I(E;V)",
    "mi_I_V": "I(I;V)",
    "mi_E_Vout": "I(E;Vout)",
    "mi_I_Vout": "I(I;Vout)",
    # Conditional mutual information
    "mi_E_I_given_C": "I(E;I|C)",
    "mi_E_V_given_C": "I(E;V|C)",
    "mi_I_V_given_C": "I(I;V|C)",
    "mi_E_Vout_given_C": "I(E;Vout|C)",
    "mi_I_Vout_given_C": "I(I;Vout|C)",
    "mi_V_C_given_E": "I(V;C|E)",
    "mi_V_C_given_I": "I(V;C|I)",
    "mi_V_C_given_E,I": "I(V;C|E,I)",
}

# Metric categories for organization
METRIC_CATEGORIES: dict[str, list[str]] = {
    "performance": [
        "accuracy",
        "auc",
        "categorical_loglikelihood",
        "mse",
        "cosine_similarity",
    ],
    "information": [
        "mi_",
        "layer_mi_",
        "mutual_information",
        "conditional_mi",
        "pairwise_mi",
    ],
    "weights": [
        "weight_",
        "exc_weight",
        "inh_weight",
        "dendritic_strength",
    ],
    "ablation": [
        "ablation_",
        "contribution_",
    ],
    "noise": [
        "noise_",
    ],
}

# Color schemes for different network categories
NETWORK_CATEGORY_COLORS: dict[str, str] = {
    # Canonical EINet categories
    "dendritic_shunting": "#1f77b4",  # Blue
    "dendritic_additive": "#ff0000",  # Red
    "dendritic_normalized_additive": "#ff7f0e",  # Orange
    "flat_normalized_additive": "#ff7f0e",  # Orange
    "flat_shunting": "#1f77b4",  # Blue (dashed line distinguishes)
    "flat_additive": "#ff0000",  # Red
    "dendritic_signed": "#2ca02c",  # Green
    "flat_signed": "#2ca02c",  # Green
    "dendritic_mlp": "#7f7f7f",  # Gray
    "flat_mlp": "#7f7f7f",  # Gray
    # MLP baselines
    "point_mlp": "#9467bd",  # Purple
    "ss_mlp": "#2ca02c",  # Green
    "ss_mlp_flat": "#2ca02c",  # Green
    "total_param_mlp": "#7f7f7f",  # Gray
    "active_param_mlp": "#7f7f7f",  # Gray
    # Legacy aliases (kept for backward compatibility)
    "shunting": "#1f77b4",
    "no_shunting": "#ff0000",
    "mlp": "#9467bd",
    # Fallback
    "unknown": "#7f7f7f",  # Gray
    "default": "#2ca02c",  # Green
}

NETWORK_CATEGORY_MARKERS: dict[str, str] = {
    # Canonical categories
    "dendritic_shunting": "o",  # Circle
    "dendritic_additive": "s",  # Square
    "dendritic_normalized_additive": "X",  # Filled X
    "flat_shunting": "^",  # Triangle up
    "flat_additive": "+",  # Plus
    "flat_normalized_additive": "x",  # X
    "dendritic_signed": "D",  # Diamond
    "flat_signed": "x",  # X
    "dendritic_mlp": "*",  # Star
    "flat_mlp": "p",  # Pentagon
    "point_mlp": "D",  # Diamond
    "ss_mlp": "*",  # Star
    "ss_mlp_flat": "d",  # Thin diamond
    "total_param_mlp": ".",  # Point
    "active_param_mlp": "d",  # Thin diamond
    # Legacy aliases
    "shunting": "o",
    "no_shunting": "s",
    "mlp": "D",
    # Fallback
    "unknown": "x",  # X
    "default": "^",  # Triangle
}

NETWORK_CATEGORY_LINESTYLES: dict[str, str] = {
    # Canonical categories
    "dendritic_shunting": "-",  # Solid
    "dendritic_additive": "-",  # Solid
    "dendritic_normalized_additive": "-",  # Dash-dot
    "flat_shunting": "--",  # Dashed
    "flat_additive": "--",  # Dashed
    "flat_normalized_additive": "--",  # Dotted
    "dendritic_signed": "-",  # Solid
    "flat_signed": "--",  # Dashed
    "dendritic_mlp": "-",  # Solid
    "flat_mlp": "--",  # Dashed
    "point_mlp": ":",  # Dotted
    "ss_mlp": "-",  # Solid
    "ss_mlp_flat": "--",  # Dashed
    "total_param_mlp": ":",  # Dotted
    "active_param_mlp": "-.",  # Dash-dot
    # Fallback
    "unknown": "-",
    "default": "-",
}

# Information theory color scheme
INFO_COLORS: dict[str, str] = {
    "excitatory": "#FF6B6B",  # Red
    "inhibitory": "#4ECDC4",  # Teal
    "combined": "#45B7D1",  # Blue
    "conditional": "#8B5A96",  # Purple
    "cross": "#FFA726",  # Orange
}


def get_metric_label(metric_name: str) -> str:
    """
    Get display label for a metric.

    Args:
        metric_name: Internal metric name

    Returns:
        Human-readable display label
    """
    # Remove _mean and _std suffixes
    clean_name = metric_name.replace("_mean", "").replace("_std", "")

    # Look up in mapping
    if clean_name in METRIC_MAPPING:
        return METRIC_MAPPING[clean_name]

    # Fallback: title case with spaces
    return clean_name.replace("_", " ").title()


def categorize_metric(metric_name: str) -> str:
    """
    Categorize a metric by type.

    Args:
        metric_name: Metric name to categorize

    Returns:
        Category name ('performance', 'information', 'weights', 'ablation', 'noise')
    """
    # Check each category's patterns (first match wins)
    for category, patterns in METRIC_CATEGORIES.items():
        if any(pattern in metric_name for pattern in patterns):
            return category

    return "other"
