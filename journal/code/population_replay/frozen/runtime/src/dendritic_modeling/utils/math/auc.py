from typing import Optional

import torch


def _binary_roc_auc_pytorch(
    y_true: torch.Tensor, y_score: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """
    Compute binary ROC AUC using PyTorch operations.

    This implements the Mann-Whitney U statistic approach:
    AUC = P(score_positive > score_negative)

    Args:
        y_true: Binary labels (0 or 1)
        y_score: Prediction scores
        dim: Dimension to compute AUC over

    Returns:
        AUC scores
    """
    # Handle the case where y_score has extra class dimension
    if y_score.dim() == y_true.dim() + 1:
        if y_score.shape[-1] == 2:
            y_score = y_score[..., 1]  # Use positive class
        elif y_score.shape[-1] == 1:
            y_score = y_score.squeeze(-1)
        else:
            raise ValueError(
                f"Unexpected number of classes in y_score: {y_score.shape[-1]}"
            )

    # Move dim to last position for easier processing
    if dim != -1:
        y_true = y_true.moveaxis(dim, -1)
        y_score = y_score.moveaxis(dim, -1)

    # Get the shape for reshaping later
    original_shape = y_true.shape[:-1]

    # Handle 1D case (no batch dimension)
    if y_true.dim() == 1:
        # Direct computation for 1D case
        pos_mask = y_true == 1
        neg_mask = y_true == 0

        n_pos = pos_mask.sum().float()
        n_neg = neg_mask.sum().float()

        if n_pos == 0 or n_neg == 0:
            return torch.tensor(0.5, device=y_true.device, dtype=torch.float32)

        pos_scores = y_score[pos_mask]
        neg_scores = y_score[neg_mask]

        # Compute AUC using vectorized comparison
        comparisons = pos_scores.unsqueeze(1) > neg_scores.unsqueeze(0)
        ties = pos_scores.unsqueeze(1) == neg_scores.unsqueeze(0)

        n_greater = comparisons.float().sum()
        n_ties = ties.float().sum()

        auc = (n_greater + 0.5 * n_ties) / (n_pos * n_neg)
        return auc

    # Handle multi-dimensional case
    # Flatten all dimensions except the last one
    y_true_flat = y_true.reshape(-1, y_true.shape[-1])
    y_score_flat = y_score.reshape(-1, y_score.shape[-1])

    batch_size = y_true_flat.shape[0]
    aucs = torch.zeros(batch_size, device=y_true.device, dtype=torch.float32)

    for i in range(batch_size):
        labels_i = y_true_flat[i]
        scores_i = y_score_flat[i]

        # Get positive and negative indices
        pos_mask = labels_i == 1
        neg_mask = labels_i == 0

        n_pos = pos_mask.sum().float()
        n_neg = neg_mask.sum().float()

        # Handle edge cases
        if n_pos == 0 or n_neg == 0:
            aucs[i] = 0.5  # Undefined case, return 0.5
            continue

        pos_scores = scores_i[pos_mask]
        neg_scores = scores_i[neg_mask]

        # Compute AUC using vectorized comparison
        comparisons = pos_scores.unsqueeze(1) > neg_scores.unsqueeze(0)
        ties = pos_scores.unsqueeze(1) == neg_scores.unsqueeze(0)

        n_greater = comparisons.float().sum()
        n_ties = ties.float().sum()

        auc_i = (n_greater + 0.5 * n_ties) / (n_pos * n_neg)
        aucs[i] = auc_i

    # Reshape back to original shape
    return aucs.reshape(original_shape)


def _multiclass_roc_auc_pytorch(
    y_true: torch.Tensor, y_score: torch.Tensor, dim: int = -1, average: str = "macro"
) -> torch.Tensor:
    """
    Compute multiclass ROC AUC using one-vs-rest approach.

    Args:
        y_true: Class labels
        y_score: Prediction scores for each class
        dim: Dimension to compute AUC over
        average: Averaging strategy ('macro', 'weighted', or None)

    Returns:
        AUC scores
    """
    # Don't move dimensions here - let _binary_roc_auc_pytorch handle it

    # Get unique classes
    unique_classes = torch.unique(y_true)
    n_classes = len(unique_classes)

    if n_classes < 2:
        raise ValueError("Need at least 2 classes for multiclass ROC AUC")

    # Determine shape for results
    if dim != -1:
        # Move dim to last position temporarily for shape calculation
        y_true_temp = y_true.moveaxis(dim, -1)
        y_true_temp.shape[:-1]
    else:
        pass  # original_shape already set above

    # Compute one-vs-rest AUC for each class
    class_aucs = []
    class_counts = []

    for class_idx, class_label in enumerate(unique_classes):
        # Create binary labels for this class vs rest
        binary_labels = (y_true == class_label).float()

        # Extract scores for this class
        # Handle different indexing based on unique_classes order
        if y_score.shape[-1] > 1:  # Multi-class scores provided
            if dim != -1:
                # If dim is not -1, we need to be careful about indexing
                y_score_temp = y_score.moveaxis(dim, -2)  # Move sample dim to -2
                class_scores = y_score_temp[..., class_idx]  # Extract class scores
                class_scores = class_scores.moveaxis(-1, dim)  # Move back
            else:
                class_scores = y_score[..., class_idx]
        else:  # Single score provided, use as-is
            class_scores = y_score.squeeze(-1)

        # Compute binary AUC for this class
        auc_class = _binary_roc_auc_pytorch(binary_labels, class_scores, dim=dim)
        class_aucs.append(auc_class)

        # Count samples for this class (for weighted average)
        class_count = binary_labels.sum(dim=dim if dim != -1 else -1)
        class_counts.append(class_count)

    # Stack results
    class_aucs = torch.stack(class_aucs, dim=-1)
    class_counts = torch.stack(class_counts, dim=-1)

    # Apply averaging
    if average == "macro":
        return class_aucs.mean(dim=-1)
    elif average == "weighted":
        total_counts = class_counts.sum(dim=-1, keepdim=True)
        weights = class_counts / total_counts
        weights = torch.nan_to_num(weights, nan=0.0)  # Handle division by zero
        return (class_aucs * weights).sum(dim=-1)
    elif average is None:
        return class_aucs
    else:
        raise ValueError(f"Unknown average: {average}")


def roc_auc_score(
    y_true: torch.Tensor,
    y_score: torch.Tensor,
    dim: Optional[int] = None,
    multi_class: str = "auto",
    average: str = "macro",
) -> torch.Tensor:
    """
    PyTorch-native ROC AUC score computation.

    Supports both binary and multiclass classification, works on GPU/CPU,
    and handles multiple batch dimensions with optional reduction.

    Args:
        y_true: Ground truth labels. Shape: (..., n_samples)
        y_score: Prediction scores. For binary: (..., n_samples) or (..., n_samples, 2)
                For multiclass: (..., n_samples, n_classes)
        dim: Dimension to compute AUC over. If None, uses last dimension
        multi_class: 'auto', 'ovr' (one-vs-rest), or 'raise'
        average: For multiclass - 'macro', 'weighted', or None

    Returns:
        ROC AUC scores. Shape depends on input shapes and dim parameter.
    """
    # Handle dimension
    if dim is None:
        dim = -1

    # Determine if binary or multiclass
    unique_labels = torch.unique(y_true)
    n_unique = len(unique_labels)

    if multi_class == "auto":
        is_binary = n_unique == 2
    elif multi_class == "ovr":
        is_binary = False
    elif multi_class == "raise" and n_unique > 2:
        raise ValueError("Multi-class ROC AUC requires multi_class='ovr'")
    else:
        is_binary = n_unique == 2

    # Ensure labels are in {0, 1, ...} format for multiclass
    if not is_binary and not torch.all((y_true >= 0) & (y_true < n_unique)):
        # Remap labels to {0, 1, ..., n_classes-1}
        label_map = {}
        for idx, label in enumerate(unique_labels):
            label_map[label.item()] = idx
        y_true_mapped = torch.zeros_like(y_true)
        for old_label, new_label in label_map.items():
            y_true_mapped[y_true == old_label] = new_label
        y_true = y_true_mapped

    if is_binary:
        # Handle binary case
        if y_score.dim() == y_true.dim() + 1 and y_score.shape[-1] == 2:
            # Probabilities for both classes provided, use positive class
            y_score = y_score[..., 1]
        elif y_score.dim() != y_true.dim():
            raise ValueError(
                f"Score shape {y_score.shape} incompatible with label shape {y_true.shape}"
            )

        return _binary_roc_auc_pytorch(y_true, y_score, dim=dim)
    else:
        # Handle multiclass case
        if y_score.dim() != y_true.dim() + 1:
            raise ValueError(
                f"For multiclass, y_score should have one more dimension than y_true. "
                f"Got y_score: {y_score.shape}, y_true: {y_true.shape}"
            )

        if y_score.shape[-1] != n_unique:
            raise ValueError(
                f"Number of classes in y_score ({y_score.shape[-1]}) doesn't match "
                f"number of unique labels ({n_unique})"
            )

        return _multiclass_roc_auc_pytorch(y_true, y_score, dim=dim, average=average)
