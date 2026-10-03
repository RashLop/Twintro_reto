import math

import numpy as np


def ranking_metrics(labels, predictions, source_ids, k=10):
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
    source_ids = np.asarray(source_ids)
    if not (len(labels) == len(predictions) == len(source_ids)):
        raise ValueError("labels, predictions y source_ids deben tener igual longitud.")
    if k < 1:
        raise ValueError("k debe ser mayor que 0.")

    ndcg_values = []
    pairwise_correct = 0
    pairwise_total = 0
    for source_id in np.unique(source_ids):
        indices = np.flatnonzero(source_ids == source_id)
        if len(indices) < 2:
            continue

        group_labels = labels[indices]
        group_predictions = predictions[indices]
        order = np.argsort(group_predictions)[::-1][:k]
        ideal_order = np.argsort(group_labels)[::-1][:k]
        discounts = 1.0 / np.log2(np.arange(2, len(order) + 2))
        gains = np.power(2.0, group_labels[order]) - 1.0
        ideal_gains = np.power(2.0, group_labels[ideal_order]) - 1.0
        ideal_dcg = float(np.sum(ideal_gains * discounts[:len(ideal_gains)]))
        if ideal_dcg > 0:
            ndcg_values.append(float(np.sum(gains * discounts) / ideal_dcg))

        for left in range(len(indices)):
            for right in range(left + 1, len(indices)):
                actual_difference = group_labels[left] - group_labels[right]
                predicted_difference = group_predictions[left] - group_predictions[right]
                if actual_difference == 0:
                    continue
                pairwise_total += 1
                if actual_difference * predicted_difference > 0:
                    pairwise_correct += 1

    return {
        f"ndcg_at_{k}": float(np.mean(ndcg_values)) if ndcg_values else 0.0,
        "pairwise_accuracy": pairwise_correct / pairwise_total if pairwise_total else 0.0,
        "evaluated_sources": len(ndcg_values),
    }