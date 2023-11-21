from typing import Any, Dict

import numpy as np
import torch
from classification.estimators.rescale_estimator import RescaleEstimator
from sklearn.metrics import roc_auc_score


class MAMMOEstimator(RescaleEstimator):
    def compute_auc(self, accuracy_metrics: Dict[str, Any]) -> float:
        targets = []
        predictions = []
        for acc_metric in accuracy_metrics:
            prediction = acc_metric["prediction"]
            target = acc_metric["target"]
            targets.append(target.numpy())
            predictions.append(torch.nn.functional.softmax(prediction.float()).detach().numpy()[:, 1])
        targets = np.concatenate(targets, axis=0)
        predictions = np.concatenate(predictions, axis=0)
        return roc_auc_score(targets, predictions)

    def aggregate_test_metrics(self, accuracy_metrics: Dict[str, Any]) -> Dict[str, Any]:
        val_metrics = super().aggregate_test_metrics(accuracy_metrics)
        val_metrics["auc"] = self.compute_auc(accuracy_metrics)
        return val_metrics
