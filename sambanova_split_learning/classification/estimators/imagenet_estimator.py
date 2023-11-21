from typing import Any, Dict

import torch
from classification.estimators.rescale_estimator import RescaleEstimator

import sambaflow.samba as samba


class IMAGENETEstimator(RescaleEstimator):
    def _assert_model_config_ok(self):
        assert self.in_height == self.in_width, "Only square images are supported while training, predicting"
        super()._assert_model_config_ok()

    def compute_log_metrics(self, mode: str, prediction: torch.Tensor, label: torch.Tensor,
                            loss: torch.Tensor) -> Dict[str, Any]:
        if self.device == "RDU":
            prediction = samba.to_torch(prediction)
            label = samba.to_torch(label)
        log_metrics = {}
        log_metrics["loss"] = loss.item()
        batch_size = label.size(0)
        _, prediction = prediction.topk(5, 1, True, True)
        prediction = prediction.t()
        correct = prediction.eq(label.view(1, -1).expand_as(prediction))
        log_metrics["accuracy"] = (correct[:1].reshape(-1).float().sum(0, keepdim=True) / batch_size).item()
        log_metrics["top_5_accuracy"] = (correct[:5].reshape(-1).float().sum(0, keepdim=True) / batch_size).item()
        return log_metrics

    def aggregate_test_metrics(self, accuracy_metrics: Dict[str, Any]) -> Dict[str, Any]:
        loss_sum = 0
        acc_sum = 0
        top_5_acc_sum = 0

        for acc_metric in accuracy_metrics:
            # Aggregate across different steps
            loss_sum += acc_metric.get("loss", 0)
            acc_sum += acc_metric.get("accuracy", 0)
            top_5_acc_sum += acc_metric.get("top_5_accuracy", 0)

        val_metrics = {
            "loss": loss_sum / len(accuracy_metrics),
            "accuracy": acc_sum / len(accuracy_metrics),
            "top_5_accuracy": top_5_acc_sum / len(accuracy_metrics)
        }

        return val_metrics
