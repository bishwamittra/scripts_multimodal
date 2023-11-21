import copy
import csv
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from classification.estimators.rescale_estimator import RescaleEstimator
from sklearn.metrics import auc, roc_auc_score, roc_curve

import sambaflow.samba as samba


class TCGAEstimator(RescaleEstimator):
    def _assert_model_config_ok(self):
        assert self.in_height == self.in_width, "Only square images are supported while training, predicting"
        super()._assert_model_config_ok()

    def compute_patient_metrics(self, accuracy_metrics: Dict[str, Any]) -> float:
        patient_predictions = {}
        default = {"predictions": [], "target": None}

        for acc_metric in accuracy_metrics:
            prediction = acc_metric["prediction"]
            target = acc_metric["target"]

            if self.device == "RDU":
                prediction = samba.to_torch(prediction)

            prediction = torch.argmax(prediction, dim=-1)
            for ii, patient_id in enumerate(acc_metric["patient_ids"]):
                if patient_id not in patient_predictions:
                    patient_predictions[patient_id] = copy.deepcopy(default)

                patient_predictions[patient_id]["predictions"].append(prediction[ii].item())
                patient_predictions[patient_id]["target"] = target[ii].item()

        patient_scores = []
        patient_targets = []
        if self.num_classes == 2:
            for patient_id, info_dict in patient_predictions.items():
                patient_targets.append(info_dict["target"])
                patient_scores.append(sum(info_dict["predictions"]) / len(info_dict["predictions"]))

            fpr, tpr, thresholds = roc_curve(patient_targets, patient_scores)
            patient_scores = np.array(patient_scores)
            patient_targets = np.array(patient_targets)
            patient_acc = sum(patient_targets == patient_scores) / len(patient_targets)
            return auc(fpr, tpr), patient_acc
        else:
            patient_scores = np.zeros((len(patient_predictions), self.num_classes))
            for ii, (patient_id, info_dict) in enumerate(patient_predictions.items()):
                patient_targets.append(info_dict["target"])

                # Calculate patient-level class probability distribution
                for class_num in range(self.num_classes):
                    patient_scores[ii, class_num] = sum(np.array(info_dict["predictions"]) == class_num)
                patient_scores[ii] = patient_scores[ii] / sum(patient_scores[ii])

            patient_acc = sum(np.array(patient_targets) == np.argmax(patient_scores, axis=-1)) / len(patient_targets)

            # Compute one vs. rest auroc
            return roc_auc_score(patient_targets, np.array(patient_scores), average='macro',
                                 multi_class='ovr'), patient_acc

    def get_fieldnames(self):
        return ["image_path", "patient_id", "prediction"]

    def save_predictions_info(self, csv_filename: str, fieldnames: List[str], directory: Path, metadata,
                              prediction: List[int]) -> None:
        with open(csv_filename, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            for idx in range(self.batch_size):
                image_path = metadata['image_path'][idx]
                patient_id = metadata['patient_id'][idx]
                pred = prediction[idx]
                row = {'image_path': image_path, 'patient_id': patient_id, 'prediction': pred}
                writer.writerow(row)
