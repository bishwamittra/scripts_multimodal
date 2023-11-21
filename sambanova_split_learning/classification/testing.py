import sambaflow.samba as samba
from samba_tools.testing.tester import TesterCallback
from sambaflow.samba.utils.benchmark_acc import AccuracyReport


class AccuracyCheck(TesterCallback):
    def __init__(self, cfg):

        super().__init__(cfg)

        if self.cfg.acc_report_json is not None:
            self.acc_report_json = str(self.cfg.acc_report_json)
        else:
            self.acc_report_json = None

        self.acc_test = self.cfg.acc_test
        if self.acc_report_json is not None and not self.acc_test:
            raise ValueError("perf mode 'Accuracy' requires --acc-test")

    def on_train_end(self, estimator, training_result):
        train_metrics, val_metrics = training_result

        if self.acc_report_json is not None and estimator.rank == 0:
            acc_metric_lookup = {'loss': 'loss', 'accuracy': 'acc'}
            train_report_metrics = {acc_metric_lookup[k]: train_metrics[k] for k in acc_metric_lookup}
            if not self.cfg.run_benchmark:
                val_report_metrics = {acc_metric_lookup[k]: val_metrics[k] for k in acc_metric_lookup}
            else:
                val_report_metrics = {}
            AccuracyReport(train_metrics=train_report_metrics,
                           val_metrics=val_report_metrics,
                           batch_size=estimator.batch_size,
                           num_iterations=estimator.epochs).save(self.acc_report_json)

        if not self.cfg.run_benchmark:
            msg = (f"Validation loss {val_metrics['loss']} after {estimator.epochs} epochs "
                   f"is above maximum bound {self.cfg.loss_thresh}")
            assert val_metrics['loss'] < self.cfg.loss_thresh, msg
            msg = (f"Validation acc {val_metrics['accuracy']} after {estimator.epochs} epochs "
                   f"is below minimum bound {self.cfg.acc_thresh}")
            assert val_metrics['accuracy'] > self.cfg.acc_thresh, msg

    def on_evaluate_end(self, estimator, evaluation_result):
        if self.cfg.run_benchmark:
            return

        val_metrics = evaluation_result
        if self.acc_report_json is not None and estimator.rank == 0:
            acc_metric_lookup = {'loss': 'loss', 'accuracy': 'acc'}
            val_report_metrics = {acc_metric_lookup[k]: val_metrics[k] for k in acc_metric_lookup}
            AccuracyReport(train_metrics={}, val_metrics=val_report_metrics,
                           batch_size=estimator.batch_size).save(self.acc_report_json)
        # Check for validation accuracy if running with accuracy test mode.
        val_accuracy, val_loss = val_metrics["accuracy"], val_metrics["loss"]

        msg = f"Validation accuracy {val_accuracy} lower than expected {self.cfg.acc_thresh}"
        assert val_accuracy > self.cfg.acc_thresh, msg
        msg = f"Validation loss {val_loss} higher than expected {self.cfg.loss_thresh}"
        assert val_loss < self.cfg.loss_thresh, msg


class PerfCheck(TesterCallback):
    def on_evaluate_end(self, estimator, evaluation_result):
        # Compare min throughput
        mean_throughputs = samba.session.profiler.get_throughputs(["predict_batch"], estimator.num_samples_per_iter)
        mean_thru = mean_throughputs["predict_batch"]
        msg = f'Expected throughput to be at least {self.cfg.min_throughput}, instead found {mean_thru}'
        assert mean_thru > self.cfg.min_throughput, msg


def setup_tests(cfg):
    tests = []
    if cfg.run_benchmark and cfg.perf_test:
        tests.append(PerfCheck(cfg))
    if cfg.acc_test:
        tests.append(AccuracyCheck(cfg))
    return tests
