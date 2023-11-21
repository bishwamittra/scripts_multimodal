from classification.estimators.rescale_estimator import RescaleEstimator


class CIFAREstimator(RescaleEstimator):
    def _assert_model_config_ok(self):
        assert self.in_height == self.in_width, "Only square images are supported while training, predicting"
        super()._assert_model_config_ok()
