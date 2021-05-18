import numpy as np

class LearnerRuntimeMetric:

    def evaluate(self, gt_runtimes: np.ndarray, predicted_scores: np.ndarray, feature_cost: float, algorithm_cutoff_time: int):
        return 0

    def get_name(self):
        return "learner_runtime_s_per_step"