import numpy as np
from .config import FunDLPipelineConfig, FunDLObjective


class DataProcessing(object):
    def __init__(
        self,
        config: FunDLPipelineConfig,
    ):
        self.__config = config

    def transform(self, y_true: np.ndarray, f_cov: np.ndarray, z_cov: np.ndarray):
        y_true = self.__process_y_true(y_true)

    def __process_y_true(self, y_true: np.ndarray) -> np.ndarray:
        if self.__config.objective is FunDLObjective.CLASSIFICATION:
            _, indices = np.unique(y_true, return_inverse=True)
            return indices
        elif self.__config is FunDLObjective.REGRESSION:
            return y_true

    def __compute_output_dim(self, y_true: np.ndarray) -> int:
        if len(y_true.shape) == 1:
            return 1
        elif len(y_true.shape) == 2:
            return y_true.shape[1]
        else:
            raise ValueError(
                f"Can only accept 1D or 2D arrays: not ({len(y_true.shape)=}): "
            )
