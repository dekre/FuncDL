from enum import Enum
from typing import List, Optional, Tuple
from pydantic import BaseModel, root_validator
from .utils import get_multiple_vals_from_dict


class FunDLObjective(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class FunDLnBlockConfig(BaseModel):
    units: int
    activation: str
    add_dropout: bool = True
    dropout_rate: float = None
    name: str

    class Config:
        extra = "allow"


def get_default_hidden_layers():
    layers = []
    layers.append(
        FunDLnBlockConfig(
            units=64, activation="sigmoid", name="hidden01", add_dropout=False
        )
    )
    layers.append(
        FunDLnBlockConfig(
            units=64, activation="linear", name="hidden02", add_dropout=False
        )
    )
    return layers


class FunDLPipelineConfig(BaseModel):
    # resp,
    # func_cov,
    # scalar_cov = (NULL,)
    objective: FunDLObjective
    basis_choice: Optional[str] = ["fourier"]
    num_basis: Optional[int] = [7]
    hidden_layers: Optional[List[FunDLnBlockConfig]] = get_default_hidden_layers()
    domain_range: List[Tuple[float, float]] = [(0, 1)]
    epochs: Optional[int] = 100
    loss_choice: Optional[str] = "mse"
    metric_choice: Optional[str] = "mean_squared_error"
    val_split: Optional[float] = 0.2
    learning_rate: Optional[float] = 0.001
    early_stopping_patience: Optional[int] = 15
    early_stopping: Optional[bool] = True
    verbose: Optional[bool] = True
    batch_size: Optional[int] = 32
    decay_rate: Optional[float] = 0
    func_resp_method: Optional[int] = 1
    covariate_scaling: Optional[bool] = True
    raw_data: Optional[bool] = False


@root_validator
def validate_domain_basis_choices(cls, values):
    domain_range, num_basis, basis_choice = get_multiple_vals_from_dict(
        "domain_range", "num_basis", "basis_choice", obj=values
    )
    if len(domain_range) != len(num_basis):
        raise ValueError(
            "The length of domain ranges doesn't match length of num_basis."
        )
    if len(domain_range) != len(basis_choice):
        raise ValueError(
            "The length of domain ranges doesn't match length of basis choices."
        )
    if len(num_basis) != len(basis_choice):
        raise ValueError(
            "The length of num_basis doesn't match length of basis choices."
        )
