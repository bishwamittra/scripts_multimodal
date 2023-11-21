from pathlib import Path
from typing import Optional

from pydantic import BaseModel


class TaskBase(BaseModel):
    task_name: Optional[str]


class MetaBase(BaseModel):
    num_sockets: Optional[int]
    kind: Optional[str]
    pod_name: Optional[str]
    script: Optional[str]
    yaml_path: Optional[Path]
    pef: Optional[Path]


class ModelBase(BaseModel):
    mapping: str
    inference: bool
    data_parallel: bool
    reduce_on_rdu: bool


class RunBase(BaseModel):
    enable_stoc_rounding: bool
    epochs: int
    mode: str
    num_workers: int
    pinned_memory: bool
    use_distributed_val: bool
    use_sambaloader: bool


class OptimBase(BaseModel):
    pass


class OutputBase(BaseModel):
    output_folder: Path
    log_dir: Optional[Path]
    ckpt_dir: Optional[Path]


class RegressionTestBase(BaseModel):
    acc_test: Optional[bool]
    mock_inference: bool


class DataBase(BaseModel):
    data_dir: Optional[Path]
