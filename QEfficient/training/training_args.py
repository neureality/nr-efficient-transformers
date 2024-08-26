# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os
from dataclasses import dataclass, field
from typing import List, Optional

from transformers import TrainingArguments


@dataclass
class QEffTrainingArguments(TrainingArguments):
    adapter_name: str = field(default="default", metadata={"help": "Adapter name loaded in the model"})
    max_ctx_len: int = field(default=128, metadata={"help": "Sequence length to compile and train for"})
    validate: bool = field(default=True, metadata={"help": "Validate the training ops in ONNX-Runtime"})
    qpc_path: Optional[str] = field(default=None, metadata={"help": "Location where to save qpc"})
    num_cores: int = field(default=14, metadata={"help": "Number of cores to compile for"})
    qeff_fp16: bool = field(default=True, metadata={"help": "Use FP16"})
    mxfp6_matmul: bool = field(default=False, metadata={"help": "Use MXFP6 for matmul"})
    device_ids: Optional[List[int]] = field(default=None, metadata={"help": "Device IDs to run on"})

    def __post_init__(self):
        super().__post_init__()
        if self.qpc_path is None:
            self.qpc_path = os.path.join(self.output_dir, "qpc")
