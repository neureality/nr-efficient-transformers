# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from dataclasses import dataclass, field

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModelForCausalLM
from torch.utils.data import random_split
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

from QEfficient.training import QEffTrainer, QEffTrainingArguments


@dataclass
class Args:
    model_id: str = field(default="facebook/opt-125m", metadata={"help": "Pre-trained model to fine-tune"})
    dataset_name: str = field(
        default="xiyuez/red-dot-design-award-product-description",
        metadata={"help": "Dataset to fine-tune on"},
    )
    train_frac: float = field(default=0.85, metadata={"help": "Fraction of training dataset to split"})


parser = HfArgumentParser((Args, QEffTrainingArguments))
args, train_args = parser.parse_args_into_dataclasses()

tokenizer = AutoTokenizer.from_pretrained(args.model_id)
model_kwargs = {}
if "opt-" in args.model_id:
    model_kwargs["dropout"] = 0.0
model = AutoModelForCausalLM.from_pretrained(args.model_id, use_cache=False, **model_kwargs)

lora_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
    bias="none",
)

peft_model = PeftModelForCausalLM(model, lora_config, train_args.adapter_name)
peft_model.print_trainable_parameters()

rd_df = load_dataset(args.dataset_name)["train"].to_pandas()[:200]
instruction = (
    """### Instruction:
Create a detailed description for the following product: {product}, belonging to category: {category}
### Response:
{description}"""
    + tokenizer.eos_token
)
ds = [
    dict(
        tokenizer(
            instruction.format(**row),
            text_target=instruction.format(**row),
            return_tensors="pt",
            padding="max_length",
            max_length=train_args.max_ctx_len,
            truncation=True,
        )
    )
    for i, row in tqdm(rd_df.iterrows(), total=len(rd_df))
]
random_generator = torch.Generator().manual_seed(37)
train_ds, eval_ds = random_split(ds, [args.train_frac, 1 - args.train_frac], random_generator)


def collate(batch):
    out_batch = {}
    for key in batch[0]:
        out_batch[key] = torch.cat([x[key] for x in batch], 0)
    return out_batch


trainer = QEffTrainer(peft_model, train_args, collate, train_ds, eval_ds)
trainer.train()

for i, sample in enumerate(eval_ds):
    output = model.generate(**sample)
    print(tokenizer.decode(output[0]))
    if i == 2:
        break
