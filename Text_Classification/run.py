import argparse
import logging
import math
import os
import random
from pathlib import Path
from absl import logging 
from absl import flags

import datasets
from datasets import load_dataset, load_metric
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from huggingface_hub import Repository


import transformers
from accelerate import Accelerator
from huggingface_hub import Repository
from transformers import (
    AdamW,
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)

from transformers.file_utils import get_full_repo_name
from transformers.utils.versions import require_version
from finetun_bert_model import BertForSequenceClassification


FLAGS= flags.FLAGS

train_file="data/processed/external_train.csv" 

validation_file="data/processed/external_eval.csv"

max_length=415

pad_to_max_length=False

model_name_or_path="ckiplab/bert-base-chinese"

use_slow_tokenizer=False

per_device_train_batch_size=12

per_device_eval_batch_size=32

learning_rate=5e-5

weight_decay=0

num_train_epochs=3

max_train_steps=None

gradient_accumulation_steps=2

lr_scheduler_type="linear"

num_warmup_steps=500

output_dir="model"

seed=42


def main(argv): 
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    #---------------------------
    ## Loading Data for Training
    #1. Preparing dataset with load_dataset api
    #2. 
    # ------------------------- 
    data_files = {}
    if FLAGS.train_file is not None:
        data_files["train"] = FLAGS.train_file
    if FLAGS.validation_file is not None:
        data_files["validation"] = FLAGS.validation_file
    extension = (FLAGS.train_file if FLAGS.train_file is not None else FLAGS.valid_file).split(".")[-1]
    raw_datasets = load_dataset(extension, data_files=data_files)

    label_list = raw_datasets["train"].unique("label")
    label_list.sort()
    num_labels = len(label_list)
    label_to_id = {v: i for i, v in enumerate(label_list)}
    id2label = {id: label for label, id in label_to_id.items()}

    sentence1_key, sentence2_key = "sentence1", "sentence2"

    ### Dataloader section: 
    #1. Preparing dataloader with (accelerator for faster loading data)
    def preprocess_function(examples):
        
        # Tokenize the texts
        texts = ((examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key]))
        result = tokenizer(*texts, padding=False, max_length=args.max_length, truncation=True)

        # Map labels to IDs
        result["labels"] = [label_to_id[l] for l in examples["label"]]
        return result
    accelerator = Accelerator()
    accelerator.wait_for_everyone()
    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(preprocess_function, 
            batched= True, 
            remove_columns= raw_datasets['train'].column_names, 
            desc="Running tokenizer on dataset", 
            )

    #Dataset to dataloader 
    data_collator= DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fpt16 else None))

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]
    train_dataloader = DataLoader(train_dataset, shuffle=True, 
                        collate_fn=data_collator, 
                        batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)


    #----------------------------------
    #Loading model section 
    #----------------------------------
    config = AutoConfig.from_pretrained(FLAGS.model_name_or_path, num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(FLAGS.model_name_or_path, use_fast=not FLAGS.use_slow_tokenizer)
    model = BertForSequenceClassification.from_pretrained( FLAGS.model_name_or_path,config=config,) 
    model.config.label2id = label_to_id
    model.config.id2label = id2label

    #----------------------------------
    # Training Strategy Configuration
    # 1. Optimizer 
    # 2. Learning Rate schedule 
    # 3. Accelerator API for speeding up Training 
    #----------------------------------

    ## Configure Optimizer --> with weight 
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": FLAGS.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=FLAGS.learning_rate)
    ## Scheduler the learning rate with training steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / FLAGS.gradient_accumulation_steps)
    if FLAGS.max_train_steps is None:
        FLAGS.max_train_steps = FLAGS.num_train_epochs * num_update_steps_per_epoch
    else:
        FLAGS.num_train_epochs = math.ceil(FLAGS.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=FLAGS.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=FLASG.num_warmup_steps,
        num_training_steps=FLASG.max_train_steps,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader)
    
    # Metric evaluate model
    metric = load_metric("f1")

    #--------------------------------------
    # Configure Train and Evaluation Loop
    #--------------------------------------
    total_batch_size = FLAGS.
     