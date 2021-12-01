import argparse
import logging
import math
import os
from absl import app
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

import wandb
from wandb.keras import WandbCallback


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
from finetune_bert_model import BertForSequenceClassification


FLAGS= flags.FLAGS

flags.DEFINE_string(
"train_file", "./data/processed/external_train.csv" , 
"Path directory for training dataset.", 
)
flags.DEFINE_string(
"validation_file", "./data/processed/external_eval.csv", 
"Validation path for val dataset."
)

flags.DEFINE_string(
"model_name_or_path", "ckiplab/bert-base-chinese", 
"Path pretrain weight Bert_base_chinese_model"
)
flags.DEFINE_string(
"output_dir" , "./pre_train", 
"directory saving, export model"
)

flags.DEFINE_integer(
"max_length", 415,
"The max length of input token"
)

flags.DEFINE_boolean(
"pad_to_max_length", False, 
"Setting padding to match same length for all sentence"
)

flags.DEFINE_boolean(
"use_slow_tokenizer", False, 
" methods downloading/caching/loading pretrained tokenizers as well as adding tokens to the vocabulary."
)
flags.DEFINE_integer(
"per_device_train_batch_size", 300, 
"Train batch_size"
)
flags.DEFINE_integer(
"per_device_eval_batch_size", 300,
"Validation batch_size." 
)

flags.DEFINE_float( 
"weight_decay", 0., # 1e-6
"Amount of weight decay")

flags.DEFINE_integer(
"num_train_epochs", 10, 
"Number of training epochs")

flags.DEFINE_integer(
"max_train_steps", None, 
"Pre-Define maximum number of steps training ")


flags.DEFINE_integer(
"gradient_accumulation_steps", 2, # Using for Multi-GPUs training
"running a configured number of steps without updating the model variables"
)

flags.DEFINE_float( 
"learning_rate", 5e-5, 
"Initial learning rate set values")

flags.DEFINE_enum(
"lr_scheduler_type", "linear", ['no', 'linear', 'sqrt'], 
"Scaling learning rate")

flags.DEFINE_integer(
"num_warmup_steps", 500, # recommend calculate warmup steps
"Predefine number of warmup steps without calculate Size of training and Batch_size")

flags.DEFINE_integer(
"seed", 42, 
"Initial value for random seed")



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
        result = tokenizer(*texts, padding=False, max_length=FLAGS.max_length, truncation=True)

        # Map labels to IDs
        result["labels"] = [label_to_id[l] for l in examples["label"]]
        return result

    # Multi-GPUs training Configure 
    accelerator = Accelerator()
    accelerator.wait_for_everyone()
    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(preprocess_function, 
            batched= True, 
            remove_columns= raw_datasets['train'].column_names, 
            desc="Running tokenizer on dataset", 
            )

    #Dataset to dataloader 
    tokenizer = AutoTokenizer.from_pretrained(FLAGS.model_name_or_path, use_fast=not FLAGS.use_slow_tokenizer)
    data_collator= DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fpt16 else None))

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]

    train_dataloader = DataLoader(train_dataset, shuffle=True, 
                        collate_fn=data_collator, 
                        batch_size=FLAGS.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=FLAGS.per_device_eval_batch_size)


    #----------------------------------
    #Loading model section 
    #----------------------------------
    config = AutoConfig.from_pretrained(FLAGS.model_name_or_path, num_labels=num_labels)
    
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
        num_warmup_steps=FLAGS.num_warmup_steps,
        num_training_steps=FLAGS.max_train_steps,)
   
    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader)
    
    # Metric evaluate model (F1 Score to measure the Error)
    metric = load_metric("f1")
    
    #--------------------------------------
    # Configure Train and Evaluation Loop
    #--------------------------------------
    total_batch_size = FLAGS.per_device_train_batch_size * accelerator.num_processes * FLAGS.gradient_accumulation_steps
    
    
    ## Configure model Tracking result 
    configs = {

        "Model_Arch": "FineTune_BertChineseBase",
        "Training mode": "Supervised",
        "Dataset": "Chinese_Text_setiment_Aidea",
        "Epochs": FLAGS.num_train_epochs,
        "Batch_size": total_batch_size,
        "Learning_rate": FLAGS.learning_rate,
        "Optimizer": "AdamW",

    }
    wandb.init(project="NLP_finetune_BERT_Architecture",
               sync_tensorboard=True, config=configs)

    
    
    progress_bar = tqdm(range(FLAGS.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    for epoch in range(FLAGS.num_train_epochs):
        model.train()
        total_loss=0
        num_batch=0
        
        for step, batch in enumerate(train_dataloader):
            
            #Forward path
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / FLAGS.gradient_accumulation_steps
            total_loss += loss
            num_batch += 1

            # Backprobagation path
            accelerator.backward(loss)
            if step % FLAGS.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if completed_steps >= FLAGS.max_train_steps:
                break
        # Epoch Loss training
        epoch_loss= total_loss/num_batch

        model.eval()
        with torch.no_grad():
            for step, batch in enumerate(eval_dataloader):
                outputs = model(**batch)
                predictions = outputs.logits.argmax(dim=-1)
                metric.add_batch(
                    predictions=accelerator.gather(predictions),
                    references=accelerator.gather(batch["labels"]),
                )

        ## Configure Checking the output
        wandb.log({
            "epochs": epoch+1,
            "train/total_loss": epoch_loss,
        })
        
    eval_metric = metric.compute()
    print(f"epoch {epoch}: {eval_metric}")


    if FLAGS.output_dir is not None:
        accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(FLAGS.output_dir, save_function=accelerator.save)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(FLAGS.output_dir)

# Pre-Training and Finetune
if __name__ == '__main__':

    app.run(main)
