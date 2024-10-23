import pandas as pd
from datasets import Dataset
import numpy as np
from datasets import load_from_disk
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    AutoModelForSequenceClassification,
)

from qwk import quadratic_weighted_kappa

import os
import torch
from score_essays import score_essay
from copy import deepcopy

from tqdm.notebook import tqdm
tqdm.pandas()

class BertScoringModel:
    def __init__(self, num_labels=4):
        self._model_name = "bert-base-uncased"
        self._tokenizer = BertTokenizer.from_pretrained(self._model_name)
        self._model = BertForSequenceClassification.from_pretrained(
            self._model_name, num_labels=num_labels
        )

        self._padding = "max_length"

    def tokenize_function(self, sample):
        inputs = sample["EssayText"]
        model_inputs = self._tokenizer(inputs, padding=self._padding, truncation=True)
        model_inputs["labels"] = [int(label) for label in sample["Score1"]]
        return model_inputs

    def get_tokenized_dataset(self, dataset, is_train, essay_set, batch_size=8):
        dataset_type = "train" if is_train else "test"
        tokenized_path = (
            f"data/tokenized_data/tokenized_set{int(essay_set)}_{dataset_type}"
        )
        if os.path.exists(tokenized_path):
            print(f"Loading tokenized dataset from {tokenized_path}")
            return load_from_disk(tokenized_path)
        else:
            print(f"Tokenizing and saving dataset to {tokenized_path}")
            tokenized_dataset = dataset.map(
                self.tokenize_function,
                batched=True,
                batch_size=batch_size,
                remove_columns=["EssaySet", "Id"],
            )
            tokenized_dataset.save_to_disk(tokenized_path)
            return tokenized_dataset

    def train(self, train_dataset, eval_dataset, essay_set,  batch_size=8, epochs=10, patience=2,):
        training_args = TrainingArguments(
            output_dir=f"./results/{self._model_name}",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=0.01,
            logging_dir=f"./results/{self._model_name}/logs",
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            remove_unused_columns=True
        )
        trainer = Trainer(
            model=self._model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
        )
        trainer.train()
        best_model_dir = f"best_models/{self._model_name}_set{essay_set}/"
        trainer.save_model(output_dir=best_model_dir)
        print(f"Model saved at directory {best_model_dir}")
        return trainer.evaluate()