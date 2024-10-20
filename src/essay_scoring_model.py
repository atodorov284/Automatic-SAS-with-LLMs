import os
from datasets import load_from_disk
import torch
from transformers import (BertForSequenceClassification, BertTokenizer,
                          Trainer, TrainingArguments)


class EssayScoringModel:
    def __init__(self, model_name='bert-base-uncased', num_labels=4):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def tokenize_function(self, sample, padding="max_length"):
        inputs = sample["EssayText"]
        model_inputs = self.tokenizer(inputs, padding=padding, truncation=True)
        model_inputs["labels"] = sample["Score1"]
        return model_inputs

    def get_tokenized_dataset(self, dataset, is_train, essay_set, batch_size=8):
        dataset_type = "train" if is_train else "test"
        tokenized_path = f"../data/tokenized_set{int(essay_set)}_{dataset_type}"
        if os.path.exists(tokenized_path):
            print(f"Loading tokenized dataset from {tokenized_path}")
            return load_from_disk(tokenized_path)
        else:
            print(f"Tokenizing and saving dataset to {tokenized_path}")
            tokenized_dataset = dataset.map(self.tokenize_function, batched=True, batch_size=batch_size, remove_columns=["EssayText", "Score1", "Id", "EssaySet"])
            tokenized_dataset.save_to_disk(tokenized_path)
            return tokenized_dataset

    def train(self, train_dataset, eval_dataset, output_dir, batch_size=8, epochs=3):
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=0.01,
            logging_dir=f'{output_dir}/logs',
            logging_steps=10,
            evaluation_strategy='epoch'
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        trainer.train()
        return trainer.evaluate()