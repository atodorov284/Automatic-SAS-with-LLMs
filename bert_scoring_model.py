import torch
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    AutoModelForSequenceClassification,
)

from tqdm.notebook import tqdm

tqdm.pandas()


class BertScoringModel:
    """
    Class to encapsulate the BERT model, tokenizer, and all operations related to
    tokenization, training, evaluation, and prediction.

    """

    def __init__(self, num_labels: int) -> None:
        """
        Class to encapsulate the BERT model, tokenizer, and all operations related to
        tokenization, training, evaluation, and prediction.

        Attributes:
            _model_name (str): The name of the BERT model.
            _tokenizer (BertTokenizer): The BERT tokenizer.
            _model (BertForSequenceClassification): The BERT model.
            _padding (str): The padding strategy.
            _device (torch.device): The device to be used for training.
        """

        self._model_name = "bert-base-uncased"
        self._tokenizer = BertTokenizer.from_pretrained(self._model_name)
        self._model = BertForSequenceClassification.from_pretrained(
            self._model_name, num_labels=num_labels
        )

        self._padding = "max_length"
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)

    def tokenize_function(self, sample: dict) -> dict:
        """
        Tokenize a given sample using the BERT tokenizer.

        Args:
            sample (dict): The sample to be tokenized.

        Returns:
            dict: The tokenized sample.
        """
        inputs = sample["EssayText"]
        model_inputs = self._tokenizer(inputs, padding=self._padding, truncation=True)
        model_inputs["labels"] = [int(label) for label in sample["Score1"]]
        return model_inputs

    def get_tokenized_dataset(
        self, dataset: torch.utils.data.Dataset, batch_size: int = 8
    ) -> torch.utils.data.Dataset:
        """
        Tokenize a given dataset using the BERT tokenizer.

        Args:
            dataset (torch.utils.data.Dataset): The dataset to be tokenized.
            batch_size (int, optional): The batch size to use. Defaults to 8.

        Returns:
            torch.utils.data.Dataset: The tokenized dataset.
        """
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            batch_size=batch_size,
            remove_columns=["EssaySet", "Id"],
        )
        return tokenized_dataset

    def train(
        self,
        train_dataset: torch.utils.data.Dataset,
        eval_dataset: torch.utils.data.Dataset,
        save_name: str,
        batch_size: int = 8,
        epochs: int = 10,
        patience: int = 2,
        use_lora: bool = False,
    ) -> dict:
        """
        Train the BERT model on a given dataset.

        Args:
            train_dataset (torch.utils.data.Dataset): The training dataset.
            eval_dataset (torch.utils.data.Dataset): The evaluation dataset.
            save_name (str): The name to save the model with.
            batch_size (int, optional): The batch size to use. Defaults to 8.
            epochs (int, optional): The number of epochs to train for. Defaults to 10.
            patience (int, optional): The patience for early stopping. Defaults to 2.
            use_lora (bool, optional): Whether to use LoRA. Defaults to False.
        """
        if use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=16,
                lora_alpha=16,
                target_modules=["query", "value"],
                lora_dropout=0.1,
            )
            self._model = get_peft_model(self._model, lora_config)

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
            remove_unused_columns=True,
        )
        trainer = Trainer(
            model=self._model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
        )
        trainer.train()
        best_model_dir = f"best_models/{self._model_name}_{save_name}/"
        trainer.save_model(output_dir=best_model_dir)
        print(f"Model saved at directory {best_model_dir}")
        return trainer.evaluate()

    def score_single_essay(self, essay_text: str) -> int:
        """
        Score a single essay.

        Args:
            essay_text (str): The essay text.

        Returns:
            int: The predicted score.
        """
        self._model.eval()
        inputs = self._tokenizer(
            essay_text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
        )
        inputs = {key: value.to(self._device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        logits = outputs.logits
        predicted_score = torch.argmax(logits, dim=-1).item()
        return predicted_score

    def load_trained(self, path: str, num_labels: int, use_lora: bool = False) -> None:
        """
        Load a trained BERT model.

        Args:
            path (str): The path to the trained model.
            num_labels (int): The number of labels.
            use_lora (bool, optional): Whether to use LoRA. Defaults to False.
        """
        self._model = AutoModelForSequenceClassification.from_pretrained(
            path, num_labels=num_labels
        )
        if use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=16,
                lora_alpha=16,
                target_modules=["query", "value"],
                lora_dropout=0.1,
            )
            self._model = PeftModel(self._model, lora_config)
        self._model.to(self._device)
