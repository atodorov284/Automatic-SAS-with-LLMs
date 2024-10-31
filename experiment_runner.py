from abc import ABC, abstractmethod
from datasets import Dataset
import numpy as np
from bert_scoring_model import BertScoringModel
from qwk import quadratic_weighted_kappa
from copy import deepcopy
from tqdm.notebook import tqdm

tqdm.pandas()

RANDOM_SEED = 4242


class ExperimentStrategy(ABC):
    '''
    Abstract class for running experiments.
    '''
    def __init__(self, batch_size: int = 32, epochs: int = 10, patience: int = 2, split_ratio: float = 0.8) -> None:
        '''
        Args:
            batch_size: batch size for training
            epochs: number of epochs for training
            patience: number of epochs to wait before stopping training
            split_ratio: ratio of data to use for training
        '''
        self._model = None
        self._batch_size = batch_size
        self._epochs = epochs
        self._patience = patience
        self._split_ratio = split_ratio
        self._num_essay_sets = 10 + 1  # for range(1, 11)

    def run_experiment(
        self,
        train_data: Dataset,
        test_data: Dataset,
        instruction: bool = False,
        retrain: bool = False,
        evaluate: bool = False,
        use_lora: bool = False,
    ) -> None:
        '''
        Run an experiment with given parameters.

        Args:
            train_data: training data
            test_data: test data
            instruction: whether to use instruction or not
            retrain: whether to retrain the model or not
            evaluate: whether to evaluate the model or not
            use_lora: whether to use LORA or not
        '''
        if retrain:
            print("Tokenizing, cross-validation, and training:")
            self._train_model(train_data, instruction, use_lora)

        if evaluate:
            print("Making predictions on test and computing QWK:")
            self._make_predictions(test_data, instruction, use_lora)

    def _make_predictions(
        self, test_data: Dataset, instruction: bool, use_lora: bool
    ) -> None:
        '''
        Make predictions on test data and compute QWK score.

        Args:
            test_data: test data
            instruction: whether to use instruction or not
            use_lora: whether to use LORA or not
        '''
        qwk_list = []
        for essay_set in range(1, self._num_essay_sets):
            test_set_filtered = deepcopy(test_data[test_data["EssaySet"] == essay_set])
            unique_labels = test_set_filtered["Score1"].nunique()
            model = self._load_model(instruction, essay_set, unique_labels, use_lora)

            test_set_filtered["Predicted Score"] = test_set_filtered.progress_apply(
                lambda row: model.score_single_essay(row["EssayText"]), axis=1
            )

            qwk = quadratic_weighted_kappa(
                test_set_filtered["Score1"],
                test_set_filtered["Predicted Score"],
                min_rating=0,
                max_rating=unique_labels,
            )
            qwk_list.append(qwk)
            print(f"Essay Set {essay_set}: QWK = {qwk}")

        print(f"Average QWK: {np.mean(qwk_list)}")

    @abstractmethod
    def _train_model(self, train_data: Dataset, instruction: bool, use_lora: bool) -> None:
        '''
        Train the model using the given training data.

        Args:
            train_data: the training data
            instruction: whether to use instruction or not
            use_lora: whether to use LORA or not
        '''
        pass

    @abstractmethod
    def _load_model(self, instruction: bool, essay_set: int, num_labels: int, use_lora: bool) -> None:
        """
        Load the model for a specific essay set.

        Args:
            instruction (bool): Whether to use instruction or not.
            essay_set (int): The essay set to load the model for.
            num_labels (int): The number of labels for classification.
            use_lora (bool): Whether to use LoRA or not.
        """
        pass


class TaskSpecificModelStrategy(ExperimentStrategy):
    def __init__(self, batch_size: int = 32, epochs: int = 10, patience: int = 2, split_ratio: float = 0.8) -> None:
        """
        Initialize the TaskSpecificModelStrategy with hyperparameters.

        Args:
            batch_size (int): The batch size for training. Defaults to 32.
            epochs (int): The number of epochs for training. Defaults to 10.
            patience (int): The patience for early stopping. Defaults to 2.
            split_ratio (float): The ratio of data used for training. Defaults to 0.8.
        """
        super().__init__(batch_size, epochs, patience, split_ratio)

    def _load_model(
        self, instruction: bool, essay_set: int, num_labels: int, use_lora: bool
    ) -> BertScoringModel:
        """
        Load the model for a specific essay set.

        Args:
            instruction (bool): Whether to use instruction or not.
            essay_set (int): The essay set to load the model for.
            num_labels (int): The number of labels for classification.
            use_lora (bool): Whether to use LoRA or not.

        Returns:
            BertScoringModel: The loaded model.
        """
        model_path = f"best_models/bert-base-uncased_set{essay_set}_instruction{instruction}_lora{use_lora}/"
        model = BertScoringModel(num_labels=num_labels)
        model.load_trained(model_path, num_labels, use_lora)
        return model

    def _train_model(self, train_data: Dataset, instruction: bool, use_lora: bool) -> None:
        """
        Train the model using the given training data.

        Args:
            train_data (Dataset): The training data.
            instruction (bool): Whether to use instruction or not.
            use_lora (bool): Whether to use LORA or not.
        """
        # self._num_essay_sets = max(train_data['EssaySet']) + 1
        for essay_set in range(1, self._num_essay_sets):
            essay_set_train_data = train_data[
                train_data["EssaySet"] == essay_set
            ].sample(frac=self._split_ratio, random_state=RANDOM_SEED)

            essay_set_validation_data = train_data[
                train_data["EssaySet"] == essay_set
            ].drop(essay_set_train_data.index)

            train_dataset = Dataset.from_dict(
                {
                    "EssayText": essay_set_train_data["EssayText"].values,
                    "Id": essay_set_train_data["Score1"].values,
                    "EssaySet": essay_set_train_data["EssaySet"].values,
                    "Score1": essay_set_train_data["Score1"].values,
                }
            )

            validation_dataset = Dataset.from_dict(
                {
                    "EssayText": essay_set_validation_data["EssayText"].values,
                    "Id": essay_set_validation_data["Score1"].values,
                    "EssaySet": essay_set_validation_data["EssaySet"].values,
                    "Score1": essay_set_validation_data["Score1"].values,
                }
            )

            num_of_labels = essay_set_train_data["Score1"].nunique()
            scoring_model = BertScoringModel(num_labels=num_of_labels)

            tokenized_train_dataset = scoring_model.get_tokenized_dataset(
                train_dataset
            )

            tokenized_val_dataset = scoring_model.get_tokenized_dataset(
                validation_dataset
            )

            evaluation_results = scoring_model.train(
                tokenized_train_dataset,
                tokenized_val_dataset,
                save_name=f"set{essay_set}_instruction{instruction}_lora{use_lora}",
                batch_size=self._batch_size,
                epochs=self._epochs,
                patience=self._patience,
                use_lora=use_lora,
            )

            print(f"Final evaluation results: {evaluation_results}")


class CombinedTaskModelStrategy(ExperimentStrategy):
    def __init__(self, batch_size: int = 32, epochs: int = 10, patience: int = 3, split_ratio: float = 0.8) -> None:
        """
        Initialize the CombinedTaskModelStrategy with hyperparameters.

        Args:
            batch_size (int): The batch size for training. Defaults to 32.
            epochs (int): The number of epochs for training. Defaults to 10.
            patience (int): The patience for early stopping. Defaults to 3.
            split_ratio (float): The ratio of data used for training. Defaults to 0.8.
        """
        super().__init__(batch_size, epochs, patience, split_ratio)
        self._model = None

    def _load_model(self, instruction, essay_set, num_labels, use_lora):
        if self._model is None:
            model_path = f"best_models/bert-base-uncased_all_sets_instruction{instruction}_lora{use_lora}/"
            self._model = BertScoringModel(num_labels=num_labels)
            self._model.load_trained(model_path, num_labels, use_lora)
        return self._model

    def _train_model(self, train_data: Dataset, instruction: bool, use_lora: bool) -> None:
        """
        Train the model using the given training data.

        Args:
            train_data (Dataset): The training data.
            instruction (bool): Whether to use instruction or not.
            use_lora (bool): Whether to use LORA or not.
        """
        # self._num_essay_sets = max(train_data['EssaySet']) + 1

        essay_set_train_data = train_data.sample(
            frac=self._split_ratio, random_state=RANDOM_SEED
        )

        essay_set_validation_data = train_data.drop(essay_set_train_data.index)

        train_dataset = Dataset.from_dict(
            {
                "EssayText": essay_set_train_data["EssayText"].values,
                "Id": essay_set_train_data["Score1"].values,
                "EssaySet": essay_set_train_data["EssaySet"].values,
                "Score1": essay_set_train_data["Score1"].values,
            }
        )

        validation_dataset = Dataset.from_dict(
            {
                "EssayText": essay_set_validation_data["EssayText"].values,
                "Id": essay_set_validation_data["Score1"].values,
                "EssaySet": essay_set_validation_data["EssaySet"].values,
                "Score1": essay_set_validation_data["Score1"].values,
            }
        )

        num_of_labels = train_data["Score1"].nunique()
        scoring_model = BertScoringModel(num_labels=num_of_labels)

        tokenized_train_dataset = scoring_model.get_tokenized_dataset(
            train_dataset
        )

        tokenized_val_dataset = scoring_model.get_tokenized_dataset(
            validation_dataset
        )

        evaluation_results = scoring_model.train(
            tokenized_train_dataset,
            tokenized_val_dataset,
            save_name=f"all_sets_instruction{instruction}_lora{use_lora}",
            batch_size=self._batch_size,
            epochs=self._epochs,
            patience=self._patience,
            use_lora=use_lora,
        )
        print(f"Final evaluation results: {evaluation_results}")


class BaselineBertModelStrategy(ExperimentStrategy):
    def __init__(self, batch_size: int = 32, epochs: int = 10, patience: int = 2, split_ratio: float = 0.8) -> None:
        """
        Initialize the BaselineBertModelStrategy with hyperparameters. They are actually not needed but are there as it inherits from the ABC.

        Args:
            batch_size (int): The batch size for training. Defaults to 32.
            epochs (int): The number of epochs for training. Defaults to 10.
            patience (int): The patience for early stopping. Defaults to 2.
            split_ratio (float): The ratio of data used for training. Defaults to 0.8.
        """
        super().__init__(batch_size, epochs, patience, split_ratio)
        self._model = BertScoringModel(num_labels=4)

    def _load_model(self, instruction: bool = False, essay_set: int = 0, num_labels: int = 0, use_lora: bool = False) -> BertScoringModel:
        """
        Load the model for a specific essay set.

        Args:
            instruction (bool): Whether to use instruction or not.
            essay_set (int): The essay set to load the model for.
            num_labels (int): The number of labels for classification.
            use_lora (bool, optional): Whether to use LoRA. Defaults to False.

        Returns:
            BertScoringModel: The loaded model.
        """
        return self._model

    def _train_model(
        self, train_data: Dataset, instruction: bool, use_lora: bool = False
    ) -> None:
        """
        Train the model using the given training data.

        Args:
            train_data (Dataset): The training data.
            instruction (bool): Whether to use instruction or not.
            use_lora (bool, optional): Whether to use LoRA. Defaults to False.

        Returns:
            None
        """
        print("No training for the BaselineBertModelStrategy.")
