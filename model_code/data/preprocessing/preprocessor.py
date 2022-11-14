from typer import Option
import typer
from sklearn.model_selection import train_test_split
import pandas as pd
from typing import List, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Preprocessor:
    def __init__(
        self,
        dataset_path: str,
        text_column: str,
        target_column: str,
        random_seed: int = 12,
    ) -> None:
        self.dataset = pd.read_csv(dataset_path)
        self.text_column = text_column
        self.target_column = target_column
        self.seed = random_seed

    def remove_languages(self, languages_to_remove: List[str] = ["Kannada"]):
        condition = self.dataset[self.target_column].isin(languages_to_remove)
        self.dataset = self.dataset[~condition]

    # not here
    def create_vocabulary(self):
        words = dict()

        for sentence in self.dataset[self.text_column].to_numpy():
            # simple tokenization
            # lower case sentence
            for word in sentence.lower().split():
                if word not in words:  # vocabulary
                    words[word] = len(words)  # Assign each word with a unique index
        return words

    def split_dataset(
        self, val_size: float = 0.3, test_size: float = 0.1
    ) -> Tuple[pd.DataFrame]:
        # training size
        train_size = 1 - val_size - test_size
        logger.info(f"Dataset examples: {len(self.dataset)}")

        # compute the number of examples
        train_examples = int(len(self.dataset) * train_size)
        logger.info(f"Training examples: {train_examples}")

        val_examples = int(len(self.dataset) * val_size)
        logger.info(f"Validation examples: {val_examples}")

        test_examples = int(len(self.dataset) * test_size)
        logger.info(f"Test examples: {test_examples}")

        differences = len(self.dataset) - sum(
            [train_examples, val_examples, test_examples]
        )

        train_examples += differences

        assert sum([train_examples, val_examples, test_examples]) == len(
            self.dataset
        ), "Dataset size issue"

        train, val = train_test_split(
            self.dataset, test_size=val_examples, random_state=self.seed
        )
        val, test = train_test_split(
            val, test_size=test_examples, random_state=self.seed
        )

        return train, val, test

    @staticmethod
    def save_dataset(dataset: pd.DataFrame, out_path: str):
        dataset.to_csv(out_path, index=False)

    def create_target_variable(self, target_lang: str = "Italian"):
        self.dataset["target"] = [
            1 if row == target_lang else 0
            for row in self.dataset[self.target_column].to_numpy()
        ]


def preprocess(
    data_path: str = Option(..., "-d", "--dataset"),
    val_size: float = Option(0.3, "-v", "--val-size"),
    test_size: float = Option(0.1, "-t", "--test-size"),
    text_column: str = Option("Text", "-T", "--text-col"),
    target_column: str = Option("Language", "-y", "--target-col"),
    random_seed: int = Option(12, "-s", "--seed"),
    output_path: str = Option(..., "-o", "--output-path"),
):
    preprocessor = Preprocessor(
        dataset_path=data_path,
        text_column=text_column,
        target_column=target_column,
        random_seed=random_seed,
    )

    preprocessor.remove_languages()
    preprocessor.create_target_variable()
    train, val, test = preprocessor.split_dataset(
        val_size=val_size, test_size=test_size
    )
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    # save splits
    preprocessor.save_dataset(dataset=train, out_path=output_path / "train.csv")
    preprocessor.save_dataset(dataset=val, out_path=output_path / "val.csv")
    preprocessor.save_dataset(dataset=test, out_path=output_path / "test.csv")


if __name__ == "__main__":
    typer.run(preprocess)
