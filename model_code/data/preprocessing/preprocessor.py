from typer import Option
import typer
from sklearn.model_selection import train_test_split
import pandas as pd
from typing import List, Tuple
from pathlib import Path
import logging
import re

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

    def cleaning(self):
        texts = self.dataset["Text"].to_numpy()
        cleaned_texts = list()
        indexes_to_remove = list()
        for index, sentence in enumerate(texts):
            # lowercase
            sentence = sentence.lower()
            # remove paragraph numbers
            sentence = re.sub("\[\d*\]", "", sentence.strip())
            # removing new lines
            sentence = sentence.replace("\n", " ")
            # removing tabs
            sentence = sentence.replace("\t", " ")
            # removing unicode chars
            sentence = re.sub("[^\w\s]", "", sentence)
            cleaned_texts.append(sentence)
            if isinstance(sentence, float) or len(sentence) < 3:
                indexes_to_remove.append(index)
            sentence = sentence.lstrip()

        self.dataset["Text"] = cleaned_texts
        # remove nan
        self.dataset["Text"].dropna(inplace=True)
        # remove too short sentences
        self.dataset.drop(indexes_to_remove, inplace=True)

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
            self.dataset,
            test_size=val_examples,
            random_state=self.seed,
            stratify=self.dataset[self.target_column],
        )
        val, test = train_test_split(
            val,
            test_size=test_examples,
            random_state=self.seed,
            stratify=val[self.target_column],
        )

        return train, val, test

    @staticmethod
    def save_dataset(dataset: pd.DataFrame, out_path: str):
        logger.info(f"Saving dataset to {out_path}")
        dataset.to_csv(out_path, index=False)

    def create_target_variable(self, target_lang: str = "Italian"):
        logger.info("Creating target variable")
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
    languages_to_exclude: List[str] = Option([[], "-e", "--exclude"]),
    target_language: str = Option("Italian", "-l", "--language"),
):
    preprocessor = Preprocessor(
        dataset_path=data_path,
        text_column=text_column,
        target_column=target_column,
        random_seed=random_seed,
    )

    preprocessor.cleaning()
    preprocessor.remove_languages(languages_to_remove=languages_to_exclude)
    preprocessor.create_target_variable(target_lang=target_language)
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
