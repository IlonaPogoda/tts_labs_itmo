"""Normalized / non-normalized classifier — skeleton for lab 1.

Run as a script to score yourself on the development set::

    python text_filter.py
"""

import csv

import pandas as pd
from sklearn.metrics import f1_score

DEV_SET_PATH = "data/dev_sentences.csv"


class TextFilter:
    """Decides whether an utterance is usable as a training example.

    Example:
        >>> textfilter = TextFilter()
        >>> textfilter.filter("Я вышел из дома.")
        1
        >>> textfilter.filter("Александрову Г. П.")
        0
    """

    def __init__(self):
        """Prepare the classifier's resources.

        Compiled regular expressions, abbreviation and contraction dictionaries, a
        trained model — anything that should not be rebuilt for every utterance.
        """

        # Here goes your initialization logic

        pass

    def filter(self, text: str) -> int:
        """Classify a single utterance.

        Args:
            text: Utterance text, already passed through :class:`TextNormalizer`.

        Returns:
            ``1`` if the text is normalized and the utterance can be used for
            training; 
            ``0`` if it contains something the speaker pronounced
            differently from how it is written, and the utterance should be dropped.
        """

        # Here goes your normalisation logic

        return 1


if __name__ == "__main__":
    textfilter = TextFilter()

    dev_files = pd.read_csv(
        DEV_SET_PATH, sep="|", encoding="utf-8", quoting=csv.QUOTE_NONE, header=0
    )

    dev_files["predicted"] = dev_files["text"].apply(textfilter.filter)

    f1 = f1_score(dev_files["is_normalized"], dev_files["predicted"])
    print(f"F1 Score is {f1:.4f}")
