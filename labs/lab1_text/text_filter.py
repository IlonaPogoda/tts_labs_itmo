"""Normalized / non-normalized classifier — skeleton for lab 1.

Run as a script to score yourself on the development set::

    python text_filter.py
"""

import csv
import re
import unicodedata
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

DEV_SET_PATH = "/home/lonaogoda/tts_labs_itmo/labs/lab1_text/data/dev_sentences.csv"


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
        # Создание регулярных выражений для проверки текста
        # 1. Цифры (количественные, порядковые числительные, номера моделей)
        self.re_digits = re.compile(r"\d")

        # 2. Иностранные слова (латиница)
        self.re_latin = re.compile(r"[A-Za-z]")

        # 3. Символы валют, единиц измерения, процентов, спецзнаки
        self.re_designations = re.compile(r"[%°\$€£₽№©§\+=\^~]")

        # 4. Технические символы и артефакты разметки
        self.re_tech = re.compile(r"[\*\/@<>\#_\\\|]")

        # 5. Инициалы и сокращения с точками:
        # - Инициалы: "Г. П.", "Л.И.", "А. Б."
        # - Частые сокращения: "г.", "ул.", "д.", "прим.", "руб.", "коп.", "т.д.", "т.п."
        # - Одиночные согласные или короткие слова с точкой в середине фразы (не в конце предложения)
        self.re_initials = re.compile(r"\b[А-ЯЁ]\.\s*[А-ЯЁ]?\.")
        self.re_abbrev_dot = re.compile(
            r"\b(?:г|ул|д|им|т\.д|т\.п|т\.е|т\.к|руб|коп|прим|см|стр|в|напр|кв)\.",
            re.IGNORECASE,
        )
        self.re_mid_sentence_dot = re.compile(r"\b[а-яёА-ЯЁ]{1,4}\.\s+[а-яё]")

        # 6. Аббревиатуры и акронимы:
        # От 2 заглавных букв подряд (КПСС, МГУ, США, ТУ-129)
        self.re_acronyms = re.compile(r"\b[А-ЯЁ]{2,}\b")

        # 7. Запрещенные скобки и неразрешенные символы пунктуации
        self.re_brackets = re.compile(r"[\(\)\[\]\{\}]")

        # 8. Нестандартные повторяющиеся знаки препинания (например, !.., ??, !?)
        self.re_bad_punct = re.compile(r"[!\?]\.{1,}|[!]{2,}|\?{2,}")

        # Допустимый алфавит символов (кириллица, ударение, базовая пунктуация, пробелы)
        self.allowed_punct = set(".,!?:;\"'«»…—- ")

    def _has_invalid_characters(self, text: str) -> bool:
        """Проверяет наличие символов вне разрешенного алфавита."""
        for ch in text:
            # Разрешаем символ ударения U+0301
            if ch == "\u0301":
                continue
            # Разрешаем кириллические буквы
            if "CYRILLIC" in unicodedata.name(ch, ""):
                continue
            # Разрешаем базовую пунктуацию и пробелы
            if ch in self.allowed_punct:
                continue
            return True
        return False

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

        if not text or not isinstance(text, str):
            return 0

        text_stripped = text.strip()
        if not text_stripped:
            return 0

        # Проверка 1: Цифры
        if self.re_digits.search(text_stripped):
            return 0

        # Проверка 2: Латиница
        if self.re_latin.search(text_stripped):
            return 0

        # Проверка 3: Знаки обозначений и валют (%, $, ₽ и т.д.)
        if self.re_designations.search(text_stripped):
            return 0

        # Проверка 4: Технические символы (*, /, <>, @)
        if self.re_tech.search(text_stripped):
            return 0

        # Проверка 5: Скобки
        if self.re_brackets.search(text_stripped):
            return 0

        # Проверка 6: Нестандартная пунктуация (!.., ?..)
        if self.re_bad_punct.search(text_stripped):
            return 0

        # Проверка 7: Сокращения и инициалы
        if self.re_initials.search(text_stripped):
            return 0
        if self.re_abbrev_dot.search(text_stripped):
            return 0
        if self.re_mid_sentence_dot.search(text_stripped):
            return 0

        # Проверка 8: Акронимы / Аббревиатуры из заглавных букв
        if self.re_acronyms.search(text_stripped):
            return 0

        # Проверка 9: Общая проверка на неизвестные/нестандартные Юникод-символы
        if self._has_invalid_characters(text_stripped):
            return 0

        return 1


if __name__ == "__main__":
    textfilter = TextFilter()

    dev_files = pd.read_csv(
        DEV_SET_PATH, sep="|", encoding="utf-8", quoting=csv.QUOTE_NONE, header=0
    )

    dev_files["predicted"] = dev_files["text"].apply(textfilter.filter)
    
    prc = precision_score(dev_files["is_normalized"], dev_files["predicted"])
    rec = recall_score(dev_files["is_normalized"], dev_files["predicted"])
    f1 = f1_score(dev_files["is_normalized"], dev_files["predicted"])
    print(f"F1 Score is {f1:.4f}, Precision is {prc:.4f}, Recall is {rec:.4f}")
