"""Russian text normalizer — skeleton for lab 1.

Brings corpus text into a form usable for training a speech synthesizer.
"""


class TextNormalizer:
    """Normalizes text in Russian.


        "!.."           -> "!"
        "«цитата»"      -> '"цитата"'
        "текст * мусор" -> "текст мусор"
        "де‑факто"      -> "де-факто"      # U+2011 -> ordinary hyphen

    **Word-changing edits.** The alignment for that utterance becomes invalid and the
    row must be dropped from the training set — but the logic itself is still needed
    for lab 5, where arbitrary user input arrives with no alignment at all::

        "в 1995 г."     -> "в тысяча девятьсот девяносто пятом году"
        "прим. автора"  -> "примечание автора"

    Example:
        >>> normalizer = TextNormalizer()
        >>> normalizer.normalize("Расстреливать надо таких писателей!..")
        'Расстреливать надо таких писателей!'
    """

    def __init__(self):
        """Prepare the normalizer's resources.

        Put anything expensive to build here: compiled regular expressions,
        abbreviation and contraction dictionaries, a morphological analyzer.
        Building them inside :meth:`normalize` means building them 22,200 times.
        """

        # Here goes your initialization logic

        pass

    def normalize(self, text: str) -> str:
        """Normalize a single line.

        Args:
            text: Raw utterance text, exactly as stored in the corpus metadata.

        Returns:
            The normalized text. Returning the input unchanged is valid and common —
            most lines need nothing done to them.

        Note:
            Do not strip the combining acute accent ``U+0301``. It looks like part of
            the letter and is easily lost to "unicode cleanup", but it marks explicit
            stress and becomes labelled data for stress placement in lab 3.

            Normalize to NFC. Strings in NFC and NFD render identically in a terminal
            and compare unequal.
        """

        # Here goes your normalisation logic

        return text
