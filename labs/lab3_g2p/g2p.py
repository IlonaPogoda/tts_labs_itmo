"""Grapheme-to-phoneme front end — skeleton for lab 3.

Run as a script to score the module on the prepared data::

    python g2p.py

Two phone error rates are reported on the split carried in the data's own `set` column: PER over
whole `test` utterances — the front end as lab 5 will run it — and PER over `test` tokens
absent from the training lexicon, which is the out-of-vocabulary model alone.
"""
import csv
import math
import argparse
from difflib import SequenceMatcher
import unicodedata
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import tqdm

TOKENS_PHONES_DATA = 'data/tokens_phones.csv'


SIL = '<SIL>'


def load_mfa_dictionary(path='data/russian_mfa.dict', *, all_variants=False):
    """Load MFA pronunciations in either supported dictionary format."""
    entries = defaultdict(list)
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            word = unicodedata.normalize('NFC', parts[0].lower())
            phones_start = 1
            prior = 1.0
            if len(parts) >= 6:
                try:
                    numbers = [float(x) for x in parts[1:5]]
                except ValueError:
                    pass
                else:
                    prior = numbers[0]
                    phones_start = 5
            phones = tuple(parts[phones_start:])
            if not phones:
                continue
            if 'spn' in phones or '<SIL>' in phones:
                continue
            if all(pron != phones for pron, _ in entries[word]):
                entries[word].append(
                    (phones, max(prior, 1e-10))
                )
    if all_variants:
        return dict(entries)
    return {
        word: list(variants[0][0])
        for word, variants in entries.items()
    }


class PhoneLanguageModel:
    """Train-only smoothed phone bigram model for scoring MFA variants."""

    def __init__(self, lexicon: dict[str, Counter]):
        self.uni = Counter()
        self.bi = Counter()
        self.vocabulary = set()
        for pronunciations in lexicon.values():
            for transcription, count in pronunciations.items():
                phones = transcription.split()
                if not phones:
                    continue
                seq = ['<BOS>', *phones, '<EOS>']
                self.vocabulary.update(phones)
                for left, right in zip(seq[:-1], seq[1:]):
                    self.uni[left] += count
                    self.bi[(left, right)] += count
        self.size = len(self.vocabulary) + 2

    def score(self, phones) -> float:
        """Average log-probability of phone transitions (higher is better)."""
        seq = ['<BOS>', *phones, '<EOS>']
        score = 0.0
        for left, right in zip(seq[:-1], seq[1:]):
            score += math.log((self.bi[left, right] + 0.1)
                              / (self.uni[left] + 0.1 * self.size))
        return score / max(1, len(seq) - 1)
LETTER_TO_PHONE = {
    'а': 'a',
    'б': 'b',
    'в': 'v',
    'г': 'ɡ',
    'д': 'd̪',
    'е': 'e',
    'ё': 'o',
    'ж': 'ʐ',
    'з': 'z̪',
    'и': 'i',
    'й': 'j',
    'к': 'k',
    'л': 'ɫ',
    'м': 'm',
    'н': 'n̪',
    'о': 'o',
    'п': 'p',
    'р': 'r',
    'с': 's̪',
    'т': 't̪',
    'у': 'u',
    'ф': 'f',
    'х': 'x',
    'ц': 't̪s̪',
    'ч': 'tɕ',
    'ш': 'ʂ',
    'щ': 'ɕ',
    'ы': 'ɨ',
    'э': 'ɛ',
    'ю': 'u',
    'я': 'a',
}

SOFT_CONSONANTS = {
    'б': 'bʲ',
    'в': 'vʲ',
    'д': 'dʲ',
    'з': 'zʲ',
    'к': 'c',
    'л': 'ʎ',
    'м': 'mʲ',
    'н': 'ɲ',
    'п': 'pʲ',
    'р': 'rʲ',
    'с': 'sʲ',
    'т': 'tʲ',
    'ф': 'fʲ',
    'г': 'ɟ',
    'х': 'ç',
}

FINAL_DEVOICING = {
    'b': 'p',
    'v': 'f',
    'ɡ': 'k',
    'd̪': 't̪',
    'z̪': 's̪',
    'ʐ': 'ʂ',
    'bʲ': 'pʲ',
    'vʲ': 'fʲ',
    'dʲ': 'tʲ',
    'zʲ': 'sʲ',
    'ɟ': 'c',
}

VOICELESS_PHONES = {
    'p', 'pʲ', 'f', 'fʲ', 't̪', 'tʲ', 's̪', 'sʲ',
    'k', 'c', 'ʂ', 'ɕ', 'tɕ', 't̪s̪', 'x', 'ç', 'tʂ',
}

CONSONANT_DEVOICING = {
    'b': 'p', 'bʲ': 'pʲ',
    'v': 'f', 'vʲ': 'fʲ',
    'ɡ': 'k', 'ɟ': 'c',
    'd̪': 't̪', 'dʲ': 'tʲ',
    'z̪': 's̪', 'zʲ': 'sʲ',
    'ʐ': 'ʂ', 'dʐ': 'tʂ',
}

RUSSIAN_VOWELS = set('аеёиоуыэюя')

VOWEL_PHONES = {'a', 'o', 'e', 'i', 'u', 'ɨ', 'ɛ', 'ɐ', 'ə', 'ɪ', 'ʊ', 'ʉ'}

REDUCED_VOWELS = {'ɐ', 'ə', 'ɪ', 'ʊ'}

DEFAULT_RULES = {
    'affricates': True,
    'sibilant_clusters': True,
    'silent_consonants': True,
    'double_consonants': True,
    'vowel_context': True,
    'devoicing': True,
    'endings': True,
    'mfa_variant_rerank': False,
}

SILENT_CLUSTER_RULES = (
    ('стн', 'сн'),
    ('здн', 'зн'),
    ('ндц', 'нц'),
    ('рдц', 'рц'),
    ('лнц', 'нц'),
)

SIBILANT_CLUSTER_RULES = (
    ('сч', 'щ'),
    ('зч', 'щ'),
    ('жч', 'щ'),
)


class G2P:
    """Turns a sentence into a phone sequence.

    Input is one sentence as the token sequence lab 2 produces: words without punctuation,
    with ``"<SIL>"`` wherever the pause predictor placed a pause::

        ["я", "вышел", "из", "дома", <SIL>, "когда", "стемнело"]

    Punctuation is gone by this point on purpose — its only role in the pipeline is to
    predict pauses, which lab 2 has already done.


    Args:
        lexicon: `word -> Counter({phones: count})` as written by `prepare_lexicon.py`.
            Pronunciations are space-separated NFC strings. Keys are lowercase and carry no
            stress mark, so look words up accordingly.
    """

    def __init__(self, lexicon: dict[str, Counter] | None = None,
                 rules: dict[str, bool] | None = None) -> None:
        self.rules = {**DEFAULT_RULES, **(rules or {})}
        self.lexicon = lexicon if lexicon is not None else {}
        if self.rules['mfa_variant_rerank']:
            self.mfa_variants = load_mfa_dictionary(all_variants=True)
            self.mfa_dict = {w: list(v[0][0]) for w, v in self.mfa_variants.items()}
            self.phone_lm = PhoneLanguageModel(self.lexicon)
        else:
            self.mfa_dict = load_mfa_dictionary()
            self.mfa_variants = None
            self.phone_lm = None
        self._mfa_cache = {}
        self.stress_examples: dict[str, int] = {}
        self.stress_index: dict[str, list[str]] = defaultdict(list)
        for word, pronunciations in self.lexicon.items():
            phones = pronunciations.most_common(1)[0][0].split()
            stress = self.infer_stress_from_phones(word, phones)
            if stress is not None:
                self.stress_examples[word] = stress
                self.stress_index[word[:4]].append(word)

    def normalize(self, token: str) -> str:
        return unicodedata.normalize('NFC', token.lower().strip())

    def infer_stress_from_phones(self, word: str, phones: list[str]) -> int | None:
        """Оценка индекса ударного слога по редуцированным гласным RUSLAN."""
        letters = [c for c in word if c in RUSSIAN_VOWELS]
        sounds = [p for p in phones if p in VOWEL_PHONES]
        if not letters or len(letters) != len(sounds):
            return None
        if 'ё' in letters:
            return letters.index('ё')
        candidates = [i for i, p in enumerate(sounds) if p not in REDUCED_VOWELS]
        return candidates[0] if len(candidates) == 1 else None

    def predict_stress(self, word: str) -> int | None:
        """Перенос индекса ударного слога с похожей TRAIN-словоформы."""
        letters = [c for c in word if c in RUSSIAN_VOWELS]
        if 'ё' in letters:
            return letters.index('ё')
        if len(word) < 5:
            return None
        best_score, best_stress = 0.0, None
        for candidate in self.stress_index.get(word[:4], []):
            if sum(c in RUSSIAN_VOWELS for c in candidate) != len(letters):
                continue
            score = SequenceMatcher(None, word, candidate).ratio()
            if score > best_score:
                best_score, best_stress = score, self.stress_examples[candidate]
        return best_stress if best_score >= 0.80 else None

    def normalize_oov_endings(self, word: str) -> str:
        """Приближённая обработка окончаний -ого/-его."""
        exceptions = {
            'много', 'немного',
            'строго', 'нестрого',
            'итого', 'ого',
        }
        if word in exceptions:
            return word
        adjective_endings = (
            'ового',
            'евого',
            'анного',
            'енного',
            'онного',
            'ированного',
            'ционного',
            'ского',
            'цкого',
            'шего',
        )
        if word.endswith(adjective_endings) and word.endswith(('ого', 'его')):
            return word[:-2] + 'в' + word[-1]
        return word

    def prepare_oov_spelling(self, word: str) -> str:
        """Conservative context-dependent orthographic transformations."""
        word = self.normalize(word)
        if self.rules['endings']:
            word = self.normalize_oov_endings(word)
        if self.rules['silent_consonants']:
            for old, new in SILENT_CLUSTER_RULES:
                word = word.replace(old, new)
        if self.rules['sibilant_clusters']:
            for old, new in SIBILANT_CLUSTER_RULES:
                word = word.replace(old, new)
        return word

    def devoice_consonants(self, phones: list[str]) -> list[str]:
        """Right-to-left regressive devoicing, then word-final devoicing."""
        result = phones.copy()
        for i in range(len(result) - 2, -1, -1):
            if (result[i] in CONSONANT_DEVOICING
                    and result[i + 1] in VOICELESS_PHONES):
                result[i] = CONSONANT_DEVOICING[result[i]]
        if result and result[-1] in FINAL_DEVOICING:
            result[-1] = FINAL_DEVOICING[result[-1]]
        return result

    def transcribe_oov(self, word: str) -> list[str]:
        """Rule-based residual OOV transcription using train-derived stress."""
        original_word = self.normalize(word)
        word = self.prepare_oov_spelling(original_word)
        phones: list[str] = []
        soft_vowels = {'е', 'ё', 'и', 'ю', 'я'}
        iotated = {'е': 'e', 'ё': 'o', 'ю': 'u', 'я': 'a'}
        vowel_positions = [i for i, c in enumerate(word) if c in RUSSIAN_VOWELS]
        vowel_to_syllable = {pos: n for n, pos in enumerate(vowel_positions)}
        predicted_stress = self.predict_stress(original_word)
        if predicted_stress is not None and 0 <= predicted_stress < len(vowel_positions):
            stressed_syllable = predicted_stress
        elif vowel_positions:
            stressed_syllable = max(0, len(vowel_positions) - 2)
        else:
            stressed_syllable = -1
        def vowel_phone(char: str, pos: int, base: str, preceding: str) -> str:
            syllable = vowel_to_syllable[pos]
            stressed = syllable == stressed_syllable
            if stressed:
                if self.rules['vowel_context'] and char == 'и' and preceding in 'жшц':
                    return 'ɨ'
                return base
            if char in {'о', 'а'}:
                if syllable == stressed_syllable - 1:
                    return 'ɐ'
                if pos == 0 and syllable < stressed_syllable:
                    return 'ɐ'
                return 'ə'
            if char in {'е', 'я', 'и'}:
                if self.rules['vowel_context'] and char == 'и' and preceding in 'жшц':
                    return 'ɨ'
                return 'ɪ'
            if char == 'у':
                return 'ʊ'
            return base
        i = 0
        while i < len(word):
            char = word[i]
            prev_char = word[i - 1] if i else ''
            next_char = word[i + 1] if i + 1 < len(word) else ''
            if char in {'ь', 'ъ'}:
                i += 1
                continue
            if self.rules['affricates'] and word.startswith('дж', i):
                phones.append('dʐ')
                i += 2
                continue
            if (self.rules['double_consonants'] and i > 0
                    and char == prev_char and char not in RUSSIAN_VOWELS):
                i += 1
                continue
            if char in iotated and (
                i == 0 or prev_char in RUSSIAN_VOWELS or prev_char in {'ь', 'ъ'}
            ):
                phones.extend(['j', vowel_phone(char, i, iotated[char], prev_char)])
                i += 1
                continue
            if char in SOFT_CONSONANTS and (next_char in soft_vowels or next_char == 'ь'):
                phones.append(SOFT_CONSONANTS[char])
                i += 1
                continue
            if char in LETTER_TO_PHONE:
                base = LETTER_TO_PHONE[char]
                phones.append(vowel_phone(char, i, base, prev_char)
                              if char in RUSSIAN_VOWELS else base)
            i += 1
        return self.devoice_consonants(phones) if self.rules['devoicing'] else phones

    def transcribe_mfa(self, word: str) -> list[str]:
        """Return MFA reading; optional train-only phonotactic alternative ranking."""
        if not self.rules['mfa_variant_rerank']:
            return self.mfa_dict[word]
        if word in self._mfa_cache:
            return self._mfa_cache[word]
        variants = self.mfa_variants[word]
        if len(variants) == 1:
            answer = list(variants[0][0])
        else:
            def rank(item):
                phones, prior = item
                return self.phone_lm.score(phones) + 0.08 * math.log(prior)
            answer = list(max(variants, key=rank)[0])
        self._mfa_cache[word] = answer
        return answer

    def transcribe(self, tokens: list[str] | np.ndarray) -> list[list[str]]:
        """Transcribe each token of one sentence.

        Args:
            tokens: Tokens of one sentence in order, possibly including ``"<SIL>"``.

        Returns:
            One phone list per input token, so that `len(result) == len(tokens)`. A token
            may map to an empty list, but the outer length must be preserved: the
            evaluation scores in-vocabulary, dictionary and residual tokens separately and
            needs to know which output belongs to which token. ``"<SIL>"`` maps to
            ``["<SIL>"]``.

        """
        result = []
        for token in tokens:
            if token == SIL:
                result.append([SIL])
                continue
            word = self.normalize(str(token))
            if word in self.lexicon:
                phones = self.lexicon[word].most_common(1)[0][0]
                result.append(phones.split())
            elif word in self.mfa_dict:
                result.append(self.transcribe_mfa(word))
            else:
                result.append(self.transcribe_oov(word))
        return result


def phone_error_rate(reference: list[list[str]], hypothesis: list[list[str]]) -> float:
    """Corpus-level PER: summed edit distance over summed reference length.

    Summing before dividing weights every phone equally. Averaging per-sentence rates
    would instead weight every sentence equally and let short sentences dominate.

    Args:
        reference: Reference phone sequences.
        hypothesis: Predicted phone sequences, same length and order.

    Returns:
        Phone error rate, or `nan` when the reference is empty.
    """
    total_distance = 0
    total_length = 0
    for ref, hyp in zip(reference, hypothesis):
        total_length += len(ref)
        if not hyp:
            total_distance += len(ref)
            continue
        previous = list(range(len(hyp) + 1))
        for i, ref_phone in enumerate(ref, start=1):
            current = [i] + [0] * len(hyp)
            for j, hyp_phone in enumerate(hyp, start=1):
                current[j] = min(previous[j] + 1,
                                 current[j - 1] + 1,
                                 previous[j - 1] + (ref_phone != hyp_phone))
            previous = current
        total_distance += previous[-1]
    return total_distance / total_length if total_length else float('nan')


def word_accuracy(reference: list[list[str]], hypothesis: list[list[str]]) -> float:
    """Fraction of words whose whole phone sequence is exactly right."""
    if not reference:
        return float('nan')
    return float(np.mean([ref == hyp for ref, hyp in zip(reference, hypothesis)]))


def load_tokens(path: str = TOKENS_PHONES_DATA) -> pd.DataFrame:
    """Read the training table and drop tokens with no reference transcription.

    Rows whose phones are empty or contain `spn` carry no reference and are removed.

    The `set` column comes with the file. It is a rule on the utterance id — test iff the
    number divides by 5

    Args:
        path: Path to the table written by `prepare_lexicon.py`.

    Returns:
        The table with `token` and `phones` as NFC strings, in sentence order.
    """
    tokens = pd.read_csv(path, sep='|', quoting=csv.QUOTE_NONE)
    tokens['token'] = tokens.token.astype(str).map(lambda s: unicodedata.normalize('NFC', s))
    tokens['phones'] = tokens.phones.fillna('').astype(str)

    unpronounced = (tokens.phones == '') | tokens.phones.str.split().map(lambda p: 'spn' in p)
    print(f'dropping {unpronounced.sum()} tokens without a transcription '
          f'({unpronounced.mean():.2%})')
    return tokens[~unpronounced].reset_index(drop=True)


def build_lexicon(tokens: pd.DataFrame) -> dict[str, Counter]:
    """Count pronunciations per token type over the given rows.
    """
    lexicon: dict[str, Counter] = defaultdict(Counter)
    for token, phones in tokens[['token', 'phones']].values:
        lexicon[token][phones] += 1
    return lexicon


def calc_metrics(g2p: G2P, tokens: pd.DataFrame, lexicon: dict[str, Counter]) -> None:
    """Print running-text PER, and PER and word accuracy on out-of-vocabulary tokens.

    Args:
        g2p: The module under test.
        tokens: One fold of the table from :func:`load_tokens`, in sentence order.
        lexicon: The training lexicon, used only to tell in-vocabulary tokens from OOV.
    """
    sentence_ref: list[list[str]] = []
    sentence_hyp: list[list[str]] = []
    oov_ref: list[list[str]] = []
    oov_hyp: list[list[str]] = []
    in_vocab_ref: list[list[str]] = []
    in_vocab_hyp: list[list[str]] = []

    for _, sentence in tqdm.tqdm(tokens.groupby('id', sort=False)):
        sentence = sentence.sort_values('token_id')
        words = sentence.token.values
        predicted = g2p.transcribe(words)
        if len(predicted) != len(words):
            raise ValueError(f'transcribe returned {len(predicted)} outputs '
                             f'for {len(words)} tokens')
        reference = [p.split() for p in sentence.phones.values]

        # <SIL> is stripped before scoring on both sides. Leaving it in would grade lab 2's
        # pause placement inside lab 3's metric. It is still passed through at runtime --
        # see :meth:`G2P.transcribe`.
        sentence_ref.append([p for word in reference for p in word if p != SIL])
        sentence_hyp.append([p for word in predicted for p in word if p != SIL])
        for token, ref, hyp in zip(words, reference, predicted):
            if token in lexicon:
                in_vocab_ref.append(ref)
                in_vocab_hyp.append(hyp)
            else:
                oov_ref.append(ref)
                oov_hyp.append(hyp)

    n_words = len(in_vocab_ref) + len(oov_ref)
    print(f'running text  PER {phone_error_rate(sentence_ref, sentence_hyp):.4f}')
    print(f'in-vocabulary PER {phone_error_rate(in_vocab_ref, in_vocab_hyp):.4f}  '
          f'({len(in_vocab_ref)} words, {len(in_vocab_ref) / n_words:.1%})')
    print(f'OOV           PER {phone_error_rate(oov_ref, oov_hyp):.4f}  '
          f'word accuracy {word_accuracy(oov_ref, oov_hyp):.4f}  '
          f'({len(oov_ref)} words, {len(oov_ref) / n_words:.1%})')


def test_g2p() -> None:
    parser = argparse.ArgumentParser(description='Evaluate Russian G2P')
    parser.add_argument('--mfa-rerank', action='store_true',
                        help='Rank MFA pronunciation alternatives using TRAIN phone bigrams')
    parser.add_argument('--validation', action='store_true',
                        help='Run clean TRAIN/VALID split by utterance ID (not test)')
    args = parser.parse_args()
    tokens = load_tokens()
    train = tokens[tokens.set == 'train'].copy()
    test = tokens[tokens.set == 'test'].copy()
    if args.validation:
        ids = sorted(train.id.unique())
        valid_ids = set(ids[::8])
        validation = train[train.id.isin(valid_ids)]
        train = train[~train.id.isin(valid_ids)]
        target = validation
        target_name = 'validation'
    else:
        target = test
        target_name = 'test'
    print(f'train {len(train)} tokens / {target_name} {len(target)} tokens')
    lexicon = build_lexicon(train)
    print(f'lexicon: {len(lexicon)} types, '
          f'{sum(1 for c in lexicon.values() if len(c) > 1)} with several pronunciations')
    rules = {'mfa_variant_rerank': args.mfa_rerank}
    g2p = G2P(lexicon, rules=rules)
    print(f'\nCalculate metrics, {target_name} fold:')
    calc_metrics(g2p, target, lexicon)
    residual = target[
        ~target.token.isin(lexicon)
        & ~target.token.isin(g2p.mfa_dict)
    ]
    residual_ref = [p.split() for p in residual.phones]
    residual_hyp = [g2p.transcribe([w])[0] for w in residual.token]
    print(f'Residual OOV: {len(residual)} tokens')
    print(f'Residual OOV PER {phone_error_rate(residual_ref, residual_hyp):.4f}')
    print(f'Residual OOV word accuracy {word_accuracy(residual_ref, residual_hyp):.4f}')
    mfa_only = target[
        ~target.token.isin(lexicon)
        & target.token.isin(g2p.mfa_dict)
    ]
    mfa_ref = [p.split() for p in mfa_only.phones]
    mfa_hyp = [g2p.transcribe([w])[0] for w in mfa_only.token]
    print(f'MFA-only: {len(mfa_only)} tokens')
    print(f'MFA-only PER {phone_error_rate(mfa_ref, mfa_hyp):.4f}')
    print(f'MFA-only word accuracy {word_accuracy(mfa_ref, mfa_hyp):.4f}')
    if args.mfa_rerank:
        mult = sum(len(v) > 1 for v in g2p.mfa_variants.values())
        print(f'MFA dictionary entries with multiple transcriptions: {mult}')


if __name__ == '__main__':
    test_g2p()
