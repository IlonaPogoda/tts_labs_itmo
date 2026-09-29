"""Pause predictor — skeleton for lab 2.

Run as a script to score the predictor on the prepared data::

    python pause_predictor.py

Precision, recall and F1 are computed for `is_pause_after`, and MAE for `pause_duration`
on true positives only. The last word of every utterance is excluded.
"""

import csv
import numpy as np
import tqdm
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import mean_absolute_error

import re
from pathlib import Path
from catboost import CatBoostClassifier, CatBoostRegressor
from catboost.utils import get_gpu_device_count
from sklearn.model_selection import GroupShuffleSplit

try:
    import pymorphy3
except ImportError as exc:
    raise ImportError(
        'Install morphology dependencies: pip install pymorphy3 pymorphy3-dicts-ru'
    ) from exc


LAB_DIR = Path(__file__).resolve().parent
PAUSE_PREDICTOR_DATA = LAB_DIR / 'data' / 'RUSLAN_pause_metadata.csv'
RANDOM_STATE = 42

STRONG_PUNCT = {'DOT', 'QUESTION', 'EXCLAMATION', 'SEMICOLON'}


def clean_word(token: str) -> str:
    token = str(token).lower().strip()
    return re.sub(
        r"^[^а-яёa-z0-9]+|[^а-яёa-z0-9-]+$",
        '',
        token,
        flags=re.IGNORECASE,
    )


def punct_type(token: str) -> str:
    token = str(token).strip()
    if re.search(r',\s*["»”\']?$', token):
        return 'COMMA'
    if re.search(r';\s*["»”\']?$', token):
        return 'SEMICOLON'
    if re.search(r':\s*["»”\']?$', token):
        return 'COLON'
    if re.search(r'\?\s*["»”\']?$', token):
        return 'QUESTION'
    if re.search(r'!\s*["»”\']?$', token):
        return 'EXCLAMATION'
    if re.search(r'…\s*["»”\']?$', token):
        return 'ELLIPSIS'
    if re.search(r'\.\s*["»”\']?$', token):
        return 'DOT'
    # Do not treat ordinary ASCII '-' as a dash: it also occurs inside words.
    if re.search(r'[—–]\s*["»”\']?$', token):
        return 'DASH'
    return 'NONE'


class PausePredictor():
    """Predicts where pauses fall in a sentence and how long they are.

    Input is one sentence as a sequence of `label_raw` tokens — words with their
    trailing punctuation, in order::

        ["Я", "вышел", "из", "дома,", "когда", "стемнело."]
    """

    def __init__(self, data_path=PAUSE_PREDICTOR_DATA):
        self.data_path = Path(data_path)
        self.morph = pymorphy3.MorphAnalyzer()
        self._morph_cache = {}
        self.threshold = 0.5
        self.duration_alpha = 1.0
        self.duration_median = 0.1
        self.feature_columns = None
        self.cat_features = None
        self.classifier = None
        self.regressor = None

        gpu_count = get_gpu_device_count()
        self.task_type = 'GPU' if gpu_count > 0 else 'CPU'
        print(f'CatBoost backend: {self.task_type}; GPU count={gpu_count}', flush=True)
        self._fit()

    def _morph_features(self, word: str) -> dict:
        word = clean_word(word)
        if word in self._morph_cache:
            return self._morph_cache[word]

        if not word:
            result = {
                'pos': 'NONE', 'case': 'NONE', 'number': 'NONE',
                'gender': 'NONE', 'tense': 'NONE', 'aspect': 'NONE',
            }
        else:
            parse = self.morph.parse(word)[0]
            tag = parse.tag
            result = {
                'pos': str(tag.POS or 'NONE'),
                'case': str(tag.case or 'NONE'),
                'number': str(tag.number or 'NONE'),
                'gender': str(tag.gender or 'NONE'),
                'tense': str(tag.tense or 'NONE'),
                'aspect': str(tag.aspect or 'NONE'),
            }

        self._morph_cache[word] = result
        return result

    def _features(self, tokens: list[str] | np.ndarray) -> pd.DataFrame:
        tokens = [str(t) for t in tokens]
        words = [clean_word(t) for t in tokens]
        puncts = [punct_type(t) for t in tokens]
        morphs = [self._morph_features(w) for w in words]
        n = len(tokens)
        rows = []

        for i in range(n):
            prev_i = i - 1
            next_i = i + 1
            prev_word = words[prev_i] if prev_i >= 0 else ''
            next_word = words[next_i] if next_i < n else ''
            prev_pos = morphs[prev_i]['pos'] if prev_i >= 0 else 'BOS'
            next_pos = morphs[next_i]['pos'] if next_i < n else 'EOS'
            prev_punct = puncts[prev_i] if prev_i >= 0 else 'BOS'
            next_punct = puncts[next_i] if next_i < n else 'EOS'
            relative_position = i / max(n - 1, 1)

            rows.append({
                'punct': puncts[i],
                'prev_punct': prev_punct,
                'next_punct': next_punct,
                'pos': morphs[i]['pos'],
                'prev_pos': prev_pos,
                'next_pos': next_pos,
                'pos_trigram': f'{prev_pos}|{morphs[i]["pos"]}|{next_pos}',
                'case': morphs[i]['case'],
                'number': morphs[i]['number'],
                'gender': morphs[i]['gender'],
                'tense': morphs[i]['tense'],
                'aspect': morphs[i]['aspect'],
                # Exact lexical identity is deliberately omitted to reduce overfitting.
                'word_len': len(words[i]),
                'prev_word_len': len(prev_word),
                'next_word_len': len(next_word),
                'starts_upper': int(bool(tokens[i]) and tokens[i][0].isupper()),
                'sentence_len': n,
                'token_index': i,
                'tokens_to_end': n - i - 1,
                'relative_position': relative_position,
                'near_start': int(relative_position <= 0.2),
                'near_end': int(relative_position >= 0.8),
            })

        return pd.DataFrame(rows)

    def _prepare_training_table(self, df: pd.DataFrame):
        parts = []
        y_pause = []
        y_duration = []
        groups = []

        for wave_id, sentence in tqdm.tqdm(
            df.groupby('id', sort=False),
            total=df.id.nunique(),
            desc='Preparing features',
        ):
            sentence = sentence.reset_index(drop=True)
            X_sentence = self._features(sentence.label_raw.values)
            keep = sentence.is_last_word.astype(int).values == 0
            parts.append(X_sentence.loc[keep].reset_index(drop=True))
            y_pause.extend(sentence.loc[keep, 'is_pause_after'].astype(int).tolist())
            y_duration.extend(sentence.loc[keep, 'pause_duration'].astype(float).tolist())
            groups.extend([wave_id] * int(keep.sum()))

        return (
            pd.concat(parts, ignore_index=True),
            np.asarray(y_pause, dtype=int),
            np.asarray(y_duration, dtype=float),
            np.asarray(groups),
        )

    def _common_catboost(self):
        params = {
            'random_seed': RANDOM_STATE,
            'allow_writing_files': False,
            'verbose': False,
        }
        if self.task_type == 'GPU':
            params.update(task_type='GPU', devices='0', gpu_ram_part=0.8)
        else:
            params.update(task_type='CPU', thread_count=-1)
        return params

    def _apply_rules(self, X: pd.DataFrame, pred: np.ndarray) -> np.ndarray:
        pred = np.asarray(pred, dtype=int).copy()
        strong = X.punct.isin(STRONG_PUNCT).to_numpy()
        pred[strong] = 1
        return pred

    def _fit(self):
        pause_df = pd.read_csv(
            self.data_path,
            sep='|',
            quoting=csv.QUOTE_NONE,
        )
        train_df = pause_df[pause_df.set == 'train'].copy()

        X, y, durations, groups = self._prepare_training_table(train_df)
        self.feature_columns = list(X.columns)
        self.cat_features = [
            'punct', 'prev_punct', 'next_punct',
            'pos', 'prev_pos', 'next_pos', 'pos_trigram',
            'case', 'number', 'gender', 'tense', 'aspect',
        ]

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=0.2,
            random_state=RANDOM_STATE,
        )
        fit_idx, val_idx = next(splitter.split(X, y, groups=groups))
        X_fit = X.iloc[fit_idx].reset_index(drop=True)
        X_val = X.iloc[val_idx].reset_index(drop=True)
        y_fit = y[fit_idx]
        y_val = y[val_idx]

        # Small validation search.  No test data is used here.
        best = None
        for positive_weight in [1.0, 1.25, 1.5, 1.75]:
            model = CatBoostClassifier(
                iterations=1200,
                depth=6,
                learning_rate=0.05,
                loss_function='Logloss',
                l2_leaf_reg=7.0,
                random_strength=0.5,
                class_weights=[1.0, positive_weight],
                **self._common_catboost(),
            )
            model.fit(
                X_fit,
                y_fit,
                cat_features=self.cat_features,
                eval_set=(X_val, y_val),
                early_stopping_rounds=100,
                verbose=False,
            )
            prob = model.predict_proba(X_val)[:, 1]
            iterations = model.get_best_iteration() + 1
            if iterations <= 0:
                iterations = 1200

            for threshold in np.arange(0.25, 0.701, 0.01):
                pred = self._apply_rules(X_val, (prob >= threshold).astype(int))
                score = f1_score(y_val, pred)
                candidate = (score, positive_weight, float(threshold), iterations)
                if best is None or candidate[0] > best[0]:
                    best = candidate

        val_f1, positive_weight, self.threshold, best_iterations = best
        print(
            f'Validation classifier: F1={val_f1:.4f}; '
            f'class_weight={positive_weight}; threshold={self.threshold:.2f}',
            flush=True,
        )

        self.classifier = CatBoostClassifier(
            iterations=best_iterations,
            depth=6,
            learning_rate=0.05,
            loss_function='Logloss',
            l2_leaf_reg=7.0,
            random_strength=0.5,
            class_weights=[1.0, positive_weight],
            **self._common_catboost(),
        )
        self.classifier.fit(X, y, cat_features=self.cat_features, verbose=False)

        # Duration is trained only on genuine pauses.
        d_fit = durations[fit_idx]
        d_val = durations[val_idx]
        pos_fit = y_fit == 1
        pos_val = y_val == 1
        X_dur_fit = X_fit.loc[pos_fit].reset_index(drop=True)
        X_dur_val = X_val.loc[pos_val].reset_index(drop=True)
        y_dur_fit = d_fit[pos_fit]
        y_dur_val = d_val[pos_val]
        self.duration_median = float(np.median(y_dur_fit))

        # Raw and log targets are compared by validation MAE.  Predictions may be
        # blended with the robust training median, again using validation only.
        best_duration = None
        for kind in ['raw', 'log']:
            model = CatBoostRegressor(
                iterations=1400,
                depth=6,
                learning_rate=0.03,
                loss_function='MAE',
                l2_leaf_reg=7.0,
                random_strength=0.5,
                **self._common_catboost(),
            )
            target_fit = np.log1p(y_dur_fit) if kind == 'log' else y_dur_fit
            target_val = np.log1p(y_dur_val) if kind == 'log' else y_dur_val
            model.fit(
                X_dur_fit,
                target_fit,
                cat_features=self.cat_features,
                eval_set=(X_dur_val, target_val),
                early_stopping_rounds=120,
                verbose=False,
            )
            pred = model.predict(X_dur_val)
            if kind == 'log':
                pred = np.expm1(pred)
            pred = np.clip(pred, 0.03, 1.5)
            iterations = model.get_best_iteration() + 1
            if iterations <= 0:
                iterations = 1400

            for alpha in np.arange(0.5, 1.001, 0.1):
                blended = alpha * pred + (1.0 - alpha) * self.duration_median
                mae = mean_absolute_error(y_dur_val, blended)
                candidate = (mae, kind, float(alpha), iterations)
                if best_duration is None or candidate[0] < best_duration[0]:
                    best_duration = candidate

        val_mae, duration_kind, self.duration_alpha, duration_iterations = best_duration
        self.duration_log_target = duration_kind == 'log'
        print(
            f'Validation duration: MAE={val_mae:.4f}; '
            f'target={duration_kind}; blend={self.duration_alpha:.1f}',
            flush=True,
        )

        positives = y == 1
        X_dur = X.loc[positives].reset_index(drop=True)
        y_dur = durations[positives]
        target = np.log1p(y_dur) if self.duration_log_target else y_dur

        self.regressor = CatBoostRegressor(
            iterations=duration_iterations,
            depth=6,
            learning_rate=0.03,
            loss_function='MAE',
            l2_leaf_reg=7.0,
            random_strength=0.5,
            **self._common_catboost(),
        )
        self.regressor.fit(X_dur, target, cat_features=self.cat_features, verbose=False)

    def predict(self, tokens: list[str] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Decide for each token whether a pause follows it.

        Args:
            tokens: Tokens of one sentence.

        Returns:
            `is_pause` (int, 1 if a pause follows the token) and `pause_duration`
            (float, seconds, 0.0 where there is no pause), both of length `len(tokens)`.

        Note:
            Wherever `is_pause` is 1 the duration must be positive:
            :meth:`predict_durations` relies on it.
        """
        tokens = np.asarray(tokens, dtype=object)
        if len(tokens) == 0:
            return np.zeros(0, int), np.zeros(0, float)

        X = self._features(tokens)[self.feature_columns]
        prob = self.classifier.predict_proba(X)[:, 1]
        is_pause = self._apply_rules(X, (prob >= self.threshold).astype(int))

        # The course task excludes the final pause.
        is_pause[-1] = 0

        pause_duration = np.zeros(len(tokens), float)
        pause_idx = np.flatnonzero(is_pause == 1)
        if len(pause_idx):
            pred = self.regressor.predict(X.iloc[pause_idx])
            if self.duration_log_target:
                pred = np.expm1(pred)
            pred = np.clip(pred, 0.03, 1.5)
            pred = (
                self.duration_alpha * pred
                + (1.0 - self.duration_alpha) * self.duration_median
            )
            pause_duration[pause_idx] = pred

        return is_pause, pause_duration

    def predict_durations(self, tokens: list[str] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Insert predicted pauses into the token sequence.

        This is the form the acoustic model consumes in labs 4 and 5.

        Args:
            tokens: Tokens of one sentence.

        Returns:
            Tokens with ``"<SIL>"`` after every predicted pause, and one duration per
            output token: seconds for ``"<SIL>"``, ``-1.0`` for words (left to the
            acoustic model).
        """
        def expand_is_pause(token: str, is_pause: int) -> list[str]:
            if bool(is_pause):
                return [token, '<SIL>']
            return [token]

        def expand_durations(pause_duration: float) -> list[float]:
            if pause_duration>0.:
                return [-1., pause_duration]
            return [-1.]
            
        is_pause, durations = self.predict(tokens)

        tokens_w_pauses = np.concatenate([expand_is_pause(a, b) for a, b in zip(tokens, is_pause)])
        durations_w_pauses = np.concatenate([expand_durations(dur) for dur in durations]).astype(np.float32)
        
        return tokens_w_pauses, durations_w_pauses

def calc_metrics(df: pd.DataFrame) -> dict:
    """Print precision, recall and F1 for pause placement, and MAE for pause duration.

    MAE counts only rows where both the reference and the prediction have a pause.
    """
    rec =recall_score(df.is_pause_after, df.is_pause_hat)
    prc = precision_score(df.is_pause_after, df.is_pause_hat)
    f1 = f1_score(df.is_pause_after, df.is_pause_hat)

    tp = (df.is_pause_after==1) & (df.is_pause_hat==1)
    mae = mean_absolute_error(df.loc[tp, 'pause_duration'], df.loc[tp, 'pause_duration_hat']) if tp.any() else np.nan
    print(f'PRC: {prc:.4f}, REC: {rec:.4f}, F1: {f1:.4f}; MAE: {mae:.4f};')
    return {'precision': prc, 'recall': rec, 'f1': f1, 'mae': mae}

def test_pause_predictor() -> None:
    """Run the predictor on every sentence and print train and test metrics.

    Expects the layout written by `prepare_training_data.py`: rows grouped by utterance
    in order, each utterance ending with its `is_last_word` row.
    """
    pause_df =pd.read_csv(PAUSE_PREDICTOR_DATA, sep='|', quoting=csv.QUOTE_NONE)

    pp = PausePredictor()

    lens = {i:l for i, l in pause_df.groupby('id').count().reset_index(drop=False)[['id', 'label']].values}

    is_pause_after_hat = []
    pause_duration_hat = []

    idx = 0
    for i, is_last in tqdm.tqdm(pause_df[['id', 'is_last_word']].values):
        if not is_last:
            continue
        sentence = pause_df.iloc[idx:idx+lens[i]]
        idx += lens[i]
    
        is_pause_hat, pause_dur_hat = pp.predict(sentence.label_raw.values)
        is_pause_after_hat += list(is_pause_hat)
        pause_duration_hat += list(pause_dur_hat)
    
    pause_df['is_pause_hat'] = is_pause_after_hat
    pause_df['pause_duration_hat'] = pause_duration_hat
    
    print('Calculate metrics, traning fold; Exclude last tokens in every sentence!')
    train_metrics = calc_metrics(pause_df[(pause_df.set=='train') & (pause_df.is_last_word==0)])

    print('\nCalculate metrics, testing fold; Exclude last tokens in every sentence!')
    test_metrics = calc_metrics(pause_df[(pause_df.set=='test') & (pause_df.is_last_word==0)])

    predictions_path = LAB_DIR / 'data' / 'RUSLAN_pause_predictions.csv'
    pause_df.to_csv(predictions_path, sep='|', index=False, quoting=csv.QUOTE_NONE)
    print(f'Predictions saved to {predictions_path}')
    return {'train': train_metrics, 'test': test_metrics}
    
if __name__=='__main__':
    test_pause_predictor()
