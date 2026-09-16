### Pause predictor class
import csv
import numpy as np
import tqdm
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import mean_absolute_error

PAUSE_PREDICTOR_DATA = 'data/RUSLAN_pause_metadata.csv'

class PausePredictor():

    def predict(self, tokens):
        """
           Gives back "is_pause_after" 0/1 decision and "pause_duration" for is_pause_after==1"
        """
        
        is_pause = np.zeros(len(tokens), int)
        pause_duration = np.zeros(len(tokens), float)

        return is_pause, pause_duration

    def predict_durations(self, tokens):
        def expand_is_pause(token, is_pause):
            if bool(is_pause):
                return [token, '<SIL>']
            return [token]

        def expand_durations(pause_duration):
            if pause_duration>0.:
                return [-1., pause_duration]
            return [-1.]
            
        is_pause, durations = self.predict(tokens)

        tokens_w_pauses = np.concatenate([expand_is_pause(a, b) for a, b in zip(tokens, is_pause)])
        durations_w_pauses = np.concatenate([expand_durations(dur) for dur in durations]).astype(np.float32)
        
        return tokens_w_pauses, durations_w_pauses

def calc_metrics(df):
    rec = recall_score(df.is_pause_after, df.is_pause_hat)
    prc = precision_score(df.is_pause_after, df.is_pause_hat)
    f1 = f1_score(df.is_pause_after, df.is_pause_hat)

    mae = mean_absolute_error(df[(df.is_pause_after==1) & (df.is_pause_hat==1)].pause_duration, df[(df.is_pause_after==1) & (df.is_pause_hat==1)].pause_duration_hat)
    print(f'PRC: {prc}, REC: {rec}, F1: {f1}; MAE: {mae};')

def test_pause_predictor():
    pause_df = pd.read_csv(PAUSE_PREDICTOR_DATA, sep='|', quoting=csv.QUOTE_NONE)

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
    calc_metrics(pause_df[(pause_df.set=='train') & (pause_df.is_last_word==0)])

    print('\nCalculate metrics, testing fold; Exclude last tokens in every sentence!')
    calc_metrics(pause_df[(pause_df.set=='test') & (pause_df.is_last_word==0)])
    
if __name__=='__main__':
    test_pause_predictor()