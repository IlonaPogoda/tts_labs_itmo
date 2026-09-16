### Preparing traning and testing data for pause predictor
import csv
import glob
import numpy as np
import os
import pandas as pd
from praatio import textgrid
import tqdm

RUSLAN_META = '../../data/metadata_RUSLAN_22200_normalized.csv'
ALIGN_DIR = '../../data/RUSLAN_align_v2/'
RESULT_PATH = 'data/RUSLAN_pause_metadata.csv'

def read_text_grids(ruslan, align_root):   
    word_docs = []
    phn_docs = []
    phoneme_sequences = []
    for wave_id, text in tqdm.tqdm(ruslan[['id', 'nrm']].values):
        try:
            tg = textgrid.openTextgrid(os.path.join(align_root, wave_id+'.TextGrid'), True)
        except:
            phoneme_sequences.append('')
            continue
        words, phones = tg.tiers

        words = words.entries
        for i in words:
            word_docs.append({'label':i.label, 'duration':i.end-i.start, 'id': wave_id})
        
        phoneme_sequences.append(' '.join([p.label for p in phones.entries]))
        phones = phones.entries
        for i in phones:
            if i.label=='':
                phn_docs.append({'label':'<SIL>', 'duration':i.end-i.start, 'id': wave_id})
            else:
                phn_docs.append({'label':i.label, 'duration':i.end-i.start, 'id': wave_id})
    word_df = pd.DataFrame(word_docs)
    phn_df = pd.DataFrame(phn_docs)
    return word_df, phn_df, phoneme_sequences


def align_text_and_textgrid(tokens, text):
    raw_tokens = []
    text = text.lower()
    previous_word = -1
    for t, d, i in tokens[['label', 'duration', 'id']].values:
        if t == '': # Empty, SIL token
            raw_tokens.append('<SIL>')
            continue
        splits = text.split(t, maxsplit=1)
        if len(splits)==1: # Does not found content!
            print(f'Error aligning f{i}!')
            print(t, text, tokens)
            return tokens 
        if previous_word == -1: 
            raw_tokens.append(splits[0].strip()+t)
        else:
            raw_tokens[previous_word] += splits[0].strip()
            raw_tokens.append(t)
        text = splits[1]
        previous_word = len(raw_tokens)-1
    if len(text) and (previous_word>=0):
        raw_tokens[previous_word] += text.strip()
    tokens['label_raw'] = raw_tokens
    return tokens

def add_pause_labels(align):
    is_last_word = []
    pause_after = []
    pause_duration = []
    last_word = -1
    for idx, (label, dur) in enumerate(align[['label', 'duration']].values):
        if label == '':
            if last_word>=0:
                pause_after[last_word] = True
                pause_duration[last_word] = dur
            pause_after.append(False)
            pause_duration.append(0.)
            is_last_word.append(False)
        else:
            pause_after.append(False)
            pause_duration.append(0.)
            is_last_word.append(False)
            last_word = idx
    if last_word>=0:
        is_last_word[last_word] = True
    align['is_last_word'] = is_last_word
    align['is_pause_after'] = pause_after
    align['pause_duration'] = pause_duration 
    return align
    

def main():
    ruslan = pd.read_csv(f'{RUSLAN_META}', sep='|', names=['id', 'raw', 'nrm'])
    word_df, _, _ = read_text_grids(ruslan, ALIGN_DIR)

    
    #Additional normalization to ensure good alignment
    word_df.label = word_df.label.str.replace('‐', '-') # differen hyphen symbols
    word_df.label = word_df.label.str.replace('‑', '-')
    ruslan.nrm = ruslan.nrm.str.replace('‐', '-')
    ruslan.nrm = ruslan.nrm.str.replace('‑', '-')
    ruslan.nrm = ruslan.nrm.str.replace('’', "'") # Mfa replaces ’ by '
    ruslan.nrm = ruslan.nrm.str.replace('\\((.*?)\\)', '[bracketed]', regex=True) # Mfa spells () and <> text as [bracketed]
    ruslan.nrm = ruslan.nrm.str.replace('\\<(.*?)\\>', '[bracketed]', regex=True)

    #Aligning TextGrid-normalised tokens and text, including punctuation
    aligns = []
    for n, i in tqdm.tqdm(ruslan[['nrm', 'id']].values):
        tokens = word_df[word_df.id==i]
        aligns.append(align_text_and_textgrid(tokens, n))

    # Creating labels for pause predictor training
    aligns = [add_pause_labels(a) for a in aligns]

    # Creating dataframe
    pause_df = pd.concat(aligns)
    
    pause_df = pause_df[pause_df.label_raw!='<SIL>'] # Removing pause tokens -- all information about pauses is in word tokens now
    pause_df = pause_df[pause_df.label_raw.notna()] # Removing all the files, who failed to be aligned
    pause_df = pause_df[((pause_df.pause_duration>0.1) | (~pause_df.is_pause_after))] # Removing all the pauses smaller than 100 ms
    pause_df = pause_df.reset_index(drop=True) # Reseting the index after filtering
    
    pause_df.is_last_word = pause_df.is_last_word.astype(int) # Casting bool fields into int
    pause_df.is_pause_after = pause_df.is_pause_after.astype(int) #


    # Splitting into train and test deterministically: All the files with index, ending with 0 or 5 is considered test
    pause_df['set'] = 'train'
    pause_df.loc[pause_df.id.str.split('_', expand=True)[0].astype(int)%5==0, 'set'] = 'test'
    pause_df.to_csv(f'{RESULT_PATH}', sep='|', index=False, header=True, quoting=csv.QUOTE_NONE)
    
if __name__=='__main__':
    main()