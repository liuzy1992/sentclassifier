#!/usr/bin/env python3

import pandas as pd
from sklearn.model_selection import train_test_split

def trim_string(s):
    l = s.strip().split()
    ns = ' '.join(l)

    return ns

def preprocessing(infile):
    df_raw = pd.read_csv(infile, sep='\t', header=0, quoting=3, engine='python', dtype={'pmid':str, 'sentid':str, 'senttext':str, 'label':str})

    df_raw['label'] = (df_raw['label'] == 'Y').astype('int')
    df_raw = df_raw.where(df_raw.notnull(), '')

    df_raw = df_raw.reindex(columns=['label', 'senttext'])

    df_raw['senttext'] = df_raw['senttext'].apply(trim_string)

    df_neg = df_raw[df_raw['label'] == 0]
    df_pos = df_raw[df_raw['label'] == 1]

    df_pos_train, df_pos_test = train_test_split(df_pos, train_size=0.8, random_state=2)
    df_neg_train, df_neg_test = train_test_split(df_neg, train_size=0.8, random_state=2)

    df_train = pd.concat([df_pos_train, df_neg_train], ignore_index=True, sort=False)
    df_test = pd.concat([df_pos_test, df_neg_test], ignore_index=True, sort=False)

    return df_train, df_test
