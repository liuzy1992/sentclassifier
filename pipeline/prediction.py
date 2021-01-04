#!/usr/bin/env python3

import sys
import joblib
import pandas as pd
import numpy as np
from .preprocessing import trim_string
from .vectorizer import vectorizer
from sklearn.metrics import classification_report

def prediction(infile, model_path, outfile):
    
    # preprocessing
    df = pd.read_csv(infile, 
                     sep='\t', 
                     header=0, 
                     quoting=3, 
                     engine='python', 
                     dtype={'pmid':str, 'sentid':str, 'senttext':str, 'label':str})
    
    df['label'] = (df['label'] == 'Y').astype('int')
    df = df.where(df.notnull(), '')

    df['senttext'] = df['senttext'].apply(trim_string)
    
    # load model
    clf = joblib.load(filename=model_path + "/model.m")

    # prediction
    y_pred = clf.predict(df['senttext'])
    y_prob = clf.predict_proba(df['senttext'])
    
    df['label_pred'] = y_pred
    df['prob'] = np.amax(y_prob, axis=1)

    # output
    df.to_csv(outfile, sep='\t', header=True, index=False)

    # classification report
    print("Classification report:")
    print(classification_report(df['label'], y_pred))
