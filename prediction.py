#!/usr/bin/env python3

import sys
import joblib
import pandas as pd
import numpy as np
from pipeline.preprocessing import trim_string
from pipeline.vectorizer import vectorizer

def prediction(infile, model_path, outfile):
    
    # preprocessing
    df = pd.read_csv(infile, 
                     sep='\t', 
                     header=None, 
                     quoting=3, 
                     engine='python',
                     names=['pmid', 'sentid', 'gene', 'senttext'],
                     dtype={'pmid':str, 'sentid':str, 'gene':str, 'senttext':str})
    
    df = df.where(df.notnull(), '')

    df['senttext'] = df['senttext'].apply(trim_string)
    
    # load model
    clf = joblib.load(filename=model_path + "/model.m")

    # prediction
    y_pred = clf.predict(df['senttext'])
    y_prob = clf.predict_proba(df['senttext'])
    
    df['label_pred'] = y_pred
    df.loc[df['label_pred'] == 1, 'label_pred'] = 'Y'
    df.loc[df['label_pred'] == 0, 'label_pred'] = 'N'

    df['prob'] = np.amax(y_prob, axis=1)
    df['prob'] = df['prob'].round(decimals=4)

    # output
    df.to_csv(outfile, sep='\t', header=True, index=False)

prediction(sys.argv[1], sys.argv[2], sys.argv[3])
