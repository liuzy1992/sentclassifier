#!/usr/bin/env python3

from sklearn.feature_extraction.text import TfidfVectorizer
from .tokenizer import *

class vectorizer:
    def __init__(self):
        self.vec = TfidfVectorizer(ngram_range=(1, 3), tokenizer=spacy_tokenizer, max_df=0.8)

