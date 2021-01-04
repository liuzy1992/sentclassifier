#!/usr/bin/env python3

import string
import spacy
# from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS

punctuations = string.punctuation

nlp = spacy.load('en_core_web_sm')
stop_words = STOP_WORDS

# parser = English()

def spacy_tokenizer(sent):
    
    tokens = nlp(sent)

    tokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in tokens ]
    tokens = [ word for word in tokens if word not in stop_words and word not in punctuations ]

    return tokens
