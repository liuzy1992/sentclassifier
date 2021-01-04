#!/usr/bin/env python3

import sys
import time
from pipeline import *

def print_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def main(infile, model_name, outdir, jobs):
    models = ['svm', 'rf', 'nb', 'knn', 'sgd']
    assert model_name in models, 'Bad option for model name, please choose from {}.'.format(str(models))
    print("## " + print_time() + ": Loading data from {}...".format(infile))
    df_train, df_test = preprocessing(infile)
    print()
    print("## " + print_time() + ": Data loaded.\nTraining set size: {}, Test set size: {}.".format(len(df_train), len(df_test)))
    print()

    print("## " + print_time() + ": Start training {} model...".format(model_name.upper()))
    clf = modeling(df_train, model_name, jobs)
    print()
    
    saving(clf, outdir)
    print('## ' + print_time() + ": Training done!\nModel has been saved in {}/model.m".format(outdir))
    print()

    print("## " + print_time() + ": Evalutating model on test set...")
    evaluation(df_test, clf)

main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))
