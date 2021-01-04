#!/usr/bin/env python3

import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from .vectorizer import vectorizer

def model_selection(model_name):
    models = {
        'svm':{
            'model':SVC(probability=True, class_weight='balanced'),
            'param':{
                'clf__kernel':['linear', 'rbf'],
                # 'clf__C':np.logspace(-1, 3, 5),
                # 'clf__gamma':[0.01, 0.1, 1, 10, 100],
                }
            },
        'rf':{
            'model':RandomForestClassifier(oob_score=True, random_state=3),
            'param':{
                'clf__n_estimators':[50, 100, 150, 200],
                'clf__max_depth':[None, 1, 3, 5, 7, 9, 11],
                'clf__min_samples_leaf':[1, 3, 5, 7],
                'clf__max_features':['auto', None, 3, 5, 7, 9],
                }
            },
        'nb':{
            'model':MultinomialNB(),
            'param':{
                # 'clf__alpha':[0.01, 0.1, 0, 1, 10],
                'clf__alpha':[1.0],
                }
            },
        'knn':{
            'model':KNeighborsClassifier(),
            'param':{
                'clf__weights':['uniform', 'distance'],
                'clf__n_neighbors':[24, 25, 26, 27, 28, 29, 30],
                # 'clf__n_neighbors':[3, 6, 9, 12, 15, 18, 21, 24, 27, 30],
                # 'clf__leaf_size':[10, 20, 30, 40, 50],

                }
            },
        'sgd':{
            'model':SGDClassifier(penalty='l2',loss='hinge'),
            'param':{
                'clf__alpha':[0.0001, 0.001, 0.01, 0.1],
                }
            },
        }

    return models.get(model_name)

def modeling(df_train, model_name, jobs):
    
    model_info = model_selection(model_name)

    pipe = Pipeline([('vectorizer', vectorizer().vec), ('clf', model_info['model'])])

    gs = GridSearchCV(pipe, model_info['param'], cv=10, scoring='average_precision', n_jobs=jobs)

    gs.fit(df_train['senttext'], df_train['label'])

    print("Best parameters: " + str(gs.best_params_))
    print("Best score on training set: {:.6f}".format(gs.best_score_.item()))

    return gs

