#!/usr/bin/env python3

import joblib

def saving(model, outdir):
    joblib.dump(value = model, filename = outdir + '/model.m')
