from sklearn.model_selection import StratifiedKFold, GroupKFold
import numpy as np



def train_validation_data_stratified(ids, labels, n_folds, seed, groups=None):

    if groups is None:
        for train, valid in StratifiedKFold(
            n_folds, shuffle=True, random_state=seed).split(ids, labels):
            yield train, valid
    else:
        for train, valid in GroupKFold(
            n_folds, shuffle=True, random_state=seed).split(ids, labels, groups):
            yield train, valid
