from sklearn.model_selection import StratifiedKFold
import numpy as np



def train_validation_data_stratified(ids, labels, n_folds, seed):

    for train, valid in StratifiedKFold(
        n_folds, shuffle=True, random_state=seed).split(ids, labels):
        yield train, valid