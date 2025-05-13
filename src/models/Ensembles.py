import copy
import numpy as np

'''
Class for defining an ensemble of predictors
'''
class DeepEnsemble:
    def __init__(self, create_predictor, n_predictors) -> None:
        self.predictors = [create_predictor(i) for i in range(n_predictors)]
        self.n_predictors = n_predictors

    def fit(self, x, y):
        for i, predictor in enumerate(self.predictors):
            predictor.fit(x, y)
            print(f'Predictor {i+1} trained')
        
        return self

    def load(self, reinit_func, name_generator):
        for i, predictor in enumerate(self.predictors):
            predictor = reinit_func(name_generator(i), predictor)
        
        print(f'{len(self.predictors)} Predictor(s) loaded')
        return self

    '''
    Forward pass for the ensemble of predictors
    Parameters:
        x: input tensor B x D 
    '''
    def predict(self, x, std=False, all=False):
        out = [np.expand_dims(predictor.predict(x), axis=1) for predictor in self.predictors]
        out = np.concatenate(out, axis=1)
        if std:
            return out.mean(axis=1), out.std(axis=1)
        elif all:
            return out
        else:
            return out.mean(axis=1)