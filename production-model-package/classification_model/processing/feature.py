# import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer


class DictVect(BaseEstimator, TransformerMixin):
    def __init__(self, variables):

        # if not isinstance(variables, int):
        #     raise TypeError('invalid data type, var should be int')
        self.variables = variables

    def fit(self, X, y=None):
        # this helps fit the sklearn pipeline
        return self

    def transform(self, X):
        # we create a copy of our dataframe
        X = X.copy()

        # create a dictionary from your dataframe
        X_dict = X.to_dict(orient="records")

        # instantiate an object of the class dictvectorier
        dict_vect = DictVectorizer(sparse=False)

        # fit the model to the datafram, so that the model can learn from the datafrane
        dict_vect.fit(X_dict)

        # apply transformation to the dataframe
        X = dict_vect.transform(X_dict)

        # if X.shape[1] < 35:
        #     np.columnstack((X, np.zeros(shape=1337, dtype=int)))

        x_final = pd.DataFrame(data=X, columns=self.variables)

        return x_final


# this class is a transformer that computes both financial ratio and savings potential index
class VariableTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, variables):
        self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        for var in self.variables:
            if var == "amount":
                value1 = var
            if var == "price":
                value2 = var
            if var == "income":
                value3 = var
            if var == "expenses":
                value4 = var
            if var == "debt":
                value5 = var
            if var == "time":
                value6 = var

        # compute two additional features significant for our modeling
        X["fin_ratio"] = (X[value1] / X[value2]).round(2)
        X["sav_pot_index"] = (
            (X[value3] - X[value4] - (X[value5] / 100)) / (X[value1] / X[value6])
        ).round(2)

        return X
