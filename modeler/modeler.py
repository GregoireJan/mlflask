from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
import os.path

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import dump, load


basepath = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
modelpath = os.path.join(basepath, "models/RDFModel.joblib")

class Modeler:
    def __init__(self):
        # Loading breast cancer data from sklearn
        cancer = load_breast_cancer()
        data = np.c_[cancer.data, cancer.target]
        columns = np.append(cancer.feature_names, ["target"])
        self.df_base = pd.DataFrame(data, columns=columns)
        #
    def prepro(self,df,fit='yes'):
        # Preprocessing: removing columns & scaling
        col_selec = ['mean fractal dimension', 'texture error', 'smoothness error', 'symmetry error', 'fractal dimension error']
        robust_scaler = RobustScaler()
        if fit == 'yes':
            X = df.drop(columns=col_selec + ['target'])
            Xtr = robust_scaler.fit_transform(X)
            y = df.loc[:, 'target']
            return Xtr, y
        else:
            X = df.drop(columns=col_selec)
            Xtr = robust_scaler.fit_transform(X)
            return Xtr
        #
    def fit(self):
        Xtr, y =  Modeler().prepro(self.df_base)
        # Splitting between train/test subsets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(Xtr, y, test_size = 0.2, random_state = 42)
        # Fitting Random Forest to train data
        self.model = RandomForestClassifier().fit(self.X_train, self.y_train)
        accuracy = accuracy_score(self.y_test, self.model.predict(self.X_test))
        if accuracy > 0.90:
            dump(self.model, modelpath)
        else:
            raise Exception("The accuracy is metric is below the 90% threshold: ",accuracy)
        #
    def predict(self, df_pred):
        # Predicting cancer outcome using the fitted model
        Xtr =  Modeler().prepro(df_pred,'no')
        rdf = load(modelpath)
        prediction = rdf.predict(Xtr)
        return prediction[0]

# m = Modeler()

# # if not os.path.isfile('models/RDFModel.joblib'):
# # m.prepro()
# m.fit()
# pred = m.predict(309)
# print(pred)