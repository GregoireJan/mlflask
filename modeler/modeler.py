from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np

from scipy import stats
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.ensemble import RandomForestClassifier


class Modeler:
    def __init__(self):
        cancer = load_breast_cancer()
        data = np.c_[cancer.data, cancer.target]
        columns = np.append(cancer.feature_names, ["target"])
        self.df = pd.DataFrame(data, columns=columns)
        #
    def prepro(self):
        col_std = np.std(self.df[self.df['target']==0])/np.std(self.df[self.df['target']==1]) > 2
        dfstd = self.df[col_std.index[~col_std]]
        dfstd_pm = pd.Series(data=list(stats.ttest_ind(dfstd[self.df['target']==0],dfstd[self.df['target']==1],equal_var=True).pvalue),index=list(dfstd.columns))
        col_selec = list(dfstd_pm[dfstd_pm > 0.05].index)
        #
        robust_scaler = RobustScaler()
        X = self.df.drop(columns=col_selec + ['target'])
        Xtr = robust_scaler.fit_transform(X)
        y = self.df.loc[:, 'target']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(Xtr, y, test_size = 0.2, random_state = 42)
        #
    def fit(self):
        self.model = RandomForestClassifier().fit(self.X_train, self.y_train)
        #
    def predict(self, input):
        prediction = self.model.predict([input])
        return prediction[0]

# m = Modeler()
# m.prepro()
# m.fit()
# pred = m.predict([1.799e+01, 1.038e+01, 1.228e+02, 1.001e+03, 1.184e-01, 2.776e-01,
#        3.001e-01, 1.471e-01, 2.419e-01, 1.095e+00, 8.589e+00, 1.534e+02,
#        4.904e-02, 5.373e-02, 1.587e-02, 2.538e+01, 1.733e+01, 1.846e+02,
#        2.019e+03, 1.622e-01, 6.656e-01, 7.119e-01, 2.654e-01, 4.601e-01,
#        1.189e-01])
# print(pred)