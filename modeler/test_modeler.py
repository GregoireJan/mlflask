
from modeler import Modeler
import pytest
import os
import pandas as pd

basepath = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
modelpath = os.path.join(basepath, "models/RDFModel.joblib")

# Test if model exist after fitting
def test_modeler_model():
    Modeler().fit()
    assert os.path.isfile(modelpath) == True
    
# Test if outcome of prediction is O or 1
data = {"mean radius":{"0":14.42},"mean texture":{"0":19.77},"mean perimeter":{"0":94.48},"mean area":{"0":642.5},"mean smoothness":{"0":0.09752},"mean compactness":{"0":0.1141},"mean concavity":{"0":0.09388},"mean concave points":{"0":0.05839},"mean symmetry":{"0":0.1879},"mean fractal dimension":{"0":0.0639},"radius error":{"0":0.2895},"texture error":{"0":1.851},"perimeter error":{"0":2.376},"area error":{"0":26.85},"smoothness error":{"0":0.008005},"compactness error":{"0":0.02895},"concavity error":{"0":0.03321},"concave points error":{"0":0.01424},"symmetry error":{"0":0.01462},"fractal dimension error":{"0":0.004452},"worst radius":{"0":16.33},"worst texture":{"0":30.86},"worst perimeter":{"0":109.5},"worst area":{"0":826.4},"worst smoothness":{"0":0.1431},"worst compactness":{"0":0.3026},"worst concavity":{"0":0.3194},"worst concave points":{"0":0.1565},"worst symmetry":{"0":0.2718},"worst fractal dimension":{"0":0.09353}}
@pytest.fixture(scope='module')
def df():
    return pd.DataFrame.from_dict(data)
def test_modeler_output(df):
    assert Modeler().predict(df) in (0,1)