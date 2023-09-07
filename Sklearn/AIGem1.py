from sklearn import model_selection
#from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import pickle

'''
data = pd.read_csv("Gem_prof.csv")
X = data.drop(columns=["result"])
print(X)
#X = pd.get_dummies(X)
y = data["result"]
X_train, X_test, y_train, y_test = train_test_split(X, y)
clr = RandomForestClassifier()
clr.fit(X_train, y_train)

result = clr.score(X, y)
#predictions = clr.predict(1.10121719 )
#predictions = clr.predict([ [0.10121719], [0.151719], [0.181511] ])
print(result)
#print(predictions)
'''

filename = 'Gem_model4.sav'
loaded_model = pickle.load(open(filename, 'rb'))



initial_type = [('float_input', FloatTensorType([None, 1]))]
onx = convert_sklearn(loaded_model, initial_types=initial_type)
with open("Gem4.onnx", "wb") as f:
    f.write(onx.SerializeToString())




