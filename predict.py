import pandas as pd
import numpy as np
import pickle

data = pd.read_csv('./flightdelayweather_ds_clean.csv')
data = data.drop(columns=['ArrDelay15'])

# Load the saved model
with open("model/model.pkl", 'rb') as file:
    clf = pickle.load(file)
 
# Now you can use the model
print(clf.predict(data.iloc[:100]))