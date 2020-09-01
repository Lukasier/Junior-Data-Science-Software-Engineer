import numpy as np
import pandas as pd
import sklearn
import pickle as pkl

# Prepare data for training
df = pd.read_csv("../data/train_bf.csv")

y = df["Survived"]
x = df.drop("Survived", axis=1)

# Create a classifier
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=10)

# Fit full model and predict on train
clf.fit(x, y)
preds = clf.predict(x)

# Select scoring metod
metric_result = sklearn.metrics.accuracy_score(y, preds)

# Pack model for test
pkl.dump(clf, open("../data/model.pkl", 'wb'))
model_pickle.close()

# Print accuracy of the model
info = "The train accuracy is " + str(metric_result)
print(info)