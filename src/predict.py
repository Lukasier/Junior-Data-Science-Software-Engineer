import pandas as pd
import pickle as pkl

df = pd.read_csv("../data/val_bf.csv")

# Set target for model and remove it from test set
target = df["Survived"]
del(df["Survived"])

# Unpack pre-created model
model_unpickle = open("data/model.pkl", 'rb')
model = pkl.load(model_unpickle)

# Reassign target (if it was present) and predictions.
df["prediction"] = model.predict(df)
df["target"] = target

# Calculate accuracy of the model
correct = np.count_nonzero(np.where(df["prediction"]==df["target"],1,0))
total = df.shape[0]
accuracy = correct/total

print("accuracy is", accuracy)
