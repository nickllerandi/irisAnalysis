import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv("/Users/nicholasllerandi/Downloads/Iris.csv")
df.head()
df.describe()
df.groupby("class").size()
df.hist()

x = df[["petal length", "sepal length", "sepal width"]]
y = df["petal width"]
model = sm.OLS(y, x)
results = model.fit()
print(results.summary())