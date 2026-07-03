import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("cleaned_titanic.csv")
print("First 5 Rows:")
print(df.head())
print("\nDataset Shape:")
print(df.shape)
print("\nSummary Statistics:")
print(df.describe())
df["Survived"].value_counts().plot(kind="bar")
plt.title("Survival Count")
plt.xlabel("Survived")
plt.ylabel("Number of Passengers")
plt.show()