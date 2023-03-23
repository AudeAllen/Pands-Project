# https://www.kaggle.com/code/sixteenpython/machine-learning-with-iris-dataset/notebook

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read in CSV file
data = pd.read_csv('iris_csv.csv')
data.head()

print(data.shape)
data.describe   

# Output number of species
data.species.value_counts()

plt.hist(data["sepallength"])
plt.show()
plt.savefig('sepallength.png') # Save to PNG file

plt.hist(data["sepalwidth"])
plt.show()
plt.savefig('sepalwidth.png') # Save to PNG file

plt.hist(data["petallength"])
plt.show()
plt.savefig('petallength.png') # Save to PNG file

plt.hist(data["petalwidth"])
plt.show()
plt.savefig('petalwidth.png') # Save to PNG file

