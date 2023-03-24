# https://www.kaggle.com/code/sixteenpython/machine-learning-with-iris-dataset/notebook

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read in CSV file
data = pd.read_csv('iris_csv.csv')
data.head()

# Print the number of rows and columns in dataset
print(data.shape)

#Print summary of each variable using pandas describe function
df = pd.DataFrame(data)
data.describe()
print(df.describe())

#Write pandas summary to text file
IRIS_df = (df.describe())
IRIS_df.to_string('IrisVariableSummary.txt')


# Output number of species
data.species.value_counts()



#Output histogram of each variable to .png file 
#sepallength
x = (data["sepallength"])
plt.hist(data["sepallength"])
plt.title("Histogram of Variable Sepal Length - Iris Dataset")
plt.xlabel("Sepal Length")
plt.ylabel("Count")
plt.legend()
plt.hist(data["sepallength"], color='green')
plt.savefig('sepallength.png') # Save to PNG file
plt.show()

#sepalwidth
plt.hist(data["sepalwidth"])
plt.title("Histogram of Variable Sepal Width - Iris Dataset")
plt.xlabel("Sepal Width")
plt.ylabel("Count")
plt.legend()
plt.hist(data["sepalwidth"], color='skyblue')
plt.savefig('sepalwidth.png') # Save to PNG file
plt.show()

#petallength
plt.hist(data["petallength"])
plt.title("Histogram of Variable Petal Length - Iris Dataset")
plt.xlabel("Petal Length")
plt.ylabel("Count")
plt.legend()
plt.hist(data["petallength"], color='orange')
plt.savefig('petallength.png') # Save to PNG file
plt.show()

#petalwidth
plt.hist(data["petalwidth"])
plt.title("Histogram of Variable Petal Width - Iris Dataset")
plt.xlabel("Petal Width")
plt.ylabel("Count")
plt.legend()
plt.hist(data["petalwidth"], color='brown')
plt.savefig('petalwidth.png') # Save to PNG file
plt.show()
