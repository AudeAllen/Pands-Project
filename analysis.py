# References below
# https://www.kaggle.com/code/sixteenpython/machine-learning-with-iris-dataset/notebook
# https://www.geeksforgeeks.org/box-plot-and-histogram-exploration-on-iris-data/
# https://www.geeksforgeeks.org/exploratory-data-analysis-on-iris-dataset/
# https://rpubs.com/Karolina_G/848706

# This assignment required us to produce summary statistics for the whole dataset, as well as,
# per category of a categorical variable. 
# The dataset that I am going to analyse is the IRIS dataset.
# I will start by describing the variables and descriptive statistics related to them,
# and then I will present some insights about this dataset using graphs. My final steps in this analysis
# will include correlation and regression analysis.

#Author: Audrey Allen

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read in CSV file
data = pd.read_csv('iris_csv.csv')
data.head()

# Print the number of rows and columns in dataset
print(data.shape)

# Print summary of each variable using pandas describe function
# This is an overall summary of all variables

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

plt.figure(figsize = (10, 7))
x = data["sepallength"]  
plt.hist(x, bins = 20, color = "green")
plt.title("Histogram of Variable Sepal Length - Iris Dataset")
plt.title("Sepal Length in cm")
plt.xlabel("Sepal_Length_cm")
plt.ylabel("Count")
plt.savefig('sepallength.png') # Save to PNG file
plt.show()

#sepalwidth
plt.figure(figsize = (10, 7))
x = data["sepalwidth"]  
plt.hist(x, bins = 20, color = "skyblue")
plt.title("Histogram of Variable Sepal Width - Iris Dataset")
plt.title("Sepal Width in cm")
plt.xlabel("Sepal_Width_cm")
plt.ylabel("Count")
plt.savefig('sepalwidth.png') # Save to PNG file
plt.show()

#petallength
plt.figure(figsize = (10, 7))
x = data["petallength"]  
plt.hist(x, bins = 20, color = "yellow")
plt.title("Histogram of Variable Petal Length - Iris Dataset")
plt.title("Petal Length in cm")
plt.xlabel("Petal_Length_cm")
plt.ylabel("Count")
plt.savefig('Petallength.png') # Save to PNG file
plt.show()

#petalwidth
plt.figure(figsize = (10, 7))
x = data["petalwidth"]  
plt.hist(x, bins = 20, color = "brown")
plt.title("Histogram of Variable Petal Width - Iris Dataset")
plt.title("Petal Width in cm")
plt.xlabel("Petal_Width_cm")
plt.ylabel("Count")
plt.savefig('Petalwidth.png') # Save to PNG file
plt.show()


#boxpplot sepallength
sns.boxplot( x=data["species"], y=data["sepallength"], palette="Blues") .set(title='BoxPlot Sepal Length');
plt.show()

#boxpplot sepalwidth
sns.boxplot( x=data["species"], y=data["sepalwidth"], palette="pastel") .set(title='BoxPlot Sepal Width');
plt.show()

#boxpplot petallength
sns.boxplot( x=data["species"], y=data["petallength"], palette="deep") .set(title='BoxPlot Petal Length');
plt.show()

#boxpplot petalwidth
sns.boxplot( x=data["species"], y=data["petalwidth"], palette="muted") .set(title='BoxPlot Petal Width');
plt.show()