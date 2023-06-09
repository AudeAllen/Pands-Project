# References below
# https://www.kaggle.com/code/sixteenpython/machine-learning-with-iris-dataset/notebook
# https://www.geeksforgeeks.org/box-plot-and-histogram-exploration-on-iris-data/
# https://www.geeksforgeeks.org/exploratory-data-analysis-on-iris-dataset/
# https://rpubs.com/Karolina_G/848706
# https://vitalflux.com/python-creating-scatter-plot-with-iris-dataset/

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

print (data.head())

# Print the number of rows and columns in dataset
print(data.shape)

# The below print statement groups the overall summary statistics for the four variables
#  for all of the species of Iris plant   
df = pd.DataFrame(data)
data.describe()

 
#Write both summaries to two separate text files
IRIS_df = (df.describe())
IRIS_df.to_string('OverallIrisVariableSummary.txt')

# The below print statement groups the summary statistics for each species/class of iris plant         

IRIS_df_all = (df.groupby("species").describe())
IRIS_df_all.to_string('SummarybySpeciesIrisVariable.txt')
    

# Output number of species
data.species.value_counts()

#Output histogram of each variable to .png file 
#sepallength  
#Each species displays in a different colour as it is then clearer to the species which is linearly separable from the other two 
#species - Iris-setosa is linearly separable from both Iris-versicolor and Iris-virginica


#sepallength
plt.figure(figsize = (10, 7))
x =[data.sepallength[data.species==sp_name] for sp_name in ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']]
plt.hist(x,20, histtype='bar', stacked=True)
plt.legend(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
plt.title("Sepal Length in cm")
plt.xlabel("Sepal_Length_cm")
plt.ylabel("Count")
plt.savefig('sepallengthhist.png') # Save to PNG file
plt.show()



#sepalwidth
plt.figure(figsize = (10, 7))
x =[data.sepalwidth[data.species==sp_name] for sp_name in ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']]
plt.hist(x,20, histtype='bar', stacked=True)
plt.legend(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
plt.title("Sepal Width in cm")
plt.xlabel("Sepal_Width_cm")
plt.ylabel("Count")
plt.savefig('sepalwidthhist.png') # Save to PNG file
plt.show()



#petallength
plt.figure(figsize = (10, 7))
x =[data.petallength[data.species==sp_name] for sp_name in ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']]
plt.hist(x,20, histtype='bar', stacked=True)
plt.legend(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
plt.title("Petal Length in cm")
plt.xlabel("Petal_Length_cm")
plt.ylabel("Count")
plt.savefig('petallengthhist.png') # Save to PNG file
plt.show()



#petalwidth 
plt.figure(figsize = (10, 7))
x =[data.petalwidth[data.species==sp_name] for sp_name in ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']]
plt.hist(x,20, histtype='bar', stacked=True)
plt.legend(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
plt.title("Petal Width in cm")
plt.xlabel("Petal_Width_cm")
plt.ylabel("Count")
plt.savefig('petalwidthhist.png') # Save to PNG file
plt.show()




#boxpplot sepallength
sns.boxplot( x=data["species"], y=data["sepallength"], palette="Blues") .set(title='BoxPlot Sepal Length');
plt.savefig('sepallengthboxplot.png') # Save to PNG file
plt.show()

#boxpplot sepalwidth
sns.boxplot( x=data["species"], y=data["sepalwidth"], palette="Blues") .set(title='BoxPlot Sepal Width');
plt.savefig('sepalwidthboxplot.png') # Save to PNG file
plt.show()

#boxpplot petallength
sns.boxplot( x=data["species"], y=data["petallength"], palette="Blues") .set(title='BoxPlot Petal Length');
plt.savefig('petallengthboxplot.png') # Save to PNG file
plt.show()

#boxpplot petalwidth
sns.boxplot( x=data["species"], y=data["petalwidth"], palette="Blues") .set(title='BoxPlot Petal Width');
plt.savefig('petalwidthboxplot.png') # Save to PNG file 
plt.show()


# Pairplot of all variables in the Iris Dataset

sns.pairplot(data, hue="species")
plt.savefig('Scatterplot.png') # Save to PNG file 
plt.show()

# Correlation between all four variables on the Iris Dataset

data.groupby("species").corr()
CorrelationMatrix = data.groupby("species").corr()
print(CorrelationMatrix)
CorrelationMatrix.to_string('CorrelationMatrix.txt')