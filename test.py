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


#sepallength
fig, ax = plt.subplots()
# plot histogram
ax.hist(data['sepallength'])
# set title and labels
plt.title("Sepal Length in cm")
plt.xlabel("Sepal_Length_cm")
plt.ylabel("Frequency")
plt.savefig('sepallengthhist.png') # Save to PNG file
plt.show()




