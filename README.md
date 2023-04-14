Analysis on the Iris Dataset 
1.	Introduction to Project

Problem Statement –
1. Research the iris data set online and write a summary about it in your README. 
2. Download the data set and add it to your repository. 
3. Write a program called analysis.py that: 
1. Outputs a summary of each variable to a single text file,
2. Saves a histogram of each variable to png files, and 
3. Outputs a scatter plot of each pair of variables. 
4. Performs any other analysis you think is appropriate 

I am going to be analysing the very famous dataset called the Iris dataset. The iris dataset was created by the british statistician and biologist Ronald Fisher. The dataset contains four features (length and width of sepals and petals) of 50 samples of three species of Iris (Iris Setosa, Iris Virginica and Iris Versicolor).	
I am going to be using python code and explanations of this code to discuss this dataset. The explanations and analyses will be in my README file with code examples. Python pandas will be used as Pandas has many functions that can help analyse this dataset. There will be many other functions and libraries within Python that I will be using in order to show my findings in this project.  I will be discussing these later on in my project.
This assignment will firstly produce summary statistics for the whole dataset using the Pandas dataframe function. A dataframe is a 2-dimensional labeled data structure with columns of potentially different types. It is a simple table with rows and columns. The iris dataset file will be read in from a .csv file. As well as a summary file I will be producing summary statistics per category of a categorical variable.
I am going to start by summerising the variables and descriptive statistics related to them. I will then present some insights about this dataset using a number of graphs. 
I will also include correlation and regression analysis in my analysis.	
I am going to be using Visual Studio Code to write my python code and readme files. I had some trouble downloading Jupyter Notebook so I will be doing all of my commenting and descriptions of my code and graphs in my README file.
The main python script is called analysis.py. 
I hope my analysis of this dataset will help give me more insight and knowledge of python and also machine learning.




2.	Software needed to run the project
-	Visual Studio Code
-	Python 3
-	Github - https://github.com/AudeAllen/Pands-Project.git
How to run Python code 
import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

Need to fill out a bit more – discussion about pandas, matplotlib, seaborn 

3.	Background on Iris Dataset
There is a lot of information and analyses done on the Iris dataset. The British statistician Ronald Fisher published the dataset in 1936.
As mentioned in my introduction iris dataset gives the measurements in centimeters of the variables sepal length and width and petal length and width, respectively, for 50 flowers from each of 3 species of iris. The species are Iris setosa, versicolor, and virginica. 	
This dataset is a simple dataset which contains 5 variables and thus I can demonstrate my knowledge of many data science concepts using python and other tools.
In this project we will see various features of the dataset including data distribution, correlations between certain features and the presence of outliers.
What we will see from my analysis is that one species of iris (Iris Setosa) is linearly separable from the other two. The other two classes are not linearly separable. Linearly separable applies to a set of points. Two sets are said to be linearly separable if a line can be drawn that separates the points such that a set points is on one side of the line and another set of points is on the other side of the line.
See pictures of each species of flower below.

 	


Basic Analysis using Panda Dataframe
First thing I need to do is read in the Iris dataset. It is in .csv format. The tool I am using for this is Visual studio code. All of  my code and files will be uploaded to github after I have made any changes.
data = pd.read_csv('iris_csv.csv')
data.head()
To run my python program go into Visual Studio Code and go to the directory C:\Users\audreyallen\Desktop\Pands1\Pands-Project and type in py analysis.py.
Using the Pandas dataframe describe method I can display the overall summary statistics for all four variables for all species of the Iris plant.  The describe () method returns 	statistics on the numerical data in the .csv file.
df = pd.DataFrame(data)
data.describe()
The summary is output to a text file called ‘OverallIrisVariableSummary.txt’
IRIS_df = (df.describe())
IRIS_df.to_string('OverallIrisVariableSummary.txt')
Summary Columns returned
count - The number of not-empty values
mean - The average (mean) value
std - The standard deviation
min - the minimum value
25% - The 25% percentile - how many of the values are less than the given percentile – 25%
50% - The 50% percentile - how many of the values are less than the given percentile – 50%
75% - The 75% percentile - how many of the values are less than the given percentile – 75%
max - the maximum value


Summary Observations on Overall Summary Statistics
 

-	Count - There are no null values as the count shows 150 of each variable
-	Mean - The Mean of the sepallength is greater than the mean of the other three variables. The petalwidth column has the lowest average measurements
-	Std -The standard deviation is a quanity expressing how much a group differs from the mean of the group – The higher the standard deviation the more variation or spread in the data. In the summary above we can see that the variable petallength has the highest standard deviation at 1.76. This means that there is the most deviation from the mean with the variable petallength than any of the other 3 variables.  The closer a the standard deviation is to 1 means it is closer to the mean. Sepallength is the closet to 1 at 0.82 in the summary set above and therefore shows that the values of sepallength are clustered around the mean. (5.84).  Petalwidth is also close to 1 which shows that this varaibel does not vary hugely from the mean. The standard deviation for sepalwidth shows that there a values that are lower than the mean at 0.43. We can assume that sepalwidth gives us less information.
-	Min – The variable petalwidth has the lowest value out of the four variables at 0.1. Sepallength has the highest minimum value out of all variables.
-	25% - This signifies the percentage of values that are 25% below that value. The sepallength has the highest value for this statistic at 5.8 and interestingly for all the other centile statistics (50% and 75%). Petalwidth has the lowest values for the 3 centile statistics at 0.3 for 25%, 1.3 at 50% and 1.8 at 75%. 
-	50% - This is the median value. The median should be close to the mean value. If it is not then it signifies that there  maybe outliers and that the data is skewed.   Petalength has the largest difference between mean and median at 0.57. This is evident in the fact that petallength also has the highest standard deviation so the spread of the data for this variable varies the most.
-	75% - 	This is sometimes known as the uperquartile range and signifies the percentage of values that are 75% below that value and 25% below that value.
-	Max – Sepallength has the largest measurement for all four species at 7.9. Petalwidth has the lowest at 2.5. Again what we can see from the data is that the variable with the largest difference between the min and max value is again petallength which ties into the standard deviation being so high.
	sepallength	Sepalwidth	petallength	petallength
Min	4.3	2.0	1.0	0.10
Max	7.9	4.4	6.9	2.5
Difference	3.6	2.4	5.9	2.4
 	
Summary Observations grouped by Species
Using Pandas dataframe again this summary by species and variable can be completed. The dataframe is grouped by species and using the describe  method again the summary statistics are output. The output of the summary statistics grouped by species is output to a  text file called ‘SummarySpeciesIrisVariable.txt’. See below for sample code and output.
IRIS_df_all = (df.groupby("species").describe())
IRIS_df_all.to_string('SummarybySpeciesIrisVariable.txt')

Output
Ouput of the group decribe method is below. For each variable sepallength, sepalwidth, petallength and petalwidth count, mean, std, min, 25%, 50%, 75% and max statistics are calculated. I am going to briefly go through the summary table but I think looking at graphs and visual aids might might more sense when trying to analyse the data.
 

From looking at the summary table I have a couple of observations.
-	All statistics calculated for the Iris-Setosa (mean, std, min, 25%, 50%, 75%, max ) are lower than for the other two species versicolor and virginica for 3 variables sepallength, petallength and petalwidth. Iris-Setosas mean, std, min, 25%, 50%, 75% and max values are higher only for sepalwidth. 
-	Iris-virginica species is characterized by the highest dispersion of all variables except for sepal width.
-	The summary table above is very informative but I am going to display some graphs and visuals in order to show what the summary table does not show clearly – similarities between the species and marked differences.


Histogram of each variable for each species











	













References

http://rstudio-pubs-static.s3.amazonaws.com/450733_9a472ce9632f4ffbb2d6175aaaee5be6.html

https://levelup.gitconnected.com/unveiling-the-mysteries-of-the-iris-dataset-a-comprehensive-analysis-and-machine-learning-f5c4f9dbcd6d

https://www.geeksforgeeks.org/exploratory-data-analysis-on-iris-dataset/


https://kindsonthegenius.com/blog/what-is-a-linear-seperator-what-is-a-hyperplane-simple-and-brief-explanation/#:~:text=First%2C%20the%20concept%20of%20linear,other%20side%20of%20the%20line

https://rpubs.com/Karolina_G/848706  - show summary statistics and graphs






	

	 

















	




References

http://rstudio-pubs-static.s3.amazonaws.com/450733_9a472ce9632f4ffbb2d6175aaaee5be6.html



	

	 

