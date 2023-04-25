**Analysis on the Iris Dataset**

# Table of contents
* [1 Introduction Project](#1-introduction-project)
 	* [Problem Statement](#problem-statement)
* [2 Software needed to run the project](#2-software-needed-to-run-the-project)	


1 Introduction Project 
======
### ***Problem Statement***

1. Research the iris data set online and write a summary about it in your README. 
2. Download the data set and add it to your repository. 
3. Write a program called analysis.py that: 
1. Outputs a summary of each variable to a single text file,
2. Saves a histogram of each variable to png files, and 
3. Outputs a scatter plot of each pair of variables. 
4. Performs any other analysis you think is appropriate 

I am going to be analysing the very famous dataset called the Iris dataset. The iris dataset was created by the British statistician and biologist Ronald Fisher. The dataset contains four features (length and width of sepals and petals) of 50 samples of three species of Iris (Iris Setosa, Iris Virginica and Iris Versicolor).	
I am going to be using python code and explanations of this code to discuss this dataset. The explanations and analyses will be in my README file with code examples. Python pandas will be used as Pandas has many functions that can help analyse this dataset. There will be many other functions and libraries within Python that I will be using in order to show my findings in this project.  I will be discussing these later on in my project.
This assignment will firstly produce summary statistics for the whole dataset using the Pandas dataframe function. A dataframe is a 2-dimensional labelled data structure with columns of potentially different types. It is a simple table with rows and columns. The iris dataset file will be read in from a .csv file. As well as a summary file I will be producing summary statistics per category of a categorical variable.
I am going to start by summarising the variables and descriptive statistics related to them. I will then present some insights about this dataset using a number of graphs. 
I will also include correlation and regression analysis in my analysis.	
I am going to be using Visual Studio Code to write my python code and readme files. I had some trouble downloading Jupyter Notebook so I will be doing all of my commenting and descriptions of my code and graphs in my README file.
The main python script is called analysis.py. 
I hope my analysis of this dataset will help give me more insight and knowledge of python and also machine learning.



2 Software needed to run the project 

-	Visual Studio Code
-	Python 3
-	Github - https://github.com/AudeAllen/Pands-Project.git
How to run Python code 
import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

Need to fill out a bit more – discussion about pandas, matplotlib, seaborn 

3.	**Background on Iris Dataset**
There is a lot of information and analyses done on the Iris dataset. The British statistician Ronald Fisher published the dataset in 1936.
As mentioned in my introduction iris dataset gives the measurements in centimeters of the variables sepal length and width and petal length and width, respectively, for 50 flowers from each of 3 species of iris. The species are Iris setosa, versicolor, and virginica. 	
This dataset is a simple dataset which contains 5 variables and thus I can demonstrate my knowledge of many data science concepts using python and other tools.
In this project we will see various features of the dataset including data distribution, correlations between certain features and the presence of outliers.
What we will see from my analysis is that one species of iris (Iris Setosa) is linearly separable from the other two. The other two classes are not linearly separable. Linearly separable applies to a set of points. Two sets are said to be linearly separable if a line can be drawn that separates the points such that a set points is on one side of the line and another set of points is on the other side of the line.
See pictures of each species of flower below.
![image](https://user-images.githubusercontent.com/123590406/232835878-5c2764b0-5fec-416c-a7ac-4b81166d4d32.png)
 	


**Basic Analysis using Panda Dataframe**
First thing I need to do is read in the Iris dataset. It is in .csv format. The tool I am using for this is Visual studio code. All of my code and files will be uploaded to github after I have made any changes.
![image](https://user-images.githubusercontent.com/123590406/232836070-966b1276-e820-4725-9e54-5b6a0449e5b5.png)

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


**Summary Observations on Overall Summary Statistics**

![image](https://user-images.githubusercontent.com/123590406/232836232-908f2561-a136-4d65-827f-bf75eb76b442.png)


-	Count - There are no null values as the count shows 150 of each variable
-	Mean - The Mean of the sepallength is greater than the mean of the other three variables. The petalwidth column has the lowest average measurements
-	Std -The standard deviation is a quanity expressing how much a group differs from the mean of the group – The higher the standard deviation the more variation or spread in the data. In the summary above we can see that the variable petallength has the highest standard deviation at 1.76. This means that there is the most deviation from the mean with the variable petallength than any of the other 3 variables.  The closer a the standard deviation is to 1 means it is closer to the mean. Sepallength is the closet to 1 at 0.82 in the summary set above and therefore shows that the values of sepallength are clustered around the mean. (5.84).  Petalwidth is also close to 1 which shows that this varaibel does not vary hugely from the mean. The standard deviation for sepalwidth shows that there a values that are lower than the mean at 0.43. We can assume that sepalwidth gives us less information.
-	Min – The variable petalwidth has the lowest value out of the four variables at 0.1. Sepallength has the highest minimum value out of all variables.
-	25% - This signifies the percentage of values that are 25% below that value. The sepallength has the highest value for this statistic at 5.8 and interestingly for all the other centile statistics (50% and 75%). Petalwidth has the lowest values for the 3 centile statistics at 0.3 for 25%, 1.3 at 50% and 1.8 at 75%. 
-	50% - This is the median value. The median should be close to the mean value. If it is not then it signifies that there may be outliers and that the data is skewed.   Petalength has the largest difference between mean and median at 0.57. This is evident in the fact that petallength also has the highest standard deviation so the spread of the data for this variable varies the most.
-	75% - 	This is sometimes known as the uperquartile range and signifies the percentage of values that are 75% below that value and 25% below that value.
-	Max – Sepallength has the largest measurement for all four species at 7.9. Petalwidth has the lowest at 2.5. Again what we can see from the data is that the variable with the largest difference between the min and max value is again petallength which ties into the standard deviation being so high.

![image](https://user-images.githubusercontent.com/123590406/232836492-c260c432-d437-4c88-808d-a0f704537838.png)
 	
**Summary Observations grouped by Species**
Using Pandas dataframe again this summary by species and variable can be completed. The dataframe is grouped by species and using the describe method again the summary statistics are output. The output of the summary statistics grouped by species is output to a text file called ‘SummarySpeciesIrisVariable.txt’. See below for sample code and output.
IRIS_df_all = (df.groupby("species").describe())
IRIS_df_all.to_string('SummarybySpeciesIrisVariable.txt')

**Output**
Ouput of the group describe method is below. For each variable sepallength, sepalwidth, petallength and petalwidth count, mean, std, min, 25%, 50%, 75% and max statistics are calculated. I am going to briefly go through the summary table but I think looking at graphs and visual aids might might more sense when trying to analyse the data.

![image](https://user-images.githubusercontent.com/123590406/232836550-cf2e40d4-de5f-4861-a000-4dfbdbc5c342.png)
 

From looking at the summary table I have a couple of observations.
-	All statistics calculated for the Iris-Setosa (mean, std, min, 25%, 50%, 75%, max ) are lower than for the other two species versicolor and virginica for 3 variables sepallength, petallength and petalwidth. Iris-Setosas mean, std, min, 25%, 50%, 75% and max values are higher only for sepalwidth. 
-	Iris-virginica species is characterized by the highest dispersion of all variables except for sepal width.
-	The summary table above is very informative but I am going to display some graphs and visuals in order to show what the summary table does not show clearly – similarities between the species and marked differences.


**Analysis of Iris Dataset**

I am going to analyse the Iris dataset using two types of analysing data 			
1.	**Univariate** – Probably the simplest way to analyse data. Uni means only one variable is being analysed. It does not deal with causes or regression. This type of analyses summarizes the data and finds patterns in the data – For the univariate analysis in this project I will use histograms and boxplots to describe and summarize the data.

2.	**Bivariate** – Bivariate analysis means I will be using two variables to analyse a dataset. A scatterplot is a typical visualisation that would be used when doing bivariate analysis. Using a visualisation tool like a scatter plot we can see if there are any 	obvious relationship between different variables. 

**Univariate Analysis on the Iris Dataset**

**Histogram of each variable for each species**

I am going to analyse the data using histograms created using the Matplotlib library. A histogram can be used to illustrate the distribution of data in a visual form. It can be easy to see any unusual observations such as outliers and gaps.
I have created a histogram in python for each of the variables in the dataset.
-	Sepallength – Separated by colour to show the 3 species of Iris 
-	Sepalwidth – Separated by colour to show the 3 species of Iris
-	Petallength – Separated by colour to show the 3 species of Iris
-	Petalwidth – Separated by colour to show the 3 species of Iris
Plots can show how variables in the Iris data set relate to each other and trends and patterns that may indicate relationships between the variables. Histograms are a good representation of the distribution of data. The pandas hist function calls matplotlib.pyplot.hist which creates the histogram. I have created a separate histogram for each variable so we can see the overall distribution. I have also created the histograms to show the differences between the different species (Each species was a different colour). 



**Sepal Length Histogram** 

![image](https://user-images.githubusercontent.com/123590406/232836624-2c8775a0-09fd-4c7b-b63d-629ffd90a4f9.png)

 
**Observations**
The above histogram for sepal length shows the overall distribution of the variable sepal length across all species of Iris Flower. What we can see is that the range goes from the lowest value which is 4.3 to the highest value of 7.9 with Iris-Setosa having the lowest measurements for sepal length and iris-virginica having the highest. Both iris-versicolor and iris-virginica overlap on measurements and seem interlinked whereas iris-setosa is out on its own having much lower measurements.	


**Sepal Width Histogram **

![image](https://user-images.githubusercontent.com/123590406/232836736-2d07f115-1149-4bc5-90bc-2b7fe8da15bd.png)

 
**Observations**
The histogram above analyses the variable sepal width. Unlike sepal length the measurements are lower with a min and max of 2.0 and a max of 4.4. This time iris-setosa are showing the highest measurements with iris-versicolor showing the lowest between all three species.


**Petal Length Histogram** 
 
![image](https://user-images.githubusercontent.com/123590406/232836884-a74d0154-82f2-4474-b593-8d056e8febf1.png)

Observations
Iris-setosa is linearly separable from the other species when looking at the univariate analysis of petal length. There is a significant gap between the measurements for petal length for the iris-setosa and iris-versicolor and iris-virginica. The iris-Setosa species has the lowest measurements for petal measurement at 1.0 with iris-virginica having the max measurement for this variable at 6.9. There is some overlap between versicolor and virginica but iris–setosa is separated completely from the other two species.


**Petal Width Histogram **

![image](https://user-images.githubusercontent.com/123590406/232836909-7f5cb641-c280-435e-bebb-8091f67d4846.png)


**Observations**
Again the iris-setosa is linearly separable from the other two species in the above histogram. The iris-setosa again having the min measurement for petal width at 0.1 and virginica having the max measurement at 2.5. 
From the above graphical illustrations of the data using the four variables split by class we can ascertain that by analysing petal width and length we can separate at least the iris-setosa from the other two species.

Boxplot of each variable for each species
A boxplot can be used for either univariate or bivariate analysis.  For my analysis I am going to use the boxplot for univariate analysis on the same variables sepal length, sepal width, petal length and petal width. The boxplot is a standardized way of displaying the distribution of data based on a five number summary (“minimum”, first quartile (Q1), median, third quartile (Q3), and “maximum”. Boxplots can show your outliers and if your data is skewed.



**Boxplot Sepal Length**

![image](https://user-images.githubusercontent.com/123590406/232837024-5c928e8a-5917-4a70-bfe2-049115624349.png)
 
**Boxplot Sepal Width**

![image](https://user-images.githubusercontent.com/123590406/232837048-cc1b9629-01ce-4706-9b7a-6cbb81beb51c.png)
 
**Boxplot Petal Length** 

![image](https://user-images.githubusercontent.com/123590406/232837098-32b1cca7-7d3e-4486-b100-33d876ef6a0a.png)


**Boxplot Petal Width**
 
![image](https://user-images.githubusercontent.com/123590406/232837122-b00e4e01-72c6-4a89-a10f-cb11c7991660.png)


**Boxplot Observations**
From the above graphs we can see the iris-setosa has the lowest in all measurements except sepal width where it has the largest measurements. The little diamond shape outside the boxplot is an indicator of an outlier. There are not very many outliers so it won’t have a significant impact on my analysis. The Iris-setosa has the most outliers.

	

**Bivariate Analysis on the Iris Dataset**

**Scatterplots of the Iris Dataset**

A scatter plot is a useful plot as it visually shows how the different variables or features in the data set correlate with one another.  It is a graph of the ordered pairs of two variables. One variable is plotted on the x-axis while the other variable is plotted on the y-axis. A scatter plot can be used to visualize relationships between numerical variables such as the petal measurements and the sepal measurements in the Iris data set.
There is a function in the seaborn library called the Pairplot() function. This function plots pairwise relationships between variables with the use of scatterplots. It is a very useful visual to see correlations between certain variables. We can also see if the data is linearly separable which we know it is from previous discussions. 

![image](https://user-images.githubusercontent.com/123590406/232863012-8c12315e-d8ce-4e29-a637-23b398799a23.png)
As well as showing the visual of the Pairplot I have created a correlation matrix which shows the correlation between all four variables in the Iris Dataset. A correlation matrix involves a rows and columns table that shows the variables. Every cell in a matrix contains the correlation coefficient. 1 is considered a strong relationship between variables, 0 a neutral relationship and -1 is not a strong relationship. 
I think it would be useful to detail the correlation matrix in conjunction with the Pairplot.

The correlation matrix is contained in my analysis.py python program. See below. I have output the results table to a text file called CorrelationMatrix.txt


![image](https://user-images.githubusercontent.com/123590406/232863084-933b98de-ea95-4d59-bd68-4e7e55014977.png)


**Correlation Matrix**

![image](https://user-images.githubusercontent.com/123590406/232863145-d0696667-fbfe-4577-99ac-973a7a7e3958.png)

**Pair Plot **

The Pairplot has been output to a .png file called Scatterplot.png.  See below screenshot.

![image](https://user-images.githubusercontent.com/123590406/232863207-406bc692-f7d2-4214-840a-1159340c26c2.png)


**Observations of the Pairplot and Correlation Matrix**

A linearly separable data set is one where the observations or data points can be separated by a straight line drawn through the data. We can see that the Iris dataset is indeed linearly separable and that the iris-setosa is separable from the other two species.  However there is some overlap between iris-versicolor and iris-virginica. 
There is a strong correlation on the sepal length and the sepal width for the Iris-Setosa. This can be seen in the correlation matrix where the correlation coefficient is 0.74. This is not the case for the other two species where the correlation coefficient is 0.5 and 0.45 for both the versicolor and virginica respectively.

There is a strong correlation also between the sepal length and the petal length for the iris-virginica at 0.86 and also the Iris-versicolor at 0.75. There is also a strong correlation between the petal length and petal width for the iris- versicolor at 0.78. The least correlation is between petal length and sepal width for the iris-setosa at 0.17.
<<<<<<< HEAD


=======
>>>>>>> e7c6c75abacddf614441dea46225274f48ad8d7e












**References**

http://rstudio-pubs-static.s3.amazonaws.com/450733_9a472ce9632f4ffbb2d6175aaaee5be6.html

https://levelup.gitconnected.com/unveiling-the-mysteries-of-the-iris-dataset-a-comprehensive-analysis-and-machine-learning-f5c4f9dbcd6d

https://www.geeksforgeeks.org/exploratory-data-analysis-on-iris-dataset/


https://kindsonthegenius.com/blog/what-is-a-linear-seperator-what-is-a-hyperplane-simple-and-brief-explanation/#:~:text=First%2C%20the%20concept%20of%20linear,other%20side%20of%20the%20line

https://rpubs.com/Karolina_G/848706  - show summary statistics and graphs
https://levelup.gitconnected.com/unveiling-the-mysteries-of-the-iris-dataset-a-comprehensive-analysis-and-machine-learning-f5c4f9dbcd6d#:~:text=The%20histogram%20provides%20insights%20into,at%20around%200.2%20to%200.3

https://medium.com/@avulurivenkatasaireddy/exploratory-data-analysis-of-iris-data-set-using-python-823e54110d2d








	

	 



