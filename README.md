# Logistic Regression Project 

In this project we will be working with an advertising data set, indicating whether or not a particular internet user clicked on an Advertisement. We will try to create a model that will predict whether or not they will click on an ad based off the features of that user.

This data set contains the following features:

* 'Daily Time Spent on Site': consumer time on site in minutes
* 'Age': cutomer age in years
* 'Area Income': Avg. Income of geographical area of consumer
* 'Daily Internet Usage': Avg. minutes a day consumer is on the internet
* 'Ad Topic Line': Headline of the advertisement
* 'City': City of consumer
* 'Male': Whether or not consumer was male
* 'Country': Country of consumer
* 'Timestamp': Time at which consumer clicked on Ad or closed window
* 'Clicked on Ad': 0 or 1 indicated clicking on Ad

## Project Source 
Kaggle

## Project Outcomes:
By the end of this project we would have sucessfully answered a few questions :

### *The Imports*
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

### Read in the advertising.csv file and set it to a data frame called ad_data.
```
ad_data = pd.read_csv("advertising.csv")
```

### Check the head of ad_data
```
ad_data.head()
```

### Use info and describe() on ad_data
```
ad_data.info()
ad_data.describe()
```

### *Exploratory Data Analysis*
#### Let's use seaborn to explore the data!

### Create a histogram of the Age
```
sns.set_style('whitegrid')
sns.histplot(x='Age',data=ad_data,bins=30,color='#21797E')
```

### Create a jointplot showing Area Income versus Age.
```
sns.jointplot(x='Area Income',y='Age',data=ad_data,color='#657E21')
```

### Create a jointplot showing the kde distributions of Daily Time spent on site vs. Age.
```
sns.jointplot(x='Daily Time Spent on Site',y='Age',data=ad_data,kind='kde',color='#EA3E17',fill=True)
```

### Create a jointplot of 'Daily Time Spent on Site' vs. 'Daily Internet Usage'
```
sns.jointplot(x='Daily Time Spent on Site',y='Daily Internet Usage',data=ad_data,color='green')
```

### create a pairplot with the hue defined by the 'Clicked on Ad' column feature.
```
sns.pairplot(data=ad_data,hue='Clicked on Ad',diag_kind='hist',palette='coolwarm')
```

### *Logistic Regression*

### Split the data into training set and testing set using train_test_split
```
ad_data.columns
X = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage','Male']]
y = ad_data['Clicked on Ad']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
```

### Train and fit a logistic regression model on the training set.
```
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
```

### *Predictions and Evaluations*
### Now predict values for the testing data.
```
predictions = logmodel.predict(X_test)
```

### Create a classification report for the model.
```
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))
```
### Project Setup:
To clone this repository you need to have Python compiler installed on your system alongside pandas and seaborn libraries. I would rather suggest that you download jupyter notebook if you've not already.

To access all of the files I recommend you fork this repo and then clone it locally. Instructions on how to do this can be found here: https://help.github.com/en/github/getting-started-with-github/fork-a-repo

The other option is to click the green "clone or download" button and then click "Download ZIP". You then should extract all of the files to the location you want to edit your code.

Installing Jupyter Notebook: https://jupyter.readthedocs.io/en/latest/install.html </br>
Installing Pandas library: https://pandas.pydata.org/pandas-docs/stable/install.html









