# Logistic Regression on Customer satisfaction
This project revolves around try to understand what correlation and predictors can be used for a public dataset taken from Kaggle containing airline customer satastifaction information.
I will be reviewing some public data found that relates to airline passenger satisfaction from a dataset found from the resource Kaggle. I will be using a logistic regression model to determine which factors are most likely to lead to satisfaction and then build the model to be able to predict if new data was fed into the model, predict if the customer would be satisfied or not. The overall research question for this data is, which service quality factors most significantly predict passenger satisfaction in airline services, and to what extent do they influence the likelihood of a passenger being satisfied.

## Dataset Description
This dataset contains an airline passenger satisfaction survey. It contains data that is a high level view of non-identifable information of the customer and then some data that would contribute to understanding factors that may help understand what correlated to satisfaction.

The data has been pulled from a public source
https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction
into a CSV file format, I have chosen to use Python because of the number of rows of data exceeding 100,000+, whereas excel can handle this it’s already beginning to exceed a point where excel will handle it comfortably and I will notice overhead issues.

## Data Cleansing 
Little was required in terms of cleaning the data, mapping the dependent variables to boolean values (so changing Satisfied to a 1 and neutral or dissatisfied to a 0) was done to allow a logistic regression model to be performed on this dataset.
Columns such as Arrival/Depature delay are normalised, using StandardScaler, the data was normalised to create better performance in the model, because the ‘Delay in minutes’ columns had a wider range, normalising the data in these columns allows the data the to train faster because the data is easier to work as the scale has been reduced to closer to 0. This improves the models performance as the data is on a similar scale.

Null values were replaced on Delay columns with the mean to prevent the need to remove data from the datasets. Because of the low percentage of missing values, imputation is a sensible approach to replacing these missing values with the mean of those fields.

The data is split into 2 data files for training and testing the data, totaling up to be 125,000+ rows of data.
### Importing code and loading datasets
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
#Map the values for the dependent variables to Boolean values to be used later in the model
train_df['satisfaction'] = train_df['satisfaction'].map({'satisfied': 1, 'neutral or dissatisfied': 0})
test_df['satisfaction'] = test_df['satisfaction'].map({'satisfied': 1, 'neutral or dissatisfied': 0})

#Search for number null values
train_df.isnull().sum()
```


# URL Image Link
![Image URL](https://i0.wp.com/statisticsbyjim.com/wp-content/uploads/2020/07/TimeSeriesTrade.png?fit=576%2C384&ssl=1)

# URL Example

[URL Github](https://github.com/)

# Image from repo example

![Testing image folder](/asset/images/bus-road-against-sky.jpg)
