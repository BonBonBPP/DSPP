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
```
Unnamed: 0                             0
id                                     0
Gender                                 0
Customer Type                          0
Age                                    0
Type of Travel                         0
Class                                  0
Flight Distance                        0
Inflight wifi service                  0
Departure/Arrival time convenient      0
Ease of Online booking                 0
Gate location                          0
Food and drink                         0
Online boarding                        0
Seat comfort                           0
Inflight entertainment                 0
On-board service                       0
Leg room service                       0
Baggage handling                       0
Checkin service                        0
Inflight service                       0
Cleanliness                            0
Departure Delay in Minutes             0
Arrival Delay in Minutes             310
satisfaction                           0
dtype: int64
```

```
test_df.isnull().sum()
```
```
Unnamed: 0                            0
id                                    0
Gender                                0
Customer Type                         0
Age                                   0
Type of Travel                        0
Class                                 0
Flight Distance                       0
Inflight wifi service                 0
Departure/Arrival time convenient     0
Ease of Online booking                0
Gate location                         0
Food and drink                        0
Online boarding                       0
Seat comfort                          0
Inflight entertainment                0
On-board service                      0
Leg room service                      0
Baggage handling                      0
Checkin service                       0
Inflight service                      0
Cleanliness                           0
Departure Delay in Minutes            0
Arrival Delay in Minutes             83
satisfaction                          0
dtype: int64
```


### Normalising Data
Using StandardScaler, the data was normalised to create better performance in the model, because the ‘Delay in minutes’ columns had a wider range, normalising the data in these columns allows the data the to train faster because the data is easier to work as the scale has been reduced to closer to 0. This improves the models performance as the data is on a similar scale.
```
#Replace null values using imputation for mean values of the relevant fields.
train_df['Arrival Delay in Minutes'].fillna(train_df['Arrival Delay in Minutes'].mean(), inplace=True)
train_df['Departure Delay in Minutes'].fillna(train_df['Departure Delay in Minutes'].mean(), inplace=True)
test_df['Arrival Delay in Minutes'].fillna(test_df['Arrival Delay in Minutes'].mean(), inplace=True)
test_df['Departure Delay in Minutes'].fillna(test_df['Departure Delay in Minutes'].mean(), inplace=True)

# Initialise the scaler
scaler = StandardScaler()

#Normalise the numerical features in the training data set 
train_df[['Arrival Delay in Minutes', 'Departure Delay in Minutes']] = scaler.fit_transform(train_df[['Arrival Delay in Minutes', 'Departure Delay in Minutes']])
# Normalise the numerical features in the testing data set
test_df[['Arrival Delay in Minutes', 'Departure Delay in Minutes']] = scaler.transform(test_df[['Arrival Delay in Minutes', 'Departure Delay in Minutes']])
```
## Exploratory Data Analysis

### Data suitablity
Checking the suitabilty of the data in terms of the balance of what we are predicting in logistic regression is important as imbalanced classes in dependent variables can lead to several issues, including the risk of overfitting. However, overfitting itself is not the most immediate risk; rather, the primary concerns are:

Bias towards the majority class: The model may become biased towards predicting the majority class, leading to poor performance on the minority class. This means the model might predict "satisfied" for most cases, even when the true label is "dissatisfied."

Poor generalisation: The model may not generalize well to new data, particularly for the minority class. This happens because the model doesn't learn enough about the minority class to make accurate predictions.
The training data is measured on the left and the test data is measured on the right.

![Dependent variable distribution (training data)](/asset/images/SatisfiedDistri.png) ![Dependent variable distribution (test data)](/asset/images/SatisfiedDistri2.png)

the fairly equal spread suggests this data will be suitable to create a model against and there should be a lower risk of bias or poor generalisation.

### Understanding spread of data in independent variables
Viewing the spread of the independent variables helps us to understand the likely potential predictators in our model.

Gender:
Almost equal distribution between male (49%) and female (51%).

Customer Type:
Majority are loyal customers (82%) compared to disloyal customers (18%).

Type of Travel:
Higher proportion of business travel (69%) than personal travel (31%).

Class:
Predominantly business class (48%), followed by economy class (45%), and a small portion in economy plus (7%).

Inflight Wifi Service:
Ratings are fairly distributed with higher counts in mid-ranges (2, 3, 4) and fewer in extremes (1, 5).

Departure/Arrival Time Convenient:
Similar distribution across ratings with slightly fewer extremes.

Ease of Online Booking:
More ratings in the mid-range (2, 3, 4) with fewer extreme ratings.

Gate Location:
Again, higher distribution in the mid-ranges (2, 3, 4).

Food and Drink:
Mid-range ratings are more common, with fewer extremes.

Online Boarding:
Significant portion of high ratings (4, 5), indicating satisfaction with this service.

Seat Comfort:
Mixed distribution with no clear concentration in any specific rating.

Inflight Entertainment:
More mid-range ratings with fewer extremes.

On-Board Service:
Evenly spread across mid-range ratings, with fewer extremes.

Leg Room Service:
Similar mid-range distribution with fewer extremes.

Baggage Handling:
Higher satisfaction (rating 5) is noticeable compared to other ratings.

Check-In Service:
Predominantly higher ratings (4, 5), indicating satisfaction.

Inflight Service:
Higher ratings (4, 5) suggest good service quality.

Cleanliness:
More concentrated in the higher ratings (4, 5), indicating cleanliness satisfaction.

### Data Distribution:

The data is fairly balanced in terms of gender, with noticeable skewness towards loyal customers and business travel.
Service ratings generally show a higher frequency of mid-range values, indicating moderate satisfaction levels.

#### Potential Predictors:

Customer type and class may be strong predictors due to the clear majority in certain categories.
Service ratings like online boarding, inflight service, check-in service, and cleanliness might be significant predictors given their higher satisfaction levels.



![Independent variable distribution](/asset/images/FIeldGraphicals.png)
