# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

Step 1: Data Import and Initial Inspection

Load the dataset:

Read the dataset from a CSV file (Employee (1).csv) using pandas read_csv function.

Inspect the first few rows using head() to understand the structure of the data.

Check data information using info() to understand the column types and check for missing values.

Count the number of missing values in each column using isnull().sum().

Check the distribution of the target variable (left) using value_counts() to understand the class balance.

Step 2: Data Preprocessing

Label Encoding:

Import the LabelEncoder class from sklearn.preprocessing to convert categorical variables into numeric values.

Use LabelEncoder to encode the salary column into numeric values (e.g., "low", "medium", "high" → 0, 1, 2).

Step 3: Feature Selection

Select Input Features:

Create a new DataFrame x containing the independent variables (features) that will be used to predict employee attrition (e.g., satisfaction_level, last_evaluation, number_project, etc.).

Define Target Variable:

Create the target variable y, which is the column left representing employee attrition (1 if left, 0 if stayed).

Step 4: Train-Test Split

Split the Data:

Split the dataset into training and testing sets using the train_test_split function.

Set aside 20% of the data for testing (test_size=0.2), and use 80% for training.

Set a random_state for reproducibility.

Step 5: Model Training

Initialize the Decision Tree Model:

Import DecisionTreeClassifier from sklearn.tree.

Initialize the model with criterion='entropy' to use entropy as the criterion for splitting nodes in the tree.

Train the Model:

Use the fit() function to train the decision tree model on the training data (x_train, y_train).

Step 6: Model Evaluation

Make Predictions:

Use the trained model to make predictions on the test set (x_test) using the predict() function.

Evaluate Model Accuracy:

Import metrics from sklearn and calculate the model accuracy by comparing the predicted values (y_predict) with the actual test values (y_test) using accuracy_score().

Step 7: Predict New Data

Make Predictions for New Observations:

Use the trained model to predict whether an employee with given feature values will leave or stay.

For example, use the model to predict for a new employee with the feature values [0.5, 0.8, 9, 260, 6, 0, 1, 2].



## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: SRIDHARAN J
RegisterNumber:212222040158  
*/
import pandas as pd
data = pd.read_csv("Employee (1).csv")
data.head()

data.info()
data.isnull().sum()
data['left'].value_counts()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data['salary'] = le.fit_transform(data['salary'])
data.head()

x=data[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','salary']]
x.head()

y=data['left']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state =100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(x_train,y_train)
y_predict=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_predict)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
DATA HEAD:

![Screenshot 2025-04-30 140613](https://github.com/user-attachments/assets/6937dc9f-b01a-4634-b1bc-49bc3203a418)


DATASET INFO:

![Screenshot 2025-04-30 140623](https://github.com/user-attachments/assets/0ce82750-4b95-4cf7-9ccd-76f495b7995a)


NULL DATASET:

![Screenshot 2025-04-30 140630](https://github.com/user-attachments/assets/6138cbcc-0ec7-4f86-85a5-66027b573dcf)


VALUES COUNT IN LEFT COLUMN:

![Screenshot 2025-04-30 140637](https://github.com/user-attachments/assets/a0337214-4874-429f-bc23-2cbb8b7e0256)


DATASET TRANSFORMED HEAD:

![Screenshot 2025-04-30 140705](https://github.com/user-attachments/assets/507fe53e-ddde-4f78-8400-2d46aa346104)


X.HEAD:

![Screenshot 2025-04-30 140715](https://github.com/user-attachments/assets/aa6c485d-dcfc-4583-ae2f-5aa59905a707)


ACCURACY:

![Screenshot 2025-04-30 140723](https://github.com/user-attachments/assets/078ca9b2-9754-4ad3-ad0a-5f563c1f2443)


DATA PREDICTION:

![Screenshot 2025-04-30 140730](https://github.com/user-attachments/assets/b0129c65-f28a-4e9a-853c-e0702088cd0e)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
